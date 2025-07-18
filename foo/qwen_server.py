import collections
import fcntl
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

from chatmux.convert import oai_to_qwen
from chatmux.openai import (
    ChatRequest,
    ChatResponse,
    CompletionChoice,
    Logprobs,
    ResponseMessage,
)
from nebu import (
    Bucket,
    ContainerConfig,
    Message,
    Processor,
    V1EnvVar,
    is_allowed,
    processor,
)
from nebu.config import GlobalConfig as NebuGlobalConfig
from orign import Adapter

setup_script = """
apt update
apt install -y git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip uninstall -y xformers
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install trl peft transformers bitsandbytes sentencepiece accelerate tiktoken qwen-vl-utils chatmux orign
pip install -e git+https://github.com/pbarker/unsloth-zoo.git#egg=unsloth_zoo
pip install -e git+https://github.com/pbarker/unsloth.git#egg=unsloth
"""

BASE_MODEL_ID = "agentsea/Qwen2.5-VL-32B-Instruct-CARL-Gflights4"
MAX_LOADED_ADAPTERS = 4

# --- LRU Disk Cache Management (Copied and Adapted from unsloth_trainer.py) ---
LRU_METADATA_FILE = (
    "/nebu/cache/lru_disk_cache_qwen_server.json"  # Server-specific metadata
)
ADAPTER_CACHE_DIR_BASE = "/nebu/cache/adapters"  # Shared with trainer
SFT_RUNS_DIR_BASE = (
    "./sft_runs_server_placeholder"  # Placeholder, server doesn't manage these directly
)
LRU_LOCK_FILE_QWEN = (
    "/nebu/cache/lru_disk_cache_qwen_server.lock"  # Server-specific lock file
)

DEFAULT_MAX_ADAPTER_STORAGE_MB = float(
    os.getenv("MAX_ADAPTER_STORAGE_MB", 10 * 1024)
)  # 10 GB
DEFAULT_MAX_SFTRUN_STORAGE_MB = float(
    os.getenv("MAX_SFTRUN_STORAGE_MB", 1 * 1024)
)  # 1 GB (low, as server shouldn't store these)

# Global variable to hold loaded metadata
_lru_disk_metadata = {"adapters": [], "sft_runs": []}

# Note on Race Conditions:
# If multiple processes (e.g., different server instances or a trainer and a server)
# share the same ADAPTER_CACHE_DIR_BASE and potentially the same LRU_METADATA_FILE,
# proper file locking (e.g., using fcntl or a library like flufl.lock) would be
# essential around _load_lru_metadata and _save_lru_metadata to prevent metadata corruption.
# Using a server-specific metadata file here avoids direct metadata clashes with the trainer
# but they will still compete for disk space in ADAPTER_CACHE_DIR_BASE.


def _acquire_lock(lock_file_path: str) -> Optional[int]:
    lock_fd = os.open(lock_file_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except BlockingIOError:
        os.close(lock_fd)
        return None


def _release_lock(lock_fd: Optional[int]):
    if lock_fd is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _get_dir_size_mb(path_str: str) -> float:
    """Calculates the total size of a directory in megabytes."""
    total_size_bytes = 0
    if not os.path.exists(path_str):
        return 0.0
    for dirpath, _, filenames in os.walk(path_str):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                try:
                    total_size_bytes += os.path.getsize(fp)
                except OSError:
                    pass  # Ignore if file is removed during scan
    return total_size_bytes / (1024 * 1024)


def _load_lru_metadata():
    """Loads LRU metadata from the JSON file, with file locking."""
    global _lru_disk_metadata

    lock_fd = _acquire_lock(LRU_LOCK_FILE_QWEN)
    if lock_fd is None:
        print(
            f"[LRU Disk Cache] Warning: Could not acquire lock for loading {LRU_METADATA_FILE}. Using in-memory or potentially stale data."
        )
        if not _lru_disk_metadata or (
            _lru_disk_metadata.get("adapters") == []
            and _lru_disk_metadata.get("sft_runs") == []
        ):
            print(
                "[LRU Disk Cache] No in-memory LRU data, and lock failed. Initializing fresh."
            )
            _lru_disk_metadata = {"adapters": [], "sft_runs": []}
        return

    try:
        os.makedirs(os.path.dirname(LRU_METADATA_FILE), exist_ok=True)
        if os.path.exists(LRU_METADATA_FILE):
            try:
                with open(LRU_METADATA_FILE, "r") as f:
                    content = f.read()
                    if content.strip():
                        _lru_disk_metadata = json.loads(content)
                    else:
                        _lru_disk_metadata = {
                            "adapters": [],
                            "sft_runs": [],
                        }  # Initialize if empty
                if "adapters" not in _lru_disk_metadata:
                    _lru_disk_metadata["adapters"] = []
                if "sft_runs" not in _lru_disk_metadata:
                    _lru_disk_metadata["sft_runs"] = []
                # print(f"[LRU Disk Cache] Metadata loaded from {LRU_METADATA_FILE}")
            except json.JSONDecodeError:
                print(
                    f"[LRU Disk Cache] Warning: Could not decode metadata from {LRU_METADATA_FILE}. Starting fresh."
                )
                _lru_disk_metadata = {"adapters": [], "sft_runs": []}
        else:
            # print(f"[LRU Disk Cache] Metadata file not found at {LRU_METADATA_FILE}. Initializing fresh.")
            _lru_disk_metadata = {"adapters": [], "sft_runs": []}
    finally:
        _release_lock(lock_fd)


def _update_item_access(item_name: str, item_type: str, item_path: str):
    """Atomically updates or adds an item's access time and size in the LRU metadata."""
    global _lru_disk_metadata

    lock_fd = _acquire_lock(LRU_LOCK_FILE_QWEN)
    if lock_fd is None:
        print(
            f"[LRU Disk Cache] Warning: Could not acquire lock for updating {item_name}. Update skipped."
        )
        return

    try:
        # Load the latest metadata from disk
        current_disk_metadata = {"adapters": [], "sft_runs": []}
        if os.path.exists(LRU_METADATA_FILE):
            try:
                with open(LRU_METADATA_FILE, "r") as f_read:
                    content = f_read.read()
                    if content.strip():
                        current_disk_metadata = json.loads(content)
                if "adapters" not in current_disk_metadata:
                    current_disk_metadata["adapters"] = []
                if "sft_runs" not in current_disk_metadata:
                    current_disk_metadata["sft_runs"] = []
            except json.JSONDecodeError:
                print(
                    f"[LRU Disk Cache] Warning: JSON decode error in _update_item_access for {item_name}. Initializing fresh."
                )
                current_disk_metadata = {"adapters": [], "sft_runs": []}  # Fallback

        _lru_disk_metadata = current_disk_metadata  # Operate on the fresh copy

        if item_type not in _lru_disk_metadata:
            _lru_disk_metadata[item_type] = []

        item_list = _lru_disk_metadata[item_type]
        found_item = None
        for item in item_list:
            if item.get("name") == item_name:
                found_item = item
                break

        current_size_mb = _get_dir_size_mb(item_path)
        current_time = int(time.time())

        if current_size_mb == 0 and not os.path.exists(item_path):
            if found_item:
                item_list.remove(found_item)
                print(
                    f"[LRU Disk Cache] Removed non-existent item '{item_name}' ({item_type}) from metadata."
                )
            # Save changes immediately
            with open(LRU_METADATA_FILE, "w") as f_write:
                json.dump(_lru_disk_metadata, f_write, indent=4)
            return

        if found_item:
            found_item["last_accessed_ts"] = current_time
            found_item["size_mb"] = current_size_mb
            found_item["path"] = item_path
            print(
                f"[LRU Disk Cache] Updated access for '{item_name}' ({item_type}), size: {current_size_mb:.2f} MB"
            )
        else:
            item_list.append(
                {
                    "name": item_name,
                    "path": item_path,
                    "size_mb": current_size_mb,
                    "last_accessed_ts": current_time,
                }
            )
            print(
                f"[LRU Disk Cache] Added new item to LRU: '{item_name}' ({item_type}), size: {current_size_mb:.2f} MB, path: {item_path}"
            )

        # Save changes immediately
        with open(LRU_METADATA_FILE, "w") as f_write:
            json.dump(_lru_disk_metadata, f_write, indent=4)

    except Exception as e:
        print(f"[LRU Disk Cache] Error during _update_item_access for {item_name}: {e}")
    finally:
        _release_lock(lock_fd)


def _ensure_storage_limits():
    """Ensures storage usage is within limits, evicting LRU items if necessary.
    This function performs an atomic read-modify-write on the metadata and FS.
    """
    global _lru_disk_metadata, state  # state is InferenceState for qwen_server

    lock_fd = _acquire_lock(LRU_LOCK_FILE_QWEN)
    if lock_fd is None:
        print(
            "[LRU Disk Cache] Warning: Could not acquire lock for _ensure_storage_limits. Operation skipped."
        )
        return

    try:
        # Load the very latest metadata from disk under lock
        current_disk_metadata = {"adapters": [], "sft_runs": []}
        if os.path.exists(LRU_METADATA_FILE):
            try:
                with open(LRU_METADATA_FILE, "r") as f_read:
                    content = f_read.read()
                    if content.strip():
                        current_disk_metadata = json.loads(content)
                if "adapters" not in current_disk_metadata:
                    current_disk_metadata["adapters"] = []
                if "sft_runs" not in current_disk_metadata:
                    current_disk_metadata["sft_runs"] = []
            except json.JSONDecodeError:
                print(
                    "[LRU Disk Cache] Warning: JSON decode error in _ensure_storage_limits. Initializing fresh."
                )
                current_disk_metadata = {"adapters": [], "sft_runs": []}

        _lru_disk_metadata = (
            current_disk_metadata  # Work with the fresh copy for this operation
        )

        max_adapter_storage_mb = float(
            os.getenv("MAX_ADAPTER_STORAGE_MB", DEFAULT_MAX_ADAPTER_STORAGE_MB)
        )
        max_sft_run_storage_mb = float(
            os.getenv("MAX_SFTRUN_STORAGE_MB", DEFAULT_MAX_SFTRUN_STORAGE_MB)
        )
        metadata_changed = False

        # --- Manage Adapters ---
        adapters = sorted(
            _lru_disk_metadata.get("adapters", []),
            key=lambda x: x.get("last_accessed_ts", 0),  # type: ignore
        )
        current_adapter_size_mb = sum(item.get("size_mb", 0) for item in adapters)
        # print(f"[LRU Disk Cache] Current adapter storage: {current_adapter_size_mb:.2f} MB / {max_adapter_storage_mb:.2f} MB limit.")

        adapters_to_keep = []
        evicted_count = 0
        for item in adapters:
            item_name = item.get("name")
            item_path = item.get("path")
            item_size = item.get("size_mb", 0)

            if (
                current_adapter_size_mb <= max_adapter_storage_mb
                and item_path
                and os.path.exists(item_path)
            ):
                adapters_to_keep.append(item)
                continue

            if not item_path or not os.path.exists(item_path):
                if item_path:  # if path was defined, its size contributed, so subtract
                    current_adapter_size_mb -= item_size
                print(
                    f"[LRU Disk Cache] Evicting (metadata only, path missing): Adapter '{item_name}' from {item_path}"
                )
                evicted_count += 1
                metadata_changed = True
                continue

            if current_adapter_size_mb > max_adapter_storage_mb:
                print(
                    f"[LRU Disk Cache] Evicting adapter '{item_name}' (size: {item_size:.2f} MB, path: {item_path})"
                )
                try:
                    shutil.rmtree(item_path)
                    current_adapter_size_mb -= item_size
                    evicted_count += 1
                    metadata_changed = True

                    if (
                        hasattr(state, "loaded_adapter_names_lru")
                        and item_name in state.loaded_adapter_names_lru
                    ):
                        state.loaded_adapter_names_lru.remove(item_name)

                    idx_to_remove = -1
                    for i, v1_adapter_obj in enumerate(state.adapters):
                        v1_full_name = f"{v1_adapter_obj.metadata.namespace}/{v1_adapter_obj.metadata.name}"
                        if item_name == v1_full_name or (
                            item_name == v1_adapter_obj.metadata.name
                            and "/" not in item_name
                        ):
                            idx_to_remove = i
                            break
                    if idx_to_remove != -1:
                        state.adapters.pop(idx_to_remove)

                    if (
                        hasattr(state, "base_model")
                        and hasattr(state.base_model, "peft_config")
                        and item_name in state.base_model.peft_config
                    ):
                        if (
                            state.base_model.active_adapter == item_name
                            or item_name in state.base_model.active_adapters
                        ):
                            state.base_model.set_adapter([])
                        state.base_model.delete_adapter(item_name)
                        _torch = globals().get("torch")
                        if (
                            _torch
                            and hasattr(_torch.cuda, "is_available")
                            and _torch.cuda.is_available()
                        ):
                            _torch.cuda.empty_cache()
                        _gc = globals().get("gc")
                        if _gc:
                            _gc.collect()
                except OSError as e:
                    print(
                        f"[LRU Disk Cache] Error deleting adapter directory {item_path}: {e}"
                    )
                    adapters_to_keep.append(item)
            else:  # Path exists, but we are now within budget
                adapters_to_keep.append(item)

        _lru_disk_metadata["adapters"] = sorted(
            adapters_to_keep,
            key=lambda x: x.get("last_accessed_ts", 0),  # type: ignore
            reverse=True,
        )
        if evicted_count > 0:
            print(
                f"[LRU Disk Cache] Evicted {evicted_count} adapters. Remaining adapter storage: {current_adapter_size_mb:.2f} MB."
            )

        # --- Manage SFT Runs (minimal for server) ---
        sft_runs = sorted(
            _lru_disk_metadata.get("sft_runs", []),
            key=lambda x: x.get("last_accessed_ts", 0),  # type: ignore
        )
        current_sft_run_size_mb = sum(item.get("size_mb", 0) for item in sft_runs)

        sft_runs_to_keep = []
        evicted_sft_count = 0
        for item in sft_runs:
            item_name = item.get("name")
            item_path = item.get("path")
            item_size = item.get("size_mb", 0)

            if (
                current_sft_run_size_mb <= max_sft_run_storage_mb
                and item_path
                and os.path.exists(item_path)
            ):
                sft_runs_to_keep.append(item)
                continue

            if not item_path or not os.path.exists(item_path):
                if item_path:
                    current_sft_run_size_mb -= item_size
                evicted_sft_count += 1
                metadata_changed = True
                continue

            if current_sft_run_size_mb > max_sft_run_storage_mb:
                try:
                    shutil.rmtree(item_path)
                    current_sft_run_size_mb -= item_size
                    evicted_sft_count += 1
                    metadata_changed = True
                except OSError:
                    sft_runs_to_keep.append(item)
            else:
                sft_runs_to_keep.append(item)

        _lru_disk_metadata["sft_runs"] = sorted(
            sft_runs_to_keep,
            key=lambda x: x.get("last_accessed_ts", 0),  # type: ignore
            reverse=True,
        )

        if metadata_changed or evicted_count > 0 or evicted_sft_count > 0:
            with open(LRU_METADATA_FILE, "w") as f_write:
                json.dump(_lru_disk_metadata, f_write, indent=4)
            # print(f"[LRU Disk Cache] Metadata (potentially) updated by _ensure_storage_limits.")

    except Exception as e:
        print(f"[LRU Disk Cache] Error during _ensure_storage_limits: {e}")
    finally:
        _release_lock(lock_fd)


# --- End LRU Disk Cache Management ---


def init():
    import gc
    import os

    from unsloth import FastVisionModel  # type: ignore # isort: skip
    import torch  # type: ignore
    from nebu import Cache  # type: ignore
    from orign import V1Adapter

    if "state" in globals():  # <-- already loaded by an earlier worker
        print("state already loaded by an earlier worker")
        return

    gc.collect()
    torch.cuda.empty_cache()

    # os.environ.setdefault("MAX_PIXELS", "100352")

    @dataclass
    class InferenceState:
        base_model: FastVisionModel
        model_processor: Any
        base_model_id: str
        adapters: List[V1Adapter]
        cache: Cache
        loaded_adapter_names_lru: collections.deque
        max_loaded_adapters: int

    try:
        print("loading model...")
        print("--- nvidia-smi before load ---")
        os.system("nvidia-smi")
        print("--- end nvidia-smi before load ---")
        time_start_load = time.time()
        base_model, model_processor = FastVisionModel.from_pretrained(
            BASE_MODEL_ID,
            load_in_4bit=False,
            # use_fast=True,
            dtype=torch.bfloat16,
            max_seq_length=32_768,
        )
        print(f"Loaded model in {time.time() - time_start_load} seconds")
        print("--- nvidia-smi after load ---")
        os.system("nvidia-smi")
        print("--- end nvidia-smi after load ---")

        global state
        state = InferenceState(
            base_model=base_model,
            model_processor=model_processor,
            base_model_id=BASE_MODEL_ID,
            adapters=[],
            cache=Cache(),
            loaded_adapter_names_lru=collections.deque(maxlen=MAX_LOADED_ADAPTERS),
            max_loaded_adapters=MAX_LOADED_ADAPTERS,
        )

        # --- LRU Disk Cache Init ---
        _load_lru_metadata()
        _ensure_storage_limits()  # Perform initial cleanup if needed
        # --- End LRU Disk Cache Init ---

    except Exception as e:
        print(f"[Smart Inference] Error during init: {e}")
        raise e


def smart_adapter_loading_for_inference(
    model: Any,
    adapter_to_load: Any,
    adapter_name: str,
    bucket: Any,
    loaded_adapters_list: List[Any],
) -> bool:
    """
    Smart adapter loading for inference that:
    1. Uses set_adapter() for fast switching to already-loaded adapters
    2. Uses hotswap_adapter() when updating existing adapters with new weights
    3. Uses traditional loading only for completely new adapters

    Returns True if adapter was loaded/updated, False if no action needed
    """
    from peft.utils.hotswap import hotswap_adapter  # type: ignore

    global state  # Access the global state for LRU cache

    print(f"\n[Smart Inference] Managing adapter: '{adapter_name}'")

    # --- LRU Cache Management --- Start ---
    print(
        f"[LRU Cache] Before managing '{adapter_name}': {list(state.loaded_adapter_names_lru)}"
    )
    print(
        f"[LRU Cache] Current state.adapters (V1Adapter objects): {[f'{a.metadata.namespace}/{a.metadata.name}' for a in state.adapters]}"
    )

    if adapter_name in state.loaded_adapter_names_lru:
        # Adapter name is in LRU, move it to MRU (right end of deque)
        state.loaded_adapter_names_lru.remove(adapter_name)
        state.loaded_adapter_names_lru.append(adapter_name)
        print(f"[LRU Cache] Moved '{adapter_name}' to MRU.")
    else:
        # Adapter name is new to the LRU cache
        if len(state.loaded_adapter_names_lru) >= state.max_loaded_adapters:
            # Cache is full, evict the LRU adapter name (left end of deque)
            lru_adapter_name_to_evict = state.loaded_adapter_names_lru.popleft()
            print(
                f"[LRU Cache] Cache full (max {state.max_loaded_adapters}). Evicting LRU adapter name: '{lru_adapter_name_to_evict}'"
            )

            # Also remove the corresponding V1Adapter object from state.adapters
            # and delete from PeftModel
            adapter_to_remove_from_state_list = None
            for i, v1_adapter in enumerate(state.adapters):
                # Construct the full name from V1Adapter to match the format in LRU deque
                # The adapter_name in LRU deque is `content.model` which can be `namespace/name` or just `name`
                # V1Adapter has metadata.namespace and metadata.name
                current_v1_adapter_full_name = (
                    f"{v1_adapter.metadata.namespace}/{v1_adapter.metadata.name}"
                )
                # Handle cases where adapter_name might not have a namespace (though server logic implies it does)
                # For simplicity, we assume lru_adapter_name_to_evict is in the same format as content.model (namespace/name or name)
                # If lru_adapter_name_to_evict was stored as just 'name', and V1Adapter has namespace, this match needs care.
                # Given server logic: model_parts = content.model.split("/"), name is model_parts[1] or model_parts[0]
                # and adapter_name is content.model. So it seems adapter_name should be consistent.
                # We will directly compare lru_adapter_name_to_evict with reconstructed names.

                # Attempt to match considering both `namespace/name` and just `name` from lru_adapter_name_to_evict
                # potential_match_names = [current_v1_adapter_full_name, v1_adapter.metadata.name] # This line is removed

                if lru_adapter_name_to_evict == current_v1_adapter_full_name or (
                    lru_adapter_name_to_evict == v1_adapter.metadata.name
                    and "/" not in lru_adapter_name_to_evict
                ):
                    print(
                        f"[LRU Cache] Found V1Adapter for eviction: {current_v1_adapter_full_name}"
                    )
                    adapter_to_remove_from_state_list = state.adapters.pop(i)
                    break

            if adapter_to_remove_from_state_list:
                print(
                    f"[LRU Cache] Removed V1Adapter '{adapter_to_remove_from_state_list.metadata.namespace}/{adapter_to_remove_from_state_list.metadata.name}' from state.adapters list."
                )
            else:
                print(
                    f"[LRU Cache] Warning: Could not find V1Adapter for '{lru_adapter_name_to_evict}' in state.adapters list for removal."
                )

            # Delete from PeftModel
            # The lru_adapter_name_to_evict is the key used in peft_config
            # We will directly compare lru_adapter_name_to_evict with reconstructed names.

            if (
                hasattr(model, "peft_config")
                and lru_adapter_name_to_evict in model.peft_config
            ):
                try:
                    model.delete_adapter(lru_adapter_name_to_evict)
                    print(
                        f"[LRU Cache] Successfully deleted adapter '{lru_adapter_name_to_evict}' from PeftModel."
                    )
                except Exception as e:
                    print(
                        f"[LRU Cache] Error deleting adapter '{lru_adapter_name_to_evict}' from PeftModel: {e}"
                    )
            else:
                print(
                    f"[LRU Cache] Adapter '{lru_adapter_name_to_evict}' was in LRU but not in PeftModel config. No deletion needed from model, or already deleted."
                )

        # Add the new adapter name to MRU (right end of deque)
        state.loaded_adapter_names_lru.append(adapter_name)
        print(f"[LRU Cache] Added new adapter name '{adapter_name}' to MRU.")

    print(
        f"[LRU Cache] After managing '{adapter_name}': {list(state.loaded_adapter_names_lru)}"
    )
    print(
        f"[LRU Cache] Current state.adapters (V1Adapter objects) after LRU: {[f'{a.metadata.namespace}/{a.metadata.name}' for a in state.adapters]}"
    )
    # --- LRU Cache Management --- End ---

    # Check current state in PeftModel (AFTER potential eviction by LRU)
    adapter_already_in_peft_model = (
        adapter_name in model.peft_config if hasattr(model, "peft_config") else False
    )

    # Find if we have this adapter tracked IN THE V1ADAPTER LIST (state.adapters)
    existing_adapter_info = None
    existing_adapter_index = -1

    # Iterate through the potentially modified state.adapters list
    for idx, loaded_v1_adapter_in_state in enumerate(state.adapters):
        if (
            loaded_v1_adapter_in_state.metadata.name == adapter_to_load.metadata.name
            and loaded_v1_adapter_in_state.metadata.namespace
            == adapter_to_load.metadata.namespace
        ):
            existing_adapter_info = loaded_v1_adapter_in_state
            existing_adapter_index = idx
            break

    print(
        f"[Smart Inference] Adapter '{adapter_name}' in PeftModel (after LRU): {adapter_already_in_peft_model}"
    )
    print(
        f"[Smart Inference] Have V1Adapter info in state.adapters: {existing_adapter_info is not None}"
    )

    if existing_adapter_info:
        if (
            existing_adapter_info.metadata.updated_at
            == adapter_to_load.metadata.updated_at
        ):
            # CASE 1: Exact same version tracked in state.adapters, and its name is in LRU.
            # If it's also in PeftModel, just set_adapter. If not (e.g., was evicted but is being re-requested immediately),
            # it needs to be loaded traditionally.
            if adapter_already_in_peft_model:
                print(
                    "[Smart Inference] FAST SWITCH: Exact version tracked and in PeftModel, using set_adapter()"
                )
                model.set_adapter(adapter_name)
                return False  # No loading needed, just switched.
            else:
                print(
                    "[Smart Inference] INFO: Exact version tracked but NOT in PeftModel (likely just evicted by LRU). Needs traditional load."
                )
                # This will proceed to the traditional load path if hotswap condition (adapter_already_in_peft_model) is false.
                # Ensure it falls through to _load_adapter_traditionally if needed.

        # Check if adapter_already_in_peft_model for hotswap. If not, it must be loaded traditionally.
        if (
            adapter_already_in_peft_model
            and existing_adapter_info.metadata.updated_at
            != adapter_to_load.metadata.updated_at
        ):
            # CASE 2: Different version in PeftModel (and tracked) - try hotswapping
            print(
                f"[Smart Inference] HOTSWAPPING: Updating '{adapter_name}' with newer weights"
            )
            print(f"  Current: updated_at={existing_adapter_info.metadata.updated_at}")
            print(f"  New: updated_at={adapter_to_load.metadata.updated_at}")

            try:
                # Download new weights to temporary location
                temp_adapter_path = f"./adapters/{adapter_name}_temp"
                print(
                    f"[Smart Inference] Downloading new weights to {temp_adapter_path}"
                )

                time_start_copy = time.time()
                bucket.copy(adapter_to_load.model_uri, temp_adapter_path)
                print(
                    f"[Smart Inference] Downloaded in {time.time() - time_start_copy} seconds"
                )

                # Use hotswap to update the existing adapter
                time_start_hotswap = time.time()

                # Hotswap requires an active adapter to replace
                if model.active_adapter != adapter_name:
                    print(
                        f"[Smart Inference] Setting adapter '{adapter_name}' as active before hotswap"
                    )
                    model.set_adapter(adapter_name)
                    print(
                        f"[Smart Inference] Active adapter now: {model.active_adapters}"
                    )

                hotswap_adapter(
                    model,
                    temp_adapter_path,
                    adapter_name=adapter_name,
                    torch_device="cuda",
                )
                print(
                    f"[Smart Inference] Hotswapped in {time.time() - time_start_hotswap} seconds"
                )

                # Update our tracking info
                loaded_adapters_list[existing_adapter_index] = adapter_to_load
                print(
                    f"[Smart Inference] Successfully hotswapped adapter '{adapter_name}'"
                )

                # Update LRU access for the hotswapped adapter (it's "used")
                # Its on-disk content managed by LRU might be the old version,
                # but its access time is updated.
                persistent_adapter_path = os.path.join(
                    ADAPTER_CACHE_DIR_BASE, adapter_name
                )
                _update_item_access(adapter_name, "adapters", persistent_adapter_path)

                # Clean up temp files
                import shutil

                shutil.rmtree(temp_adapter_path, ignore_errors=True)

                return True

            except Exception as e:
                print(
                    f"[Smart Inference] Hotswap failed: {e}, falling back to traditional reload"
                )
                # Fallback: delete and reload traditionally
                try:
                    model.delete_adapter(adapter_name)
                    del loaded_adapters_list[existing_adapter_index]
                except:  # noqa: E722
                    pass  # Best effort cleanup
                return _load_adapter_traditionally(
                    model, adapter_to_load, adapter_name, bucket
                )
        elif (
            not adapter_already_in_peft_model and existing_adapter_info
        ):  # Tracked but not in peft (e.g. evicted), or new version of tracked
            print(
                "[Smart Inference] RELOAD (Scenario A): V1Adapter info exists in state.adapters, but adapter not in PeftModel (or new version). Loading traditionally."
            )
            # If it was a different version, existing_adapter_info would be updated by _load_adapter_traditionally.
            # If same version but evicted, _load_adapter_traditionally reloads it.
            # Remove old V1Adapter from list if it was there, _load_adapter_traditionally will add the (potentially new) one.
            if existing_adapter_index != -1:
                del state.adapters[existing_adapter_index]
            return _load_adapter_traditionally(
                model, adapter_to_load, adapter_name, bucket
            )

    # CASE 3 & 4: Adapter not tracked in state.adapters OR new adapter entirely.
    # If adapter_already_in_peft_model is true here, it means it was in PeftModel but NOT in our state.adapters list (desync, should be rare).
    # We treat it as needing a fresh load to ensure state.adapters is correct.
    if not existing_adapter_info:
        print(
            f"[Smart Inference] NEW/UNTRACKED ADAPTER: Loading '{adapter_name}' traditionally. In PeftModel already? {adapter_already_in_peft_model}"
        )
        if adapter_already_in_peft_model:
            # This is a desync. PeftModel has it, but our state.adapters doesn't. To be safe, delete and reload.
            try:
                print(
                    f"[Smart Inference] Desync: Deleting '{adapter_name}' from PeftModel before fresh load to sync state.adapters."
                )
                model.delete_adapter(adapter_name)
            except Exception as e_del_desync:
                print(
                    f"[Smart Inference] Error deleting desynced adapter '{adapter_name}': {e_del_desync}"
                )
        return _load_adapter_traditionally(model, adapter_to_load, adapter_name, bucket)

    # Fallback / safety net: if existing_adapter_info was true, but it wasn't updated_at match, and not hotswappable (not in peft_model)
    # This path should ideally be covered by the `elif not adapter_already_in_peft_model and existing_adapter_info:` above.
    # If it gets here, it implies a state that should be a traditional load.
    print(
        f"[Smart Inference] Fallback: Defaulting to traditional load for '{adapter_name}'."
    )
    if (
        existing_adapter_index != -1
    ):  # Clean up old V1Adapter if present from state.adapters
        # Check if the V1Adapter object to be removed is actually in the list to prevent errors
        # This check is more robust if existing_adapter_index could be stale.
        # However, if existing_adapter_index is valid, direct pop is fine.
        if (
            existing_adapter_index < len(state.adapters)
            and state.adapters[existing_adapter_index].metadata.name
            == adapter_to_load.metadata.name
            and state.adapters[existing_adapter_index].metadata.namespace
            == adapter_to_load.metadata.namespace
        ):
            state.adapters.pop(existing_adapter_index)
        else:  # Defensive: re-find to be sure if index might be off due to prior list modification
            idx_to_pop = -1
            for i, adpt in enumerate(state.adapters):
                if (
                    adpt.metadata.name == adapter_to_load.metadata.name
                    and adpt.metadata.namespace == adapter_to_load.metadata.namespace
                ):
                    idx_to_pop = i
                    break
            if idx_to_pop != -1:
                state.adapters.pop(idx_to_pop)

    return _load_adapter_traditionally(model, adapter_to_load, adapter_name, bucket)


def _load_adapter_traditionally(
    model: Any,
    adapter_to_load: Any,  # This is the V1Adapter object
    adapter_name: str,  # This is the string name (e.g. namespace/name or just name)
    bucket: Any,
) -> bool:
    """Traditional adapter loading. Downloads to persistent cache, then loads."""
    print(
        f"[Traditional Inference] Loading adapter '{adapter_name}' (V1Adapter: {adapter_to_load.metadata.namespace}/{adapter_to_load.metadata.name})"
    )

    # Path for persistent storage in the LRU managed cache
    # adapter_name might be "namespace/name", os.path.join handles this correctly.
    persistent_adapter_path = os.path.join(ADAPTER_CACHE_DIR_BASE, adapter_name)

    # Ensure the base directory for this specific adapter exists
    os.makedirs(persistent_adapter_path, exist_ok=True)

    # Check if adapter already exists and is up-to-date (simplistic check for now)
    # A more robust check would involve comparing metadata or checksums if available.
    # For now, if called, we assume a need to download/refresh unless specific logic is added.
    # If we want to avoid re-download, we'd check existence & `adapter_config.json`
    # and then compare `adapter_to_load.metadata.updated_at` with a stored timestamp.
    # This example will proceed to download, assuming `_load_adapter_traditionally`
    # implies a need for fresh weights or it's the first time.

    print(
        f"[Traditional Inference] Downloading from {adapter_to_load.model_uri} to {persistent_adapter_path}"
    )
    time_start_copy = time.time()
    try:
        # Sync contents of model_uri INTO persistent_adapter_path
        # Before syncing, clear the target directory to ensure a clean state if it exists
        # This handles cases where a previous partial download or older version exists.
        if os.path.exists(persistent_adapter_path):
            # Be careful with rmtree on ADAPTER_CACHE_DIR_BASE itself
            for item in os.listdir(persistent_adapter_path):
                item_path = os.path.join(persistent_adapter_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        else:
            os.makedirs(
                persistent_adapter_path, exist_ok=True
            )  # Ensure it exists if it was cleared or never there

        bucket.copy(adapter_to_load.model_uri, persistent_adapter_path)
        print(
            f"[Traditional Inference] Downloaded in {time.time() - time_start_copy} seconds"
        )
    except Exception as e_download:
        print(
            f"[Traditional Inference] Error downloading adapter {adapter_name} from {adapter_to_load.model_uri}: {e_download}"
        )
        # Don't raise immediately, attempt cleanup of potentially partial persistent_adapter_path
        # if os.path.exists(persistent_adapter_path): # Risky to delete entire folder if it's shared
        #    pass # For now, rely on LRU to eventually clean it if it's broken.
        raise

    print(
        f"[Traditional Inference] Calling model.load_adapter(model_name_or_path='{persistent_adapter_path}', adapter_name='{adapter_name}')"
    )
    time_start_load = time.time()
    try:
        # Make sure the adapter is not already in the model with this name, delete if so to ensure clean load
        if hasattr(model, "peft_config") and adapter_name in model.peft_config:
            print(
                f"[Traditional Inference] Adapter '{adapter_name}' already in PeftModel. Deleting before reload."
            )
            model.delete_adapter(adapter_name)

        model.load_adapter(persistent_adapter_path, adapter_name=adapter_name)
        print(
            f"[Traditional Inference] Loaded in {time.time() - time_start_load} seconds"
        )
    except Exception as e_load:
        print(
            f"[Traditional Inference] Error loading adapter '{adapter_name}' from path '{persistent_adapter_path}': {e_load}"
        )
        # If load fails, the downloaded files at persistent_adapter_path might be corrupted or incomplete.
        # LRU will eventually clean it up if it's not usable.
        raise

    # Track the V1Adapter object in state.adapters
    existing_v1_idx = -1
    for i, v1_adapter in enumerate(state.adapters):
        if (
            v1_adapter.metadata.namespace == adapter_to_load.metadata.namespace
            and v1_adapter.metadata.name == adapter_to_load.metadata.name
        ):
            existing_v1_idx = i
            break
    if existing_v1_idx != -1:
        state.adapters.pop(existing_v1_idx)
    state.adapters.append(adapter_to_load)

    # Update LRU disk cache for the newly downloaded/loaded adapter
    _update_item_access(adapter_name, "adapters", persistent_adapter_path)

    print(
        f"[Traditional Inference] Successfully loaded, tracked, and LRU updated for adapter '{adapter_name}' at {persistent_adapter_path}"
    )
    # DO NOT clean up persistent_adapter_path here; LRU manager will handle it.
    return True


def infer_qwen_vl(
    message: Message[ChatRequest],
) -> ChatResponse:
    full_time = time.time()
    from qwen_vl_utils import process_vision_info  # type: ignore
    from unsloth import FastVisionModel  # type: ignore

    global state

    print("message", message)
    os.system("nvidia-smi")

    training_request = message.content
    if not training_request:
        raise ValueError("No training request provided")

    # print("content", message.content)

    container_config = ContainerConfig.from_env()
    print("container_config", container_config)

    content = message.content
    if not content:
        raise ValueError("No content provided")

    load_adapter = content.model != "" and content.model != BASE_MODEL_ID
    print("load_adapter", load_adapter)

    if load_adapter:
        adapter_hot_start = time.time()

        model_parts = content.model.split("/")
        if len(model_parts) == 2:
            namespace = model_parts[0]
            name = model_parts[1]
        else:
            namespace = message.handle
            name = model_parts[0]

        print("checking for adapter", f"'{namespace}/{name}'")
        adapters = Adapter.get(namespace=namespace, name=name, api_key=message.api_key)
        if adapters:
            adapter_to_load = adapters[0]
            print("found adapter info:", adapter_to_load)

            if not is_allowed(
                adapter_to_load.metadata.owner, message.user_id, message.orgs
            ):
                raise ValueError("You are not allowed to use this adapter")

            if not adapter_to_load.base_model == BASE_MODEL_ID:
                raise ValueError(
                    "The base model of the adapter does not match the model you are trying to use"
                )

            # Use smart adapter management instead of manual delete/reload logic
            bucket = Bucket()
            adapter_was_loaded = smart_adapter_loading_for_inference(
                state.base_model, adapter_to_load, content.model, bucket, state.adapters
            )

            if adapter_was_loaded:
                print(
                    f"Adapter {content.model} loaded/updated successfully via smart loading."
                )
            else:
                print(
                    f"Adapter {content.model} was already current - used fast switching"
                )

        else:
            raise ValueError(f"Adapter '{content.model}' not found")
        print("adapter loading/checking total time: ", time.time() - adapter_hot_start)

    # Ensure peft_config exists before trying to access keys
    loaded_adapter_names = []
    if hasattr(state.base_model, "peft_config"):
        loaded_adapter_names = list(state.base_model.peft_config.keys())
    print("loaded_adapter_names: ", loaded_adapter_names)

    # Adapter status logging
    print(f"Model to use: {content.model}, Intended load_adapter: {load_adapter}")
    active_before_manipulation = "N/A"
    try:
        # PeftModel.active_adapters is a property that returns a list
        active_before_manipulation = state.base_model.active_adapters
    except AttributeError:
        # Fallback for older PEFT or if it's just active_adapter (singular string)
        try:
            active_before_manipulation = state.base_model.active_adapter
        except AttributeError:
            pass
    except Exception as e_active:
        print(f"Note: Could not get active_adapters before manipulation: {e_active}")
        pass  # Potentially no adapters loaded yet or unexpected structure
    print(
        f"Active adapter(s) before explicit manipulation: {active_before_manipulation}"
    )

    if load_adapter:
        # Goal: Ensure ONLY content.model is active and enabled.
        target_adapter_name = content.model
        print(f"[AdapterCycle] Attempting to activate adapter: {target_adapter_name}")

        # 1. Ensure the target adapter is known to the model
        if target_adapter_name not in loaded_adapter_names:
            # This implies smart_adapter_loading_for_inference failed or was skipped, which shouldn't happen if we reached here with load_adapter=True
            raise RuntimeError(
                f"Adapter {target_adapter_name} was requested but is not in model.peft_config. Keys: {loaded_adapter_names}"
            )

        # 2. Disable all other adapters first (if possible and makes sense for Unsloth)
        # For PEFT, set_adapter typically handles this by making only the specified one active.
        # However, an explicit disable_adapters() might ensure a cleaner state for Unsloth.
        if hasattr(state.base_model, "disable_adapters"):
            print("[AdapterCycle] Calling disable_adapters() first for a clean slate.")
            state.base_model.disable_adapters()

        # 3. Set the desired adapter as active
        print(f"[AdapterCycle] Calling set_adapter('{target_adapter_name}')")
        state.base_model.set_adapter(target_adapter_name)
        # Update LRU access time for the adapter that was just set active
        persistent_adapter_path_active = os.path.join(
            ADAPTER_CACHE_DIR_BASE, target_adapter_name
        )
        if os.path.exists(
            persistent_adapter_path_active
        ):  # Only update if it's from our persistent cache
            _update_item_access(
                target_adapter_name, "adapters", persistent_adapter_path_active
            )

        # 4. Explicitly enable adapters (Unsloth-specific step, if necessary)
        # Standard PEFT's set_adapter already calls enable_adapter_layers.
        # This is more of a "belt and braces" for Unsloth.
        if hasattr(state.base_model, "enable_adapters"):
            print("[AdapterCycle] Calling enable_adapters() to ensure PEFT is active.")
            state.base_model.enable_adapters()
    else:
        # Goal: Ensure NO adapters are active for base model inference.
        print("[AdapterCycle] Deactivating all adapters for base model operation.")
        # Only attempt to disable/manipulate adapters if some have been loaded.
        adapters_present_in_config = False
        if hasattr(state.base_model, "peft_config") and state.base_model.peft_config:
            adapters_present_in_config = True
            print(
                f"[AdapterCycle] Adapters are present in peft_config: {list(state.base_model.peft_config.keys())}"
            )

        if adapters_present_in_config:
            if hasattr(state.base_model, "disable_adapters"):
                print("[AdapterCycle] Calling disable_adapters().")
                state.base_model.disable_adapters()
            elif hasattr(state.base_model, "set_adapter"):  # Robust PEFT way
                print("[AdapterCycle] Calling set_adapter([]).")
                state.base_model.set_adapter([])
            elif hasattr(state.base_model, "active_adapter"):  # Fallback
                print("[AdapterCycle] Setting active_adapter = None.")
                state.base_model.active_adapter = None
            else:
                print(
                    "[AdapterCycle] Warning: No standard method found to disable adapters, though adapters were present in config."
                )
        else:
            print(
                "[AdapterCycle] No adapters found in peft_config. Assuming base model is already effectively active. Skipping disable/set_adapter calls."
            )

    active_after_manipulation = "N/A"
    try:
        active_after_manipulation = state.base_model.active_adapters
    except AttributeError:
        try:
            active_after_manipulation = state.base_model.active_adapter
        except AttributeError:
            pass
    except Exception as e_active_after:
        print(
            f"Note: Could not get active_adapters after manipulation: {e_active_after}"
        )
        pass
    print(f"Active adapter(s) after explicit manipulation: {active_after_manipulation}")

    print("setting model for inference")  # This is a logging print statement
    FastVisionModel.for_inference(state.base_model)

    # Log active adapter state *after* for_inference as well, for debugging
    active_after_for_inference = "N/A"
    try:
        active_after_for_inference = state.base_model.active_adapters
    except AttributeError:
        try:
            active_after_for_inference = state.base_model.active_adapter
        except AttributeError:
            pass
    except Exception as e_active_final:
        print(
            f"Note: Could not get active_adapters after for_inference: {e_active_final}"
        )
        pass
    print(
        f"Active adapter(s) after FastVisionModel.for_inference(): {active_after_for_inference}"
    )

    content_dict = content.model_dump()
    messages_oai = content_dict["messages"]
    messages = oai_to_qwen(messages_oai)

    # Preparation for inference
    # print("preparing inputs using messages: ", messages)
    inputs_start = time.time()
    text = state.model_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print("text: ", text)
    # print("processing vision info: ", messages)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = state.model_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # print("inputs", inputs)
    print(f"Inputs prepared in {time.time() - inputs_start} seconds")

    # Inference: Generation of the output
    generation_start = time.time()
    generated_ids = state.base_model.generate(
        **inputs, max_new_tokens=content.max_tokens
    )
    print(f"Generation took {time.time() - generation_start} seconds")
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = state.model_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print("output_text", output_text)
    print(f"Generation with decoding took {time.time() - generation_start} seconds")

    # Build the Pydantic model, referencing your enumerations and classes
    response = ChatResponse(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        model=content.model,
        object="chat.completion",
        choices=[
            CompletionChoice(
                index=0,
                finish_reason="stop",
                message=ResponseMessage(  # type: ignore
                    role="assistant", content=output_text[0]
                ),
                logprobs=Logprobs(content=[]),
            )
        ],
        service_tier=None,
        system_fingerprint=None,
        usage=None,
    )
    print(f"Total time: {time.time() - full_time} seconds")

    return response


def QwenVLServer(
    platform: str = "runpod",
    accelerators: List[str] = ["1:A100_SXM"],
    model: str = "agentsea/Qwen2.5-VL-32B-Instruct-CARL-Gflights4",
    image: str = "public.ecr.aws/d8i6n0n1/orign/unsloth-server:8b0ee04",  # "us-docker.pkg.dev/agentsea-dev/orign/unsloth-infer:latest"
    namespace: Optional[str] = None,
    env: Optional[List[V1EnvVar]] = None,
    config: Optional[NebuGlobalConfig] = None,
    hot_reload: bool = True,
    debug: bool = False,
    min_replicas: int = 1,
    max_replicas: int = 4,
    name: Optional[str] = None,
    wait_for_healthy: bool = True,
) -> Processor[ChatRequest, ChatResponse]:
    if env:
        env.append(V1EnvVar(key="BASE_MODEL_ID", value=model))
    else:
        env = [
            V1EnvVar(key="BASE_MODEL_ID", value=model),
        ]
    decorate = processor(
        image=image,
        # setup_script=setup_script,
        accelerators=accelerators,
        platform=platform,
        init_func=init,
        env=env,
        namespace=namespace,
        config=config,
        hot_reload=hot_reload,
        debug=debug,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        name=name,
        wait_for_healthy=wait_for_healthy,
    )
    return decorate(infer_qwen_vl)
