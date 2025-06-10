import collections
import fcntl  # Added for file locking
import gc
import json
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nebu import Message, Processor, processor
from nebu.config import GlobalConfig as NebuGlobalConfig
from nebu.containers.models import V1EnvVar
from nebu.errors import RetriableError
from nebu.processors.models import (
    V1Scale,
    V1ScaleDown,
    V1ScaleUp,
    V1ScaleZero,
)
from orign import V1TrainingStatus, find_latest_checkpoint
from pydantic import BaseModel

BASE_MODEL_ID = "unsloth/Qwen2.5-VL-32B-Instruct"
ADAPTER_DIR = "/nebu/cache/adapters"
MAX_LOADED_ADAPTERS = 4

# --- LRU Disk Cache Management ---
LRU_METADATA_FILE = "/nebu/cache/lru_disk_cache.json"
ADAPTER_CACHE_DIR_BASE = ADAPTER_DIR
SFT_RUNS_DIR_BASE = "./runs"  # Relative to CWD
LRU_LOCK_FILE = "/nebu/cache/lru_disk_cache.lock"  # Lock file for metadata

DEFAULT_MAX_ADAPTER_STORAGE_MB = 100 * 1024  # 100 GB
DEFAULT_MAX_SFTRUN_STORAGE_MB = 100 * 1024  # 100 GB

# Global variable to hold loaded metadata
_lru_disk_metadata = {"adapters": [], "sft_runs": []}
_lock_fd = None  # Global to hold lock file descriptor if needed, though prefer local


def _acquire_lock(lock_file_path: str) -> Optional[int]:
    lock_fd = os.open(lock_file_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # print(f"Acquired lock on {lock_file_path}")
        return lock_fd
    except BlockingIOError:
        # print(f"Could not acquire lock on {lock_file_path}, already locked.")
        os.close(lock_fd)
        return None


def _release_lock(lock_fd: Optional[int]):
    if lock_fd is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        # print("Released lock")


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

    lock_fd = _acquire_lock(LRU_LOCK_FILE)
    if lock_fd is None:
        print(
            f"Warning: Could not acquire lock for loading {LRU_METADATA_FILE}. Using in-memory or potentially stale data."
        )
        # Decide on fallback: either use current _lru_disk_metadata or reset it.
        # For safety, if we can't lock to read, assume we don't know the true state.
        # However, an existing in-memory _lru_disk_metadata might be from a previous successful load.
        # Let's proceed with current in-memory state if lock fails, but log verbosely.
        if not _lru_disk_metadata or (
            _lru_disk_metadata.get("adapters") == []
            and _lru_disk_metadata.get("sft_runs") == []
        ):
            print(
                "No in-memory LRU data, and lock failed. Initializing fresh for safety."
            )
            _lru_disk_metadata = {"adapters": [], "sft_runs": []}
        return

    try:
        os.makedirs(os.path.dirname(LRU_METADATA_FILE), exist_ok=True)
        if os.path.exists(LRU_METADATA_FILE):
            try:
                with open(LRU_METADATA_FILE, "r") as f:
                    # Ensure file is not empty before loading
                    content = f.read()
                    if content.strip():
                        _lru_disk_metadata = json.loads(content)
                    else:
                        _lru_disk_metadata = {"adapters": [], "sft_runs": []}
                # Ensure basic structure
                if "adapters" not in _lru_disk_metadata:
                    _lru_disk_metadata["adapters"] = []
                if "sft_runs" not in _lru_disk_metadata:
                    _lru_disk_metadata["sft_runs"] = []
                # print(f"LRU metadata loaded from {LRU_METADATA_FILE}")
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode LRU metadata from {LRU_METADATA_FILE}. Starting fresh."
                )
                _lru_disk_metadata = {"adapters": [], "sft_runs": []}
        else:
            # print(f"LRU metadata file not found at {LRU_METADATA_FILE}. Initializing fresh.")
            _lru_disk_metadata = {"adapters": [], "sft_runs": []}
    finally:
        _release_lock(lock_fd)


def _update_item_access(item_name: str, item_type: str, item_path: str):
    """Updates or adds an item's access time and size in the LRU metadata.
    Locking is handled by _load_lru_metadata and _save_lru_metadata.
    This function assumes _lru_disk_metadata is the current, authoritative copy.
    """
    global _lru_disk_metadata

    # It's better to acquire a lock for the read-modify-write cycle here
    # if _load_lru_metadata itself isn't called immediately before.
    # Let's refine: _update_item_access should be atomic.

    lock_fd = _acquire_lock(LRU_LOCK_FILE)
    if lock_fd is None:
        print(
            f"Warning: Could not acquire lock for updating LRU item {item_name}. Update skipped."
        )
        return

    try:
        # Load the latest metadata from disk to avoid operating on a stale in-memory version
        # This mimics _load_lru_metadata's core file reading logic but within an existing lock
        temp_metadata = {"adapters": [], "sft_runs": []}  # Default if file is empty/new
        if os.path.exists(LRU_METADATA_FILE):
            try:
                with open(LRU_METADATA_FILE, "r") as f_read:
                    content = f_read.read()
                    if content.strip():  # Check if file is not empty
                        temp_metadata = json.loads(content)
                if "adapters" not in temp_metadata:
                    temp_metadata["adapters"] = []
                if "sft_runs" not in temp_metadata:
                    temp_metadata["sft_runs"] = []
            except json.JSONDecodeError:
                print(
                    f"Warning: JSON decode error during _update_item_access for {item_name}. Using fresh metadata for this op."
                )
                # If decode fails, we might overwrite. This is a risk.
                # Or, we could choose to not proceed with the update.
                # For now, let's proceed assuming we'll save a corrected version.
                temp_metadata = {"adapters": [], "sft_runs": []}  # Fallback

        _lru_disk_metadata = (
            temp_metadata  # Update global with what we just read under lock
        )

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

        if current_size_mb == 0 and not os.path.exists(
            item_path
        ):  # Item effectively gone
            if found_item:
                item_list.remove(found_item)
                print(
                    f"Removed non-existent item '{item_name}' ({item_type}) from LRU metadata."
                )
            # Save immediately after modification
            with open(LRU_METADATA_FILE, "w") as f_write:
                json.dump(_lru_disk_metadata, f_write, indent=4)
            return  # Lock will be released in finally

        if found_item:
            found_item["last_accessed_ts"] = current_time
            found_item["size_mb"] = current_size_mb
            found_item["path"] = item_path
            print(
                f"Updated LRU access for '{item_name}' ({item_type}), size: {current_size_mb:.2f} MB"
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
                f"Added new item to LRU: '{item_name}' ({item_type}), size: {current_size_mb:.2f} MB, path: {item_path}"
            )

        # Save immediately after modification
        with open(LRU_METADATA_FILE, "w") as f_write:
            json.dump(_lru_disk_metadata, f_write, indent=4)

    except Exception as e:
        print(f"Error during _update_item_access for {item_name}: {e}")
    finally:
        _release_lock(lock_fd)


def _ensure_storage_limits():
    """Ensures storage usage is within limits, evicting LRU items if necessary.
    This function performs a read-modify-write cycle on the metadata and file system.
    """
    global _lru_disk_metadata, state

    lock_fd = _acquire_lock(LRU_LOCK_FILE)
    if lock_fd is None:
        print(
            "Warning: Could not acquire lock for _ensure_storage_limits. Operation skipped."
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
                    "Warning: JSON decode error in _ensure_storage_limits. Initializing fresh for this op."
                )
                # Fallback to an empty structure if decode fails, to prevent operating on corrupted data
                current_disk_metadata = {"adapters": [], "sft_runs": []}

        _lru_disk_metadata = current_disk_metadata  # Work with the fresh copy

        max_adapter_storage_mb = float(
            os.getenv("MAX_ADAPTER_STORAGE_MB", DEFAULT_MAX_ADAPTER_STORAGE_MB)
        )
        max_sft_run_storage_mb = float(
            os.getenv("MAX_SFTRUN_STORAGE_MB", DEFAULT_MAX_SFTRUN_STORAGE_MB)
        )

        # --- Manage Adapters ---
        adapters = sorted(
            _lru_disk_metadata.get("adapters", []),
            key=lambda x: x.get("last_accessed_ts", 0),  # type: ignore
        )
        current_adapter_size_mb = sum(item.get("size_mb", 0) for item in adapters)
        # print(f"Current adapter storage: {current_adapter_size_mb:.2f} MB / {max_adapter_storage_mb:.2f} MB limit.")

        adapters_to_keep = []
        evicted_adapter_count = 0
        metadata_changed = False

        for item in adapters:
            if (
                current_adapter_size_mb <= max_adapter_storage_mb
                and item.get("path")
                and os.path.exists(item.get("path"))
            ):
                adapters_to_keep.append(item)
                continue

            item_path = item.get("path")
            item_name = item.get("name")
            item_size = item.get("size_mb", 0)

            if not item_path or not os.path.exists(item_path):
                if item_path:  # Only subtract size if it was supposed to be there
                    current_adapter_size_mb -= item_size
                print(
                    f"Evicting (metadata only, path missing): Adapter '{item_name}' from {item_path}"
                )
                evicted_adapter_count += 1
                metadata_changed = True
                continue

            if current_adapter_size_mb > max_adapter_storage_mb:
                print(
                    f"Evicting adapter '{item_name}' (size: {item_size:.2f} MB, path: {item_path}) to free up space."
                )
                try:
                    shutil.rmtree(item_path)
                    current_adapter_size_mb -= item_size
                    evicted_adapter_count += 1
                    metadata_changed = True

                    if (
                        hasattr(state, "loaded_adapter_names_lru")
                        and item_name in state.loaded_adapter_names_lru
                    ):
                        state.loaded_adapter_names_lru.remove(item_name)
                    if (
                        hasattr(state, "base_model")
                        and hasattr(state.base_model, "peft_config")
                        and item_name in state.base_model.peft_config
                    ):
                        state.base_model.delete_adapter(item_name)
                        _torch = globals().get("torch")
                        if (
                            _torch
                            and hasattr(_torch.cuda, "is_available")
                            and _torch.cuda.is_available()
                        ):
                            _torch.cuda.empty_cache()
                        gc.collect()
                except OSError as e:
                    print(f"Error deleting adapter directory {item_path}: {e}")
                    adapters_to_keep.append(item)  # Keep if deletion failed
            else:  # Path exists, but we are within budget now
                adapters_to_keep.append(item)

        _lru_disk_metadata["adapters"] = sorted(
            adapters_to_keep,
            key=lambda x: x.get("last_accessed_ts", 0),  # type: ignore
            reverse=True,
        )
        if evicted_adapter_count > 0:
            print(
                f"Evicted {evicted_adapter_count} adapters. Remaining adapter storage: {current_adapter_size_mb:.2f} MB."
            )

        # --- Manage SFT Runs ---
        sft_runs = sorted(
            _lru_disk_metadata.get("sft_runs", []),
            key=lambda x: x.get("last_accessed_ts", 0),  # type: ignore
        )
        current_sft_run_size_mb = sum(item.get("size_mb", 0) for item in sft_runs)
        # print(f"Current SFT run storage: {current_sft_run_size_mb:.2f} MB / {max_sft_run_storage_mb:.2f} MB limit.")

        sft_runs_to_keep = []
        evicted_sft_count = 0
        for item in sft_runs:
            if (
                current_sft_run_size_mb <= max_sft_run_storage_mb
                and item.get("path")
                and os.path.exists(item.get("path"))
            ):
                sft_runs_to_keep.append(item)
                continue

            item_path = item.get("path")
            item_name = item.get("name")
            item_size = item.get("size_mb", 0)

            if not item_path or not os.path.exists(item_path):
                if item_path:
                    current_sft_run_size_mb -= item_size
                print(
                    f"Evicting (metadata only, path missing): SFT run '{item_name}' from {item_path}"
                )
                evicted_sft_count += 1
                metadata_changed = True
                continue

            if current_sft_run_size_mb > max_sft_run_storage_mb:
                print(
                    f"Evicting SFT run '{item_name}' (size: {item_size:.2f} MB, path: {item_path}) to free up space."
                )
                try:
                    shutil.rmtree(item_path)
                    current_sft_run_size_mb -= item_size
                    evicted_sft_count += 1
                    metadata_changed = True
                except OSError as e:
                    print(f"Error deleting SFT run directory {item_path}: {e}")
                    sft_runs_to_keep.append(item)
            else:  # Path exists, but we are within budget
                sft_runs_to_keep.append(item)

        _lru_disk_metadata["sft_runs"] = sorted(
            sft_runs_to_keep,
            key=lambda x: x.get("last_accessed_ts", 0),  # type: ignore
            reverse=True,
        )
        if evicted_sft_count > 0:
            print(
                f"Evicted {evicted_sft_count} SFT runs. Remaining SFT run storage: {current_sft_run_size_mb:.2f} MB."
            )

        if (
            metadata_changed or evicted_adapter_count > 0 or evicted_sft_count > 0
        ):  # Save if any changes were made
            with open(LRU_METADATA_FILE, "w") as f_write:
                json.dump(_lru_disk_metadata, f_write, indent=4)
            # print(f"LRU metadata (potentially) updated by _ensure_storage_limits.")

    except Exception as e:
        print(f"Error during _ensure_storage_limits: {e}")
    finally:
        _release_lock(lock_fd)


# --- End LRU Disk Cache Management ---


class TrainingRequest(BaseModel):
    adapter: str
    dataset: str
    model: str = BASE_MODEL_ID
    max_length: int = 32_768
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5
    logging_steps: int = 1
    save_steps: int = 5
    lora_alpha: int = 128
    lora_rank: int = 64
    lora_dropout: float = 0
    optimizer: str = "adamw_8bit"
    owner: Optional[str] = None
    labels: Optional[Dict[str, str]] = None


class TrainingResponse(BaseModel):
    loss: float
    train_steps_per_second: float
    train_samples_per_second: float
    train_runtime: float
    adapter: str
    adapter_uri: str


scale = V1Scale(
    up=V1ScaleUp(above_pressure=10, duration="5m"),
    down=V1ScaleDown(below_pressure=2, duration="10m"),
    zero=V1ScaleZero(duration="10m"),
)

setup_script = """
apt update
apt install -y git
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install trl peft transformers bitsandbytes sentencepiece accelerate orign
pip install -e git+https://github.com/pbarker/unsloth-zoo.git#egg=unsloth_zoo
pip install unsloth
pip install -e git+https://github.com/pbarker/unsloth-zoo.git#egg=unsloth_zoo
pip install huggingface_hub[hf_xet]
"""


def init():
    import gc

    from unsloth import FastVisionModel  # type: ignore # isort: skip
    import torch  # type: ignore  # type: ignore
    from nebu import Cache  # type: ignore
    from orign import V1Adapter
    from peft import LoraConfig, PeftModel  # type: ignore  # noqa: F401
    from unsloth_zoo.peft_utils import get_peft_regex  # type: ignore

    if "state" in globals():  # <-- already loaded by an earlier worker
        print("state already loaded by an earlier worker")
        return

    gc.collect()
    torch.cuda.empty_cache()  # os.environ.setdefault("MAX_PIXELS", "100352")

    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # --- LRU Disk Cache Init ---
    _load_lru_metadata()
    _ensure_storage_limits()  # Perform initial cleanup if needed
    # --- End LRU Disk Cache Init ---

    @dataclass
    class TrainingState:
        base_model: FastVisionModel
        model_processor: Any
        base_model_id: str
        adapters: List[V1Adapter]
        cache: Cache
        loaded_adapter_names_lru: collections.deque  # LRU cache for adapter names
        max_loaded_adapters: int  # Max number of adapters to keep in memory

    print("Loading base model and tokenizer...")
    base_model, model_processor = FastVisionModel.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        max_seq_length=32_768,
        use_gradient_checkpointing="unsloth",
    )
    print("Base model and tokenizer loaded.")

    print("\nApplying initial PEFT setup with FastVisionModel.get_peft_model...")
    plumbed_model: PeftModel = FastVisionModel.get_peft_model(
        base_model,
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        finetune_vision_layers=True,
        finetune_language_layers=True,
        # target_modules is determined internally by Unsloth based on above flags
    )
    print(f"Type of model after get_peft_model: {type(plumbed_model)}")

    global G_INITIAL_TARGET_MODULES_PATTERN

    # --- Capture the target_modules from the "default" adapter ---
    if "default" in plumbed_model.peft_config:
        G_INITIAL_TARGET_MODULES_PATTERN = plumbed_model.peft_config[
            "default"
        ].target_modules
        print(
            "Captured initial target_modules pattern from 'default' adapter's config."
        )

        # Delete the default adapter since we'll manage our own adapters
        print("Deleting 'default' adapter created by get_peft_model.")
        plumbed_model.delete_adapter("default")
    else:
        print(
            "Warning: 'default' adapter not found. Attempting to generate target_modules pattern manually."
        )
        G_INITIAL_TARGET_MODULES_PATTERN = get_peft_regex(  # type: ignore
            base_model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )
        print("Generated initial target_modules pattern (fallback).")

    if G_INITIAL_TARGET_MODULES_PATTERN is None:
        raise RuntimeError(
            "Could not determine initial target_modules pattern. Aborting."
        )

    plumbed_model.active_adapter = None
    print(
        f"Initial target_modules pattern to be reused: '{str(G_INITIAL_TARGET_MODULES_PATTERN)[:200]}...'"
    )

    global state
    state = TrainingState(
        base_model=plumbed_model,
        model_processor=model_processor,
        base_model_id=BASE_MODEL_ID,
        adapters=[],
        cache=Cache(),
        loaded_adapter_names_lru=collections.deque(maxlen=MAX_LOADED_ADAPTERS),
        max_loaded_adapters=MAX_LOADED_ADAPTERS,
    )


def add_or_load_adapter_for_model(
    model: "PeftModel",  # type: ignore # noqa: F821
    adapter_name: str,
    resume_training: bool,
) -> str:
    """
    Smart adapter management that:
    1. Uses set_adapter() for fast switching to already-loaded adapters
    2. Uses traditional add_adapter() for new adapters
    3. Uses hotswap_adapter() only when updating existing adapters with new weights
    """
    from peft import LoraConfig  # type: ignore  # noqa: F401

    # --- LRU Cache Management ---
    print(
        f"[LRU Cache] Before managing '{adapter_name}': {list(state.loaded_adapter_names_lru)}"
    )

    if adapter_name in state.loaded_adapter_names_lru:
        # Adapter is already in LRU, move it to MRU (right end of deque)
        state.loaded_adapter_names_lru.remove(adapter_name)
        state.loaded_adapter_names_lru.append(adapter_name)
        print(f"[LRU Cache] Moved '{adapter_name}' to MRU.")
    else:
        # Adapter is new to the LRU cache
        if len(state.loaded_adapter_names_lru) >= state.max_loaded_adapters:
            # Cache is full, evict the LRU adapter (left end of deque)
            lru_adapter_to_evict = state.loaded_adapter_names_lru.popleft()
            print(
                f"[LRU Cache] Cache full. Evicting LRU adapter: '{lru_adapter_to_evict}'"
            )
            if lru_adapter_to_evict in model.peft_config:
                try:
                    model.delete_adapter(lru_adapter_to_evict)
                    print(
                        f"[LRU Cache] Successfully deleted adapter '{lru_adapter_to_evict}' from PeftModel."
                    )
                except Exception as e:
                    print(
                        f"[LRU Cache] Error deleting adapter '{lru_adapter_to_evict}' from PeftModel: {e}"
                    )
            else:
                print(
                    f"[LRU Cache] Adapter '{lru_adapter_to_evict}' was in LRU but not in PeftModel config. No deletion needed from model."
                )

        # Add the new adapter to MRU (right end of deque)
        state.loaded_adapter_names_lru.append(adapter_name)
        print(f"[LRU Cache] Added '{adapter_name}' to MRU.")

    print(
        f"[LRU Cache] After managing '{adapter_name}': {list(state.loaded_adapter_names_lru)}"
    )
    # --- End LRU Cache Management ---

    # Check hotswap availability with detailed debugging
    hotswap_available = False
    hotswap_adapter = None
    try:
        from peft.utils.hotswap import hotswap_adapter  # type: ignore

        hotswap_available = True
        print(
            f"[Smart Adapter] hotswap_adapter successfully imported: {hotswap_adapter}"
        )
    except ImportError as e:
        print(f"[Smart Adapter] hotswap_adapter import failed: {e}")
    except Exception as e:
        print(
            f"[Smart Adapter] Unexpected error importing hotswap_adapter: {type(e).__name__}: {e}"
        )

    print(f"[Smart Adapter] Hotswap functionality available: {hotswap_available}")

    global G_INITIAL_TARGET_MODULES_PATTERN
    print(
        f"\n[Smart Adapter] Managing adapter: '{adapter_name}', resume: {resume_training}"
    )

    adapter_base_folder = os.path.join(ADAPTER_DIR, adapter_name)
    os.makedirs(adapter_base_folder, exist_ok=True)
    path_containing_adapter_files = os.path.join(adapter_base_folder, adapter_name)

    # Check if adapter is already loaded
    adapter_already_loaded = adapter_name in model.peft_config
    has_saved_weights = os.path.isdir(path_containing_adapter_files) and os.listdir(
        path_containing_adapter_files
    )

    print(
        f"[Smart Adapter] Adapter '{adapter_name}' already loaded: {adapter_already_loaded}"
    )
    print(f"[Smart Adapter] Has saved weights: {has_saved_weights}")

    # Add more detailed state debugging
    if adapter_already_loaded:
        print(
            f"[Smart Adapter] Current adapter config for '{adapter_name}': {model.peft_config[adapter_name]}"
        )
        if hasattr(model.peft_config[adapter_name], "target_modules"):
            print(
                f"[Smart Adapter] Target modules: {model.peft_config[adapter_name].target_modules}"
            )

    if has_saved_weights:
        print(
            f"[Smart Adapter] Saved weight files: {os.listdir(path_containing_adapter_files)}"
        )
        # Check if required files exist
        config_file = os.path.join(path_containing_adapter_files, "adapter_config.json")
        weights_file = os.path.join(
            path_containing_adapter_files, "adapter_model.safetensors"
        )
        print(f"[Smart Adapter] Config file exists: {os.path.exists(config_file)}")
        print(f"[Smart Adapter] Weights file exists: {os.path.exists(weights_file)}")

        if os.path.exists(config_file):
            try:
                import json

                with open(config_file, "r") as f:
                    config_data = json.load(f)
                print(
                    f"[Smart Adapter] Saved config target_modules: {config_data.get('target_modules', 'not found')}"
                )
                print(
                    f"[Smart Adapter] Saved config r: {config_data.get('r', 'not found')}"
                )
                print(
                    f"[Smart Adapter] Saved config lora_alpha: {config_data.get('lora_alpha', 'not found')}"
                )
            except Exception as e:
                print(f"[Smart Adapter] Failed to read config file: {e}")

    print(f"[Smart Adapter] Resume training flag: {resume_training}")
    print(f"[Smart Adapter] Model class: {model.__class__.__name__}")
    print(f"[Smart Adapter] Model device: {getattr(model, 'device', 'no device attr')}")

    if adapter_already_loaded and has_saved_weights and resume_training:
        # CASE 1: Adapter is loaded but we have newer weights to hotswap in
        print(
            f"[Smart Adapter] HOTSWAPPING: Updating existing adapter '{adapter_name}' with new weights"
        )
        print(f"[Smart Adapter] Hotswap source path: {path_containing_adapter_files}")
        print(f"[Smart Adapter] Hotswap target adapter: {adapter_name}")
        print(f"[Smart Adapter] Model device: {getattr(model, 'device', 'unknown')}")
        print(f"[Smart Adapter] Model type: {type(model)}")
        print(
            f"[Smart Adapter] Available files at source: {os.listdir(path_containing_adapter_files) if os.path.exists(path_containing_adapter_files) else 'path does not exist'}"
        )

        if not hotswap_available or hotswap_adapter is None:
            print(
                "[Smart Adapter] Hotswap not available, falling back to traditional reload"
            )
            model.delete_adapter(adapter_name)
            return add_adapter_traditionally(
                model, adapter_name, resume_training, path_containing_adapter_files
            )

        try:
            # Add more detailed debugging before hotswap
            print(
                f"[Smart Adapter] Current adapter configs: {list(model.peft_config.keys())}"
            )
            print(f"[Smart Adapter] Current active adapters: {model.active_adapters}")

            # Hotswap requires an active adapter to replace
            if model.active_adapter != adapter_name:
                print(
                    f"[Smart Adapter] Setting adapter '{adapter_name}' as active before hotswap"
                )
                model.set_adapter(adapter_name)
                print(f"[Smart Adapter] Active adapter now: {model.active_adapters}")

            hotswap_start_time = time.time()
            hotswap_adapter(
                model,
                path_containing_adapter_files,
                adapter_name=adapter_name,
                torch_device="cuda" if hasattr(model, "device") else None,
            )
            hotswap_end_time = time.time()
            print(
                f"[Smart Adapter] Hotswap time taken: {hotswap_end_time - hotswap_start_time} seconds"
            )
            print(
                f"[Smart Adapter] Successfully hotswapped new weights into '{adapter_name}'"
            )
            _update_item_access(
                adapter_name, "adapters", os.path.join(ADAPTER_DIR, adapter_name)
            )
        except Exception as e:
            print(
                f"[Smart Adapter] Hotswap failed with detailed error: {type(e).__name__}: {str(e)}"
            )
            print("[Smart Adapter] Exception traceback:")
            import traceback

            traceback.print_exc()
            print(
                "[Smart Adapter] Falling back to traditional reload due to hotswap failure"
            )
            # Fallback: delete and reload traditionally
            model.delete_adapter(adapter_name)
            return add_adapter_traditionally(
                model, adapter_name, resume_training, path_containing_adapter_files
            )

    elif adapter_already_loaded:
        # CASE 2: Adapter is already loaded, just switch to it (fastest!)
        print(
            f"[Smart Adapter] FAST SWITCH: Adapter '{adapter_name}' already loaded, using set_adapter()"
        )

    else:
        # CASE 3: New adapter, load it traditionally
        print(f"[Smart Adapter] NEW ADAPTER: Loading '{adapter_name}' traditionally")
        return add_adapter_traditionally(
            model, adapter_name, resume_training, path_containing_adapter_files
        )

    # Set the adapter as active
    model.set_adapter(adapter_name)
    print(f"[Smart Adapter] Active adapter set to: '{model.active_adapters}'")
    _update_item_access(
        adapter_name, "adapters", os.path.join(ADAPTER_DIR, adapter_name)
    )
    return adapter_base_folder


def add_adapter_traditionally(
    model: "PeftModel",  # type: ignore # noqa: F821
    adapter_name: str,
    resume_training: bool,
    path_containing_adapter_files: str,
) -> str:
    """Traditional adapter loading for new adapters"""
    from peft import LoraConfig  # type: ignore

    global G_INITIAL_TARGET_MODULES_PATTERN

    print(f"[Traditional] Adding new adapter '{adapter_name}'")
    new_lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=1e-3,
        bias="none",
        target_modules=G_INITIAL_TARGET_MODULES_PATTERN,
    )

    try:
        model.add_adapter(adapter_name=adapter_name, peft_config=new_lora_config)
        print(f"[Traditional] Added adapter '{adapter_name}' successfully")
    except Exception as e:
        print(f"[Traditional] Error adding adapter '{adapter_name}': {e}")
        raise

    # Load weights if resuming and they exist
    if (
        resume_training
        and os.path.isdir(path_containing_adapter_files)
        and os.listdir(path_containing_adapter_files)
    ):
        print(
            f"[Traditional] Loading weights for '{adapter_name}' from {path_containing_adapter_files}"
        )
        try:
            model.load_adapter(
                path_containing_adapter_files, adapter_name, is_trainable=True
            )
            print(
                f"[Traditional] Successfully loaded weights for adapter '{adapter_name}'"
            )
        except Exception as e:
            print(f"[Traditional] Error loading weights: {e}")

    model.set_adapter(adapter_name)
    # Path for adapter directory
    adapter_disk_path = os.path.join(ADAPTER_DIR, adapter_name)
    _update_item_access(adapter_name, "adapters", adapter_disk_path)
    return os.path.dirname(path_containing_adapter_files)


def drop_adapter_from_model(model: "PeftModel", adapter_name_to_drop: str):  # type: ignore # noqa: F821
    import torch  # type: ignore

    global state  # Ensure we are using the global state

    print(f"\n[Adapter Management] Request to drop adapter: '{adapter_name_to_drop}'")

    # Remove from LRU cache if present
    if adapter_name_to_drop in state.loaded_adapter_names_lru:
        state.loaded_adapter_names_lru.remove(adapter_name_to_drop)
        print(f"[LRU Cache] Removed '{adapter_name_to_drop}' from LRU cache.")
    else:
        print(f"[LRU Cache] Adapter '{adapter_name_to_drop}' not found in LRU cache.")

    # Deactivate and delete from PeftModel
    if adapter_name_to_drop in model.peft_config:
        print(
            f"[Adapter Management] Deleting adapter '{adapter_name_to_drop}' from PeftModel."
        )
        if model.active_adapter == adapter_name_to_drop:
            model.active_adapter = None  # Deactivate if it was active
            print(
                f"[Adapter Management] Deactivated active adapter '{adapter_name_to_drop}'."
            )
        try:
            model.delete_adapter(adapter_name_to_drop)
            print(
                f"[Adapter Management] Successfully deleted '{adapter_name_to_drop}' from PeftModel."
            )
        except Exception as e:
            print(
                f"[Adapter Management] Error deleting adapter '{adapter_name_to_drop}' from PeftModel: {e}"
            )
    else:
        print(
            f"[Adapter Management] Adapter '{adapter_name_to_drop}' not found in loaded PeftModel adapters. No deletion needed from model."
        )

    torch.cuda.empty_cache()
    gc.collect()
    print(
        f"[Adapter Management] Finished dropping adapter '{adapter_name_to_drop}'. LRU: {list(state.loaded_adapter_names_lru)}\n"
    )


def train_lora_adapter(
    adapter_name_to_train: str,
    training_dataset: Any,
    num_epochs: int = 1,
    resume_from_saved_state: bool = False,
    checkpoint_path: Optional[str] = None,
):
    import json  # Added for reading trainer_state.json
    import os  # ensure os is imported locally if not already

    import torch  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore
    from unsloth import FastVisionModel, is_bf16_supported  # type: ignore
    from unsloth.trainer import UnslothVisionDataCollator  # type: ignore

    global state

    print(
        f"\n--- Starting train_lora_adapter for adapter: '{adapter_name_to_train}' (Epochs: {num_epochs}, Resume: {resume_from_saved_state}) ---"
    )

    os.system("nvidia-smi")

    # Use smart adapter management
    start_adapter_time = time.time()
    adapter_base_save_folder = add_or_load_adapter_for_model(
        state.base_model,
        adapter_name_to_train,
        resume_from_saved_state or bool(checkpoint_path),
    )
    end_adapter_time = time.time()
    print(
        f"Time taken to load adapter: {end_adapter_time - start_adapter_time} seconds"
    )
    print(
        f"Adapter base save folder for '{adapter_name_to_train}': {adapter_base_save_folder}"
    )

    loaded_adapter_names = []
    if hasattr(state.base_model, "peft_config"):
        loaded_adapter_names = list(state.base_model.peft_config.keys())
    print("loaded_adapter_names: ", loaded_adapter_names)

    # The smart management ensures the correct adapter is active
    print(f"Training will use adapter: '{adapter_name_to_train}'")
    print(f"Currently active adapters: {state.base_model.active_adapters}")

    print("\nPreparing model for training with FastVisionModel.for_training...")
    model_ready_for_training = FastVisionModel.for_training(state.base_model)
    print("Model prepared for training.")

    print("\nModel's trainable parameters (on instance passed to SFTTrainer):")
    try:
        model_ready_for_training.print_trainable_parameters()
    except AttributeError:
        total_params = sum(p.numel() for p in model_ready_for_training.parameters())
        trainable_params = sum(
            p.numel() for p in model_ready_for_training.parameters() if p.requires_grad
        )
        print(
            f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}"
        )

    initial_learning_rate = 4e-4  # 5e-5
    print(f"Using initial learning_rate for SFTTrainer: {initial_learning_rate}")

    sft_config_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=num_epochs,
        learning_rate=initial_learning_rate,
        optim="adamw_8bit",
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        save_strategy="epoch",
        save_total_limit=1,
        output_dir=f"./runs/{adapter_name_to_train}",
        logging_steps=1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model_ready_for_training,
        tokenizer=state.model_processor,
        data_collator=UnslothVisionDataCollator(
            model_ready_for_training, state.model_processor, resize="max"
        ),
        train_dataset=training_dataset,
        args=sft_config_args,
    )

    # For SFTTrainer's resume_from_checkpoint:
    # If checkpoint_path is explicitly given, use it.
    # Else, if resume_from_saved_state is True, pass True to trainer.train() to load latest from output_dir.
    # Otherwise, no resume.
    sft_trainer_resume_arg = None
    if checkpoint_path:
        sft_trainer_resume_arg = checkpoint_path
        print(
            f"SFTTrainer will attempt to resume from EXPLICIT checkpoint path: '{sft_trainer_resume_arg}'"
        )
    elif resume_from_saved_state:
        sft_trainer_resume_arg = (
            True  # Let Trainer find the latest checkpoint in output_dir
        )
        print(
            f"SFTTrainer will attempt to resume from the latest checkpoint in its output_dir: {sft_config_args.output_dir}"
        )
    else:
        print(
            "SFTTrainer training from scratch (no SFTTrainer checkpoint specified or found for resume)."
        )

    # Check if the directory for SFTTrainer resume exists if a path was constructed (not True)
    if isinstance(sft_trainer_resume_arg, str) and not os.path.isdir(
        sft_trainer_resume_arg
    ):
        print(
            f"Warning: SFTTrainer resume path '{sft_trainer_resume_arg}' not found. Training from scratch."
        )
        sft_trainer_resume_arg = None  # Fallback to no resume

    print("\nInspecting SFT checkpoint for prior epoch count before training starts...")
    sft_checkpoint_dir_to_inspect = None
    if isinstance(sft_trainer_resume_arg, str) and os.path.isdir(
        sft_trainer_resume_arg
    ):
        sft_checkpoint_dir_to_inspect = sft_trainer_resume_arg
        print(
            f"SFTTrainer is configured to resume from explicit path: {sft_checkpoint_dir_to_inspect}"
        )
    elif sft_trainer_resume_arg is True:
        # SFTTrainer will look in sft_config_args.output_dir
        print(
            f"SFTTrainer is configured to resume from latest in output_dir: {sft_config_args.output_dir}"
        )
        # We need to find what SFTTrainer would find.
        # find_latest_checkpoint is imported from orign, which should be available.
        latest_sft_checkpoint_in_output_dir = find_latest_checkpoint(
            sft_config_args.output_dir
        )
        if latest_sft_checkpoint_in_output_dir:
            sft_checkpoint_dir_to_inspect = latest_sft_checkpoint_in_output_dir
            print(
                f"Found latest SFT checkpoint for inspection: {sft_checkpoint_dir_to_inspect}"
            )
        else:
            print(
                f"No SFT checkpoint found in {sft_config_args.output_dir} to inspect for prior epochs."
            )
    else:
        print(
            "SFTTrainer is not configured to resume from a checkpoint. No prior SFT epochs to report from trainer_state.json."
        )

    if sft_checkpoint_dir_to_inspect:
        trainer_state_path_to_inspect = os.path.join(
            sft_checkpoint_dir_to_inspect, "trainer_state.json"
        )
        print(
            f"Attempting to read SFT trainer state from: {trainer_state_path_to_inspect}"
        )
        if os.path.exists(trainer_state_path_to_inspect):
            try:
                with open(trainer_state_path_to_inspect, "r") as f:
                    sft_state_data = json.load(f)
                sft_epochs_completed = sft_state_data.get("epoch", 0.0)
                sft_global_step = sft_state_data.get("global_step", 0)
                print(
                    f"  >> SFT Checkpoint State: Epochs completed = {sft_epochs_completed}, Global steps = {sft_global_step}"
                )
            except Exception as e:
                print(
                    f"  >> Warning: Failed to read/parse SFT {trainer_state_path_to_inspect}: {e}"
                )
        else:
            print(f"  >> Warning: SFT {trainer_state_path_to_inspect} not found.")
    print("--- End of SFT Checkpoint Inspection ---")

    print("\nStarting SFTTrainer training...")
    trainer.train(resume_from_checkpoint=sft_trainer_resume_arg)
    print("SFTTrainer training finished.")

    # Save adapter weights - much simpler now that we train the actual adapter
    print(
        f"\nSaving adapter weights for '{adapter_name_to_train}' to base folder: {adapter_base_save_folder}"
    )
    model_ready_for_training.save_pretrained(adapter_base_save_folder)
    print("Adapter weights saved.")

    # With smart management, we keep the adapter loaded for fast future access
    # The LRU cache in add_or_load_adapter_for_model will handle eviction.
    # We'll just deactivate the current adapter if it's active.
    print(
        f"[LRU Management] Adapter '{adapter_name_to_train}' remains in LRU cache: {list(state.loaded_adapter_names_lru)}. Deactivating if active."
    )
    if (
        state.base_model.active_adapter == adapter_name_to_train
        or adapter_name_to_train in state.base_model.active_adapters
    ):  # Check both single and multiple active adapters
        state.base_model.active_adapter = None  # Deactivate
        print(
            f"[LRU Management] Deactivated adapter '{adapter_name_to_train}'. No active adapter set by default."
        )
    else:
        print(
            f"[LRU Management] Adapter '{adapter_name_to_train}' was not active. No deactivation needed."
        )

    del trainer, model_ready_for_training
    torch.cuda.empty_cache()
    gc.collect()
    print(
        f"--- train_lora_adapter for adapter: '{adapter_name_to_train}' completed ---\n"
    )

    # --- LRU Disk Cache: Update access for adapter after training (weights saved within train_lora_adapter) ---
    _update_item_access(
        adapter_name_to_train,
        "adapters",
        os.path.join(ADAPTER_DIR, adapter_name_to_train),
    )
    # ---


def train_unsloth_sft(message: Message[TrainingRequest]) -> TrainingResponse:
    import gc
    import json
    import shutil

    import requests
    import torch  # type: ignore
    from chatmux import oai_to_unsloth
    from nebu import (
        Bucket,
        ContainerConfig,
        V1ResourceReference,
        is_allowed,
    )
    from orign import Adapter, Training, V1LoraParams

    global state
    if not hasattr(state, "base_model") or not hasattr(state, "model_processor"):
        raise RuntimeError(
            "Base model and processor not initialized. Ensure init() has run."
        )

    # --- LRU Disk Cache: Make space before starting ---
    print("Ensuring storage limits before starting training operations...")
    _ensure_storage_limits()
    # ---

    # First ensure CUDA cache is cleared
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Force garbage collection multiple times to ensure all tensors are released
    gc.collect()

    print("message", message)
    if not message.content:
        raise ValueError("No training request provided")
    training_request: TrainingRequest = message.content

    container_config = ContainerConfig.from_env()
    print("container_config", container_config)

    bucket = Bucket()

    print("determining adapter namespace and name...")
    adapter_parts = training_request.adapter.split("/")
    if len(adapter_parts) == 2:
        print("adapter_parts", adapter_parts)
        adapter_namespace = adapter_parts[0]
        adapter_name = adapter_parts[1]
    else:
        adapter_name = training_request.adapter
        if training_request.owner:
            print("owner", training_request.owner)
            adapter_namespace = training_request.owner
        else:
            print("no owner, using message.handle", message.handle)
            adapter_namespace = message.handle

    print("adapter_namespace", adapter_namespace)
    if not adapter_namespace:
        raise ValueError("Could not determine adapter namespace")

    # Define local and bucket paths
    # ADAPTER_DIR is global: /nebu/cache/adapters
    local_adapter_weights_dir_for_current_adapter = os.path.join(
        ADAPTER_DIR, adapter_name
    )
    # train_lora_adapter saves adapter weights into a subfolder named adapter_name within this
    actual_local_adapter_files_path = os.path.join(
        local_adapter_weights_dir_for_current_adapter, adapter_name
    )

    local_sft_runs_dir = f"./runs/{adapter_name}"

    # Bucket URIs
    adapter_weights_bucket_uri = f"{container_config.namespace_volume_uri}/adapters/{adapter_namespace}/{adapter_name}"
    sft_checkpoints_bucket_uri = f"{container_config.namespace_volume_uri}/sft_runs/{adapter_namespace}/{adapter_name}"

    if training_request.labels:
        training_labels = training_request.labels.copy()
    else:
        training_labels = {}
    training_labels["message_id"] = message.id
    training_labels["container_id"] = os.getenv("NEBU_CONTAINER_ID", "unknown")
    random_chars = secrets.token_hex(3)

    adapter_ref = V1ResourceReference(
        name=adapter_name,
        namespace=adapter_namespace,
        kind="Adapter",
    )
    print("adapter_ref: ", adapter_ref)

    training = None
    try:
        print("creating training with api_key", message.api_key)
        training = Training(
            name=adapter_name + "-" + random_chars,
            namespace=adapter_namespace,
            config_data=message.model_dump(),
            adapter=adapter_ref,
            labels=training_labels,
            unique_adapter_active=True,
            api_key=message.api_key,
        )
        print("\n >> marking initial training as running")
        training.update(status=V1TrainingStatus.RUNNING)
    except Exception as e:
        print(
            f"FATAL: Failed to create or update Training resource for {adapter_ref}: {e}  --- retrying..."
        )
        raise RetriableError(
            f"Failed to set up Training resource: {e}  --- retrying..."
        ) from e

    failure = False
    try:
        start_adapter_time = time.time()
        adapter_resource = None
        try:
            adapters_found = Adapter.get(
                adapter_namespace, adapter_name, api_key=message.api_key
            )
            if adapters_found:
                adapter_resource = adapters_found[0]
        except Exception:
            adapters_found = []  # noqa
        print("found adapter resource", adapter_resource)

        is_continue = False
        epochs_trained_so_far = 0
        sft_checkpoint_to_resume_from = (
            None  # This will be local_sft_runs_dir if resuming
        )

        if adapter_resource:
            print("Found existing adapter resource: ", adapter_resource)
            epochs_trained_so_far = adapter_resource.epochs_trained

            if not is_allowed(
                adapter_resource.metadata.owner, message.user_id, message.orgs
            ):
                raise ValueError("You are not allowed to train this existing adapter")

            # Sync adapter weights from bucket to local ADAPTER_DIR/adapter_name
            # train_lora_adapter expects them in ADAPTER_DIR/adapter_name/adapter_name
            print(
                f"Attempting to sync adapter weights from {adapter_weights_bucket_uri} to {actual_local_adapter_files_path}"
            )
            os.makedirs(
                actual_local_adapter_files_path, exist_ok=True
            )  # Ensure target dir exists for sync

            adapter_sync_start_time = time.time()
            try:
                bucket.sync(adapter_weights_bucket_uri, actual_local_adapter_files_path)
                print(f"Synced adapter weights to {actual_local_adapter_files_path}")
                is_continue = True
                # --- LRU Disk Cache: Update access for synced adapter ---
                _update_item_access(
                    adapter_name,
                    "adapters",
                    local_adapter_weights_dir_for_current_adapter,
                )
                # ---
            except Exception as e:
                print(
                    f"Warning: Failed to sync adapter weights from {adapter_weights_bucket_uri}: {e}. May proceed without them if adapter is being created fresh by train_lora_adapter."
                )
            adapter_sync_end_time = time.time()
            print(
                f"Time taken to sync adapter weights: {adapter_sync_end_time - adapter_sync_start_time} seconds"
            )

            # Check if we have a specific checkpoint URI instead of a general SFT runs directory
            checkpoint_uri = adapter_resource.checkpoint_uri
            # If checkpoint URI points to a specific checkpoint (contains "/checkpoint-")
            if checkpoint_uri and "/checkpoint-" in checkpoint_uri:
                checkpoint_name = os.path.basename(checkpoint_uri)
                print(
                    f"Found specific checkpoint reference: {checkpoint_name} in {checkpoint_uri}"
                )

                # Create local directory for this specific checkpoint
                local_checkpoint_dir = os.path.join(local_sft_runs_dir, checkpoint_name)
                os.makedirs(local_checkpoint_dir, exist_ok=True)

                # Sync that specific checkpoint from bucket
                try:
                    print(
                        f"Syncing specific checkpoint from {checkpoint_uri} to {local_checkpoint_dir}"
                    )
                    sync_start_time = time.time()
                    bucket.sync(checkpoint_uri, local_checkpoint_dir)
                    sync_end_time = time.time()
                    print(
                        f"Time taken to sync specific checkpoint: {sync_end_time - sync_start_time} seconds"
                    )
                    print(f"Successfully synced checkpoint {checkpoint_name}")
                    sft_checkpoint_to_resume_from = local_checkpoint_dir
                    is_continue = True
                except Exception as e:
                    print(f"Failed to sync specific checkpoint: {e}")
                    is_continue = False
            else:
                # Fallback to old behavior in case checkpoint_uri is not a specific checkpoint
                print(
                    f"No specific checkpoint reference found in adapter. Will sync from general checkpoint directory: {checkpoint_uri}"
                )
                try:
                    bucket.sync(checkpoint_uri, local_sft_runs_dir)
                    # Look for latest checkpoint in the synced directory
                    latest_checkpoint = find_latest_checkpoint(local_sft_runs_dir)
                    if latest_checkpoint:
                        print(
                            f"Found latest checkpoint in synced directory: {latest_checkpoint}"
                        )
                        sft_checkpoint_to_resume_from = latest_checkpoint
                        is_continue = True
                    else:
                        print("No valid checkpoint found in synced directory")
                        is_continue = False
                except Exception as e:
                    print(f"Failed to sync from general checkpoint directory: {e}")
                    is_continue = False
        else:
            print(
                f"No existing Adapter resource found for {adapter_namespace}/{adapter_name}. This will be a new training."
            )
            # Clean up local directories for a fresh start if they exist from a failed previous run
            if os.path.exists(local_adapter_weights_dir_for_current_adapter):
                shutil.rmtree(local_adapter_weights_dir_for_current_adapter)
                print(
                    f"Cleaned up existing local adapter dir: {local_adapter_weights_dir_for_current_adapter}"
                )
            if os.path.exists(local_sft_runs_dir):
                shutil.rmtree(local_sft_runs_dir)
                print(f"Cleaned up existing local SFT runs dir: {local_sft_runs_dir}")
            os.makedirs(local_adapter_weights_dir_for_current_adapter, exist_ok=True)
            os.makedirs(local_sft_runs_dir, exist_ok=True)

        end_adapter_time = time.time()
        print(
            f"Time taken to download adapter: {end_adapter_time - start_adapter_time} seconds"
        )

        print("Downloading dataset")
        time_start_download = time.time()
        response = requests.get(training_request.dataset)
        response.raise_for_status()
        print(f"Downloaded dataset in {time.time() - time_start_download} seconds")

        lines = response.content.decode("utf-8").splitlines()
        time_start_convert = time.time()
        converted_dataset = [
            oai_to_unsloth(json.loads(line)) for line in lines if line.strip()
        ]
        print(f"Converted dataset in {time.time() - time_start_convert} seconds")
        print("dataset example", converted_dataset[:1])

        # Calculate the cumulative target number of epochs
        cumulative_target_epochs = epochs_trained_so_far + training_request.epochs
        print(
            f"Adapter '{adapter_name}': Has {epochs_trained_so_far} epochs trained. Requesting additional {training_request.epochs} epochs. Target cumulative epochs: {cumulative_target_epochs}."
        )
        print(
            f"Calling train_lora_adapter for '{adapter_name}'. Resume SFT: {is_continue}, SFT checkpoint path hint: {sft_checkpoint_to_resume_from}"
        )

        time_start_train = time.time()
        train_lora_adapter(
            adapter_name_to_train=adapter_name,
            training_dataset=converted_dataset,
            num_epochs=cumulative_target_epochs,  # Pass cumulative target epochs
            resume_from_saved_state=is_continue,  # For adapter weights loading
            checkpoint_path=sft_checkpoint_to_resume_from
            if is_continue
            else None,  # For SFTTrainer resume
        )
        print(
            f"train_lora_adapter completed in {time.time() - time_start_train} seconds"
        )

        # After training, sync artifacts to bucket
        # 1. Sync adapter weights
        # train_lora_adapter saves them to actual_local_adapter_files_path
        if os.path.exists(actual_local_adapter_files_path) and os.listdir(
            actual_local_adapter_files_path
        ):
            print(
                f"Syncing adapter weights from {actual_local_adapter_files_path} to {adapter_weights_bucket_uri}"
            )
            bucket.copy(actual_local_adapter_files_path, adapter_weights_bucket_uri)
            print("Synced adapter weights to bucket.")
        else:
            print(
                f"Warning: Local adapter files path {actual_local_adapter_files_path} is empty or does not exist after training. Cannot sync to bucket."
            )

        # 2. Sync SFT checkpoints
        # train_lora_adapter's SFTTrainer saves to local_sft_runs_dir
        if os.path.exists(local_sft_runs_dir) and os.listdir(local_sft_runs_dir):
            # Find the latest checkpoint directory
            latest_checkpoint = find_latest_checkpoint(local_sft_runs_dir)
            if latest_checkpoint:
                # --- Prune checkpoint directory before uploading ---
                print(f"Preparing to prune checkpoint directory: {latest_checkpoint}")
                print(f"  Target adapter to keep artifacts for: '{adapter_name}'")

                adapters_in_peft_config = []
                if hasattr(state, "base_model") and hasattr(
                    state.base_model, "peft_config"
                ):
                    adapters_in_peft_config = list(state.base_model.peft_config.keys())
                    print(
                        f"  Adapters known to be loaded in base_model's peft_config: {adapters_in_peft_config}"
                    )
                else:
                    print(
                        "  Warning: Could not retrieve loaded adapters from state.base_model.peft_config. Pruning might be affected."
                    )

                items_in_checkpoint_main_dir = os.listdir(latest_checkpoint)
                print(
                    f"  Items found in checkpoint directory '{latest_checkpoint}': {items_in_checkpoint_main_dir}"
                )

                for item_name in items_in_checkpoint_main_dir:
                    item_full_path = os.path.join(latest_checkpoint, item_name)
                    if os.path.isdir(item_full_path):
                        # Check if this directory is an adapter directory that is NOT the one we just trained
                        if (
                            item_name in adapters_in_peft_config
                            and item_name != adapter_name
                        ):
                            print(
                                f"    Pruning: Removing other adapter directory '{item_name}' from checkpoint."
                            )
                            shutil.rmtree(item_full_path)
                        elif item_name == adapter_name:
                            print(
                                f"    Keeping: Target adapter directory '{item_name}'."
                            )
                        else:
                            # This directory is not the target adapter and not clearly another known adapter from peft_config.
                            # Could be an unexpected directory, or an adapter not in peft_config (less likely).
                            # For safety, only known "other" adapters are pruned.
                            print(
                                f"    Skipping: Directory '{item_name}' (not the target, and not identified as another loaded adapter)."
                            )
                    else:
                        # Keep all files at the root of the checkpoint directory (e.g., optimizer.pt, trainer_state.json)
                        print(f"    Keeping: File '{item_name}'.")
                print(
                    f"  Pruning of checkpoint directory '{latest_checkpoint}' complete."
                )
                # --- End of pruning ---

                # Extract the checkpoint name (e.g., "checkpoint-132")
                latest_checkpoint_name = os.path.basename(latest_checkpoint)
                # Create a specific path for just the latest checkpoint
                latest_checkpoint_bucket_uri = (
                    f"{sft_checkpoints_bucket_uri}/{latest_checkpoint_name}"
                )

                print(
                    f"Syncing latest SFT checkpoint from {latest_checkpoint} to {latest_checkpoint_bucket_uri}"
                )
                bucket.copy(latest_checkpoint, latest_checkpoint_bucket_uri)
                print(f"Synced latest checkpoint ({latest_checkpoint_name}) to bucket.")

                # We'll use this specific checkpoint URI instead of the parent directory
                checkpoint_uri_for_adapter = latest_checkpoint_bucket_uri
            else:
                print(
                    f"Warning: Could not find a checkpoint directory in {local_sft_runs_dir}"
                )
                checkpoint_uri_for_adapter = sft_checkpoints_bucket_uri
        else:
            print(
                f"Warning: Local SFT runs directory {local_sft_runs_dir} is empty or does not exist after training. Cannot sync to bucket."
            )
            checkpoint_uri_for_adapter = sft_checkpoints_bucket_uri

        # --- LRU Disk Cache: Update access for SFT run directory ---
        if os.path.exists(local_sft_runs_dir):
            _update_item_access(
                f"{adapter_name}_sft_run", "sft_runs", local_sft_runs_dir
            )
        # ---

        # Collect metrics from trainer_state.json in the local SFT runs directory
        training_metrics = {}
        trainer_state_path = None  # Initialize to None

        latest_checkpoint_dir = find_latest_checkpoint(local_sft_runs_dir)
        if latest_checkpoint_dir:
            trainer_state_path = os.path.join(
                latest_checkpoint_dir, "trainer_state.json"
            )
            print(f"Found latest checkpoint directory: {latest_checkpoint_dir}")
            print(f"Attempting to load trainer_state.json from: {trainer_state_path}")
        else:
            print(
                f"Warning: No checkpoint directory found in {local_sft_runs_dir}. Looking for trainer_state.json in the root of SFT runs dir as a fallback."
            )
            # Fallback to old behavior if no checkpoint dir is found, though less likely to succeed
            trainer_state_path = os.path.join(local_sft_runs_dir, "trainer_state.json")

        if trainer_state_path and os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, "r") as f:
                    state_data = json.load(f)
                training_metrics = state_data
                print(f"Loaded training metrics from {trainer_state_path}")
                log_history = state_data.get("log_history", [])
                if log_history:
                    print(f"  Final log_history entry: {log_history[-1]}")
            except Exception as e:
                print(f"Warning: Failed to read/parse {trainer_state_path}: {e}.")
        else:
            print(f"Warning: {trainer_state_path} not found. Metrics will be empty.")

        # Update Adapter resource with the specific checkpoint URI
        final_epochs_trained = epochs_trained_so_far + training_request.epochs

        Adapter(
            name=adapter_name,
            namespace=adapter_namespace,
            model_uri=adapter_weights_bucket_uri,  # URI now points to the adapter weights in the bucket
            checkpoint_uri=checkpoint_uri_for_adapter,  # URI now points to the specific latest checkpoint
            owner=training_request.owner or message.user_id,  # type: ignore
            base_model=training_request.model,  # This is the original base model ID like "unsloth/Qwen2.5-VL-32B-Instruct"
            epochs_trained=final_epochs_trained,
            examples_trained=(
                adapter_resource.examples_trained if adapter_resource else 0
            )
            + len(converted_dataset),
            last_trained=int(time.time()),
            lora=V1LoraParams(
                r=training_request.lora_rank,
                alpha=training_request.lora_alpha,
                dropout=training_request.lora_dropout,
            ),
            labels=training_request.labels,
            api_key=message.api_key,
        )

        training.log(data=training_metrics)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Extract metrics for TrainingResponse
        # Use .get() with defaults to avoid KeyError if metrics are missing
        final_loss = 0.0
        train_steps_per_second = 0.0
        train_samples_per_second = 0.0
        train_runtime = 0.0

        if training_metrics:
            log_history = training_metrics.get("log_history", [])
            if log_history:  # Get loss from the last step
                final_loss = log_history[-1].get(
                    "loss", log_history[-1].get("train_loss", 0.0)
                )

            # These specific keys might not be in log_history but directly in trainer_stats from SFTTrainer
            # train_lora_adapter doesn't directly return trainer_stats, so we rely on trainer_state.json
            # which might have slightly different structure for aggregated stats.
            # For now, let's use what's typically available or default to 0.
            train_steps_per_second = training_metrics.get(
                "train_steps_per_second", 0.0
            )  # This key might not exist directly
            train_samples_per_second = training_metrics.get(
                "train_samples_per_second", 0.0
            )  # This key might not exist
            train_runtime = training_metrics.get(
                "train_runtime", time.time() - time_start_train
            )  # Fallback to measured time

        return TrainingResponse(
            loss=final_loss,
            train_steps_per_second=train_steps_per_second,
            train_samples_per_second=train_samples_per_second,
            train_runtime=train_runtime,
            adapter=training_request.adapter,
            adapter_uri=adapter_weights_bucket_uri,  # Return the bucket URI for the adapter weights
        )
    except Exception as e:
        print(f"Error training unsloth: {e}")
        failure = True
        if training:
            print("\n >> marking training as failed due to exception")
            training.update(status=V1TrainingStatus.FAILED)
        # If an error occurs, we must still return a TrainingResponse or raise.
        # Raising the original error is often better for debugging.
        raise
    finally:
        print(
            f"finally block: training resource exists: {bool(training)}, failure: {failure}"
        )
        if training:
            if failure:
                if (
                    training.training.status != V1TrainingStatus.FAILED
                ):  # Avoid double update if already set
                    print("\n >> ensuring training is marked as FAILED in finally")
                    training.update(status=V1TrainingStatus.FAILED)
            else:
                if training.training.status != V1TrainingStatus.COMPLETED:
                    print("\n >> marking training as COMPLETED in finally")
                    training.update(status=V1TrainingStatus.COMPLETED)


def UnslothSFT(
    platform: str = "runpod",
    accelerators: List[str] = ["1:H100_SXM"],
    image: str = "public.ecr.aws/d8i6n0n1/orign/unsloth-trainer:8b0ee04",  # "us-docker.pkg.dev/agentsea-dev/orign/unsloth-train:latest"
    scale: V1Scale = scale,
    namespace: Optional[str] = None,
    env: Optional[List[V1EnvVar]] = None,
    config: Optional[NebuGlobalConfig] = None,
    hot_reload: bool = True,
    debug: bool = False,
    min_replicas: int = 1,
    max_replicas: int = 4,
    name: Optional[str] = None,
    wait_for_healthy: bool = True,
) -> Processor[TrainingRequest, TrainingResponse]:
    decorate = processor(
        image=image,
        # setup_script=setup_script,
        accelerators=accelerators,
        platform=platform,
        scale=scale,
        namespace=namespace,
        env=env,
        init_func=init,
        # execution_mode="subprocess",
        config=config,
        hot_reload=hot_reload,
        debug=debug,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        name=name,
        wait_for_healthy=wait_for_healthy,
    )
    return decorate(train_unsloth_sft)
