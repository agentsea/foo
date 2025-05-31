import os
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

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "unsloth/Qwen2.5-VL-32B-Instruct")


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
    )


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

    print(f"\n[Smart Inference] Managing adapter: '{adapter_name}'")

    # Check current state
    adapter_already_loaded = (
        adapter_name in model.peft_config if hasattr(model, "peft_config") else False
    )

    # Find if we have this adapter tracked
    existing_adapter_info = None
    existing_adapter_index = -1

    for idx, loaded_adapter in enumerate(loaded_adapters_list):
        if (
            loaded_adapter.metadata.name == adapter_to_load.metadata.name
            and loaded_adapter.metadata.namespace == adapter_to_load.metadata.namespace
        ):
            existing_adapter_info = loaded_adapter
            existing_adapter_index = idx
            break

    print(
        f"[Smart Inference] Adapter '{adapter_name}' already loaded: {adapter_already_loaded}"
    )
    print(f"[Smart Inference] Have tracked info: {existing_adapter_info is not None}")

    if existing_adapter_info:
        if (
            existing_adapter_info.metadata.updated_at
            == adapter_to_load.metadata.updated_at
        ):
            # CASE 1: Exact same version already loaded - just switch to it (fastest!)
            print(
                "[Smart Inference] FAST SWITCH: Exact version already loaded, using set_adapter()"
            )
            model.set_adapter(adapter_name)
            return False  # No loading needed

        elif adapter_already_loaded:
            # CASE 2: Different version loaded - try hotswapping
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
                except:
                    pass  # Best effort cleanup
                return _load_adapter_traditionally(
                    model, adapter_to_load, adapter_name, bucket, loaded_adapters_list
                )
        else:
            # CASE 3: We have tracking info but adapter not actually loaded - load it
            print(
                "[Smart Inference] RELOAD: Adapter info exists but not loaded, loading traditionally"
            )
            return _load_adapter_traditionally(
                model, adapter_to_load, adapter_name, bucket, loaded_adapters_list
            )
    else:
        # CASE 4: Completely new adapter - load traditionally
        print(f"[Smart Inference] NEW ADAPTER: Loading '{adapter_name}' traditionally")
        return _load_adapter_traditionally(
            model, adapter_to_load, adapter_name, bucket, loaded_adapters_list
        )


def _load_adapter_traditionally(
    model: Any,
    adapter_to_load: Any,
    adapter_name: str,
    bucket: Any,
    loaded_adapters_list: List[Any],
) -> bool:
    """Traditional adapter loading for new adapters"""
    print(f"[Traditional Inference] Loading new adapter '{adapter_name}'")

    # Download adapter
    adapter_path = f"./adapters/{adapter_name}"
    print(f"[Traditional Inference] Downloading to {adapter_path}")

    time_start_copy = time.time()
    bucket.copy(adapter_to_load.model_uri, adapter_path)
    print(
        f"[Traditional Inference] Downloaded in {time.time() - time_start_copy} seconds"
    )

    # Load adapter
    print(f"[Traditional Inference] Loading adapter '{adapter_name}'")
    time_start_load = time.time()
    model.load_adapter(adapter_path, adapter_name=adapter_name)
    print(f"[Traditional Inference] Loaded in {time.time() - time_start_load} seconds")

    # Track the adapter
    loaded_adapters_list.append(adapter_to_load)
    print(f"[Traditional Inference] Successfully loaded adapter '{adapter_name}'")

    return True


def infer_qwen_vl(
    message: Message[ChatRequest],
) -> ChatResponse:
    full_time = time.time()
    from qwen_vl_utils import process_vision_info  # type: ignore
    from unsloth import FastVisionModel  # type: ignore

    global state

    print("message", message)
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
                print(f"Adapter {content.model} loaded/updated successfully")
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
    model: str = "unsloth/Qwen2.5-VL-32B-Instruct",
    image: str = "public.ecr.aws/d8i6n0n1/orign/unsloth-server:324d8c6",  # "us-docker.pkg.dev/agentsea-dev/orign/unsloth-infer:latest"
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
