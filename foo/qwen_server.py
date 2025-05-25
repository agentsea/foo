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

            # Check if the correct version of the adapter is already loaded
            needs_loading = True
            existing_adapter_to_unload = None
            existing_adapter_index = -1

            for idx, loaded_adapter in enumerate(state.adapters):
                print("checking against cached adapter: ", loaded_adapter)
                if (
                    loaded_adapter.metadata.name == adapter_to_load.metadata.name
                    and loaded_adapter.metadata.namespace
                    == adapter_to_load.metadata.namespace
                ):
                    if (
                        loaded_adapter.metadata.updated_at
                        == adapter_to_load.metadata.updated_at
                    ):
                        # Correct version already loaded
                        print(
                            f"Adapter version {content.model} (updated_at: {adapter_to_load.metadata.updated_at}) already loaded."
                        )
                        needs_loading = False
                        break
                    else:
                        # Different version loaded, mark for unloading
                        print(
                            f"Found different version of adapter {content.model}. Current: updated_at={loaded_adapter.metadata.updated_at}, Requested: updated_at={adapter_to_load.metadata.updated_at}"
                        )
                        existing_adapter_to_unload = loaded_adapter
                        existing_adapter_index = idx
                        break  # Stop checking, we found the one to replace

            print(
                f"Adapter hot start check took: {time.time() - adapter_hot_start} seconds"
            )

            if needs_loading:
                # Unload existing adapter if a different version was found
                if existing_adapter_to_unload:
                    try:
                        print(
                            f"Unloading existing adapter: {existing_adapter_to_unload.metadata.name} (updated_at: {existing_adapter_to_unload.metadata.updated_at})"
                        )

                        try:
                            print("peft_config: ", state.base_model.peft_config.keys())
                            # Use delete_adapter to remove it from the base model's peft config
                            state.base_model.delete_adapter(content.model)
                        except Exception as e:
                            print(
                                f"Failed to delete adapter {existing_adapter_to_unload.metadata.name}: {e}"
                            )
                        # Remove from our tracked list
                        del state.adapters[existing_adapter_index]
                        print(
                            f"Successfully unloaded adapter {existing_adapter_to_unload.metadata.name}"
                        )
                        # Optional: Clean up local files for the old adapter version?
                        # shutil.rmtree(f"./adapters/{content.model}", ignore_errors=True) # Requires import shutil
                    except Exception as e:
                        print(
                            f"Failed to unload existing adapter {existing_adapter_to_unload.metadata.name}: {e}"
                        )
                        # If unloading fails, we cannot proceed with loading the new one under the same name.
                        raise ValueError(
                            f"Failed to unload existing adapter '{existing_adapter_to_unload.metadata.name}' to load the new version."
                        ) from e

                # Download and load the new adapter
                bucket = Bucket()
                adapter_path = f"./adapters/{content.model}"  # content.model is the adapter name like 'namespace/name'
                print("copying adapter", adapter_to_load.model_uri, adapter_path)

                time_start_copy = time.time()
                bucket.copy(adapter_to_load.model_uri, adapter_path)
                print(f"Copied in {time.time() - time_start_copy} seconds")

                print("loading adapter", content.model)
                time_start_load_adapter = time.time()
                state.base_model.load_adapter(
                    adapter_path,
                    adapter_name=content.model,
                    # low_cpu_mem_usage=False, # Keep this if needed, or set based on requirements
                )
                state.adapters.append(
                    adapter_to_load
                )  # Add the newly loaded adapter info
                print(
                    f"Loaded adapter {content.model} in {time.time() - time_start_load_adapter} seconds"
                )

        else:
            raise ValueError(f"Adapter '{content.model}' not found")
        print("adapter loading/checking total time: ", time.time() - adapter_hot_start)

    # Ensure peft_config exists before trying to access keys
    loaded_adapter_names = []
    if hasattr(state.base_model, "peft_config"):
        loaded_adapter_names = list(state.base_model.peft_config.keys())
    print("loaded_adapter_names: ", loaded_adapter_names)

    if load_adapter:
        # Check if the adapter we intend to use is actually in the loaded adapters list now
        if content.model not in loaded_adapter_names:
            # This case might happen if loading failed silently, or logic error
            raise RuntimeError(
                f"Adapter {content.model} was requested but is not loaded in the model."
            )

        print("setting adapter", content.model)
        state.base_model.set_adapter(content.model)
    else:
        # Ensure no adapter is active if load_adapter is False
        print("disabling adapter")
        try:
            if hasattr(state.base_model, "disable_adapter"):
                print("Disabling any active adapter.")
                state.base_model.disable_adapter()
        except Exception as e:
            # May fail if no adapter was ever loaded or already disabled
            print(f"Failed to disable adapter (might be expected): {e}")

    print("setting model for inference")
    FastVisionModel.for_inference(state.base_model)

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
    image: str = "public.ecr.aws/d8i6n0n1/orign/unsloth-server:491f93e",  # "us-docker.pkg.dev/agentsea-dev/orign/unsloth-infer:latest",  # "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel",
    namespace: Optional[str] = None,
    env: Optional[List[V1EnvVar]] = None,
    config: Optional[NebuGlobalConfig] = None,
    hot_reload: bool = True,
    debug: bool = False,
    min_replicas: int = 1,
    max_replicas: int = 4,
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
    )
    return decorate(infer_qwen_vl)
