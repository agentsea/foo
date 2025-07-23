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
from nebulous import (
    Message,
    Processor,
    V1EnvVar,
    processor,
)
from nebulous.config import GlobalConfig as NebuGlobalConfig


def init():
    import gc
    import os

    from huggingface_hub import snapshot_download

    from unsloth import FastVisionModel  # type: ignore # isort: skip
    import torch  # type: ignore

    if "state" in globals():  # <-- already loaded by an earlier worker
        print("state already loaded by an earlier worker")
        return

    gc.collect()
    torch.cuda.empty_cache()

    @dataclass
    class InferenceState:
        base_model: FastVisionModel
        model_processor: Any
        adapter_name: str

    try:
        print("loading model...")
        print("--- nvidia-smi before load ---")
        os.system("nvidia-smi")
        print("--- end nvidia-smi before load ---")
        time_start_load = time.time()

        base_model_id = "unsloth/Qwen2.5-VL-32B-Instruct"
        adapter_name = "agentsea/Qwen2.5-VL-32B-Instruct-CARL-Gflights4"

        base_model, model_processor = FastVisionModel.from_pretrained(
            base_model_id,
            load_in_4bit=False,
            # use_fast=True,
            dtype=torch.bfloat16,
            max_seq_length=32_768,
        )
        print(f"Loaded base model in {time.time() - time_start_load} seconds")
        print(f"Loaded base model of type: {type(base_model)}")

        print(f"Loading adapter: {adapter_name}")
        time_start_adapter_load = time.time()

        # Create a sanitized name for PEFT/PyTorch, which disallows '.' or '/'
        sanitized_adapter_name = adapter_name.replace("/", "_").replace(".", "_")

        # Download adapter locally first to ensure correct naming
        local_adapter_path = f"./{sanitized_adapter_name}"
        print(f"Downloading adapter '{adapter_name}' to '{local_adapter_path}'...")
        snapshot_download(repo_id=adapter_name, local_dir=local_adapter_path)
        print("Adapter downloaded.")

        base_model.load_adapter(local_adapter_path, adapter_name=sanitized_adapter_name)
        base_model.set_adapter(sanitized_adapter_name)
        print(
            f"Loaded and set adapter in {time.time() - time_start_adapter_load} seconds"
        )

        print("--- nvidia-smi after load ---")
        os.system("nvidia-smi")
        print("--- end nvidia-smi after load ---")

        global state
        state = InferenceState(
            base_model=base_model,
            model_processor=model_processor,
            adapter_name=adapter_name,
        )

    except Exception as e:
        print(f"Error during init: {e}")
        raise e


def infer_qwen_vl(
    message: Message[ChatRequest],
) -> ChatResponse:
    full_time = time.time()
    from qwen_vl_utils import process_vision_info  # type: ignore
    from unsloth import FastVisionModel  # type: ignore

    global state

    print("message", message)
    os.system("nvidia-smi")

    content = message.content
    if not content:
        raise ValueError("No content provided")

    print("setting model for inference")
    FastVisionModel.for_inference(state.base_model)

    content_dict = content.model_dump()
    messages_oai = content_dict["messages"]
    messages = oai_to_qwen(messages_oai)

    # Preparation for inference
    inputs_start = time.time()
    text = state.model_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = state.model_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
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
        model=state.adapter_name,
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
    image: str = "public.ecr.aws/d8i6n0n1/orign/unsloth-server:6671ad5",  # "public.ecr.aws/d8i6n0n1/orign/unsloth-server:8b0ee04",  # "us-docker.pkg.dev/agentsea-dev/orign/unsloth-infer:latest"
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
    decorate = processor(
        image=image,
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
