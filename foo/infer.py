import time
import uuid
from dataclasses import dataclass
from typing import Any, List

from chatmux.convert import oai_to_qwen
from chatmux.openai import (
    ChatRequest,
    ChatResponse,
    CompletionChoice,
    Logprobs,
    ResponseMessage,
)
from nebu import (
    Adapter,
    Bucket,
    ContainerConfig,
    Message,
    is_allowed,
)
from nebu.processors.decorate import processor

setup_script = """
apt update
apt install -y git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip uninstall -y xformers
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install trl peft transformers bitsandbytes sentencepiece accelerate tiktoken qwen-vl-utils chatmux
pip install -e git+https://github.com/pbarker/unsloth-zoo.git#egg=unsloth_zoo
pip install -e git+https://github.com/pbarker/unsloth.git#egg=unsloth
"""

_RUNS = 0
BASE_MODEL_ID = "unsloth/Qwen2.5-VL-32B-Instruct"


def init():
    import os

    # TODO: remove this, was transient bug
    global _RUNS
    _RUNS += 1
    print(f">>>>>>>> init() called! runs={_RUNS}, pid={os.getpid()}")
    import gc

    import torch
    from nebu import Adapter, Cache
    from unsloth import FastVisionModel

    if "state" in globals():  # <-- already loaded by an earlier worker
        return

    gc.collect()
    torch.cuda.empty_cache()

    os.environ.setdefault("MAX_PIXELS", "100352")

    @dataclass
    class InferenceState:
        base_model: FastVisionModel
        model_processor: Any
        base_model_id: str
        adapters: List[Adapter]
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
        max_seq_length=65_536,
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


@processor(
    image="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel",
    setup_script=setup_script,
    accelerators=["1:A100_SXM"],
    platform="runpod",
    init_func=init,
)
def infer_qwen_vl(
    message: Message[ChatRequest],
) -> ChatResponse:
    full_time = time.time()
    from qwen_vl_utils import process_vision_info
    from unsloth import FastVisionModel

    global state

    print("message", message)
    training_request = message.content
    if not training_request:
        raise ValueError("No training request provided")

    print("content", message.content)

    container_config = ContainerConfig.from_env()
    print("container_config", container_config)

    content = message.content
    if not content:
        raise ValueError("No content provided")

    load_adapter = content.model != "" and content.model != BASE_MODEL_ID

    if load_adapter:
        adapter_hot_start = time.time()
        print("checking cache for adapter", f"'adapters:{content.model}'")
        val_raw = state.cache.get(f"adapters:{content.model}")
        if val_raw:
            print("val_raw", val_raw)
            val = Adapter.model_validate_json(val_raw)
            print("found adapter in cache", val)

            if not is_allowed(val.owner, message.user_id, message.orgs):
                raise ValueError("You are not allowed to use this adapter")

            if not val.base_model == BASE_MODEL_ID:
                raise ValueError(
                    "The base model of the adapter does not match the model you are trying to use"
                )

            loaded = False
            for adapter in state.adapters:
                print("cached adapter: ", adapter)
                if val.name == content.model and val.created_at == adapter.created_at:
                    loaded = True
                    print("adapter already loaded", content.model)
                    break
            print(f"Adapter hot start: {time.time() - adapter_hot_start} seconds")

            try:
                print("peft config", state.base_model.peft_config.keys())
            except Exception as e:
                print("Failed getting peft config (expected error)", e)
                pass

            if not loaded:
                bucket = Bucket()
                print("copying adapter", val.uri, f"./adapters/{content.model}")

                time_start = time.time()
                bucket.copy(val.uri, f"./adapters/{content.model}")
                print(f"Copied in {time.time() - time_start} seconds")

                print("loading adapter", content.model)
                state.base_model.load_adapter(
                    f"./adapters/{content.model}",
                    adapter_name=content.model,
                    low_cpu_mem_usage=False,
                )
                state.adapters.append(val)  # type: ignore
                print("loaded adapter", content.model)

        else:
            raise ValueError(f"Adapter '{content.model}' not found")
        print("adapter total start time ", time.time() - adapter_hot_start)

    loaded_adapter_names = list(state.base_model.peft_config.keys())
    print("loaded_adapter_names: ", loaded_adapter_names)

    if load_adapter:
        print("setting adapter", content.model)
        state.base_model.set_adapter(content.model)
    else:
        # Ensure no adapter is active if load_adapter is False
        try:
            print("Disabling any active adapter.")
            state.base_model.disable_adapter()
        except Exception as e:
            # May fail if no adapter was ever loaded or already disabled
            print(f"Failed to disable adapter (might be expected): {e}")

    print("setting model for inference")
    FastVisionModel.for_inference(state.base_model)

    conent_dict = content.model_dump()
    messages_oai = conent_dict["messages"]
    messages = oai_to_qwen(messages_oai)

    # Preparation for inference
    print("preparing inputs using messages: ", messages)
    inputs_start = time.time()
    text = state.model_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("text: ", text)
    print("processing vision info: ", messages)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = state.model_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    print("inputs", inputs)
    print(f"Inputs prepared in {time.time() - inputs_start} seconds")

    # Inference: Generation of the output
    generated_ids = state.base_model.generate(
        **inputs, max_new_tokens=content.max_tokens
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    generation_start = time.time()
    output_text = state.model_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print("output_text", output_text)
    print(f"Generation took {time.time() - generation_start} seconds")

    # Build the Pydantic model, referencing your enumerations and classes
    response = ChatResponse(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        model=content.model,
        object="chat.completion",
        choices=[
            CompletionChoice(
                index=0,
                finish_reason="stop",  # or another appropriate reason
                message=ResponseMessage(  # type: ignore
                    role="assistant", content=output_text[0]
                ),
                # Stub logprobs; in reality, you'd fill these from your model if you have them
                logprobs=Logprobs(content=[]),
            )
        ],
        service_tier=None,
        system_fingerprint=None,
        usage=None,
    )
    print(f"Total time: {time.time() - full_time} seconds")

    return response
