#!/usr/bin/env python

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import SkipExample, load_dataset
from huggingface_hub import upload_file
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments


def load_image_from_path(path: str) -> Image.Image:
    """Loads an image from disk (or a remote path) and converts to RGB."""
    return Image.open(path).convert("RGB")


def preprocess_row(example: Dict) -> Dict:
    """
    Convert the 'images' (list of file paths) into actual list of PIL.Image objects,
    storing them in `example["image_list"]`. Also store `prompt_text` and `answer_text`.
    If there are no valid images, skip the example.
    """
    img_paths = example.get("images", [])
    pil_images = []
    for path in img_paths:
        if os.path.exists(path):
            # Load the existing local image file
            pil_images.append(load_image_from_path(path))
        else:
            # Handle missing or invalid image paths
            print(f"Warning: image file not found or not local: {path}")
            # Skip this image and continue
            continue

    if not pil_images:
        # No valid images found, skip this example
        print("Warning: No valid images in this example, skipping.")
        raise SkipExample()  # Use SkipExample to skip this example

    # Store the valid images and texts in the example
    example["image_list"] = pil_images
    example["prompt_text"] = example.get("query", "")
    example["answer_text"] = example.get("response", "")
    return example


def process_batch(
    processor: AutoProcessor,
    images_list: List[List[Image.Image]],
    prompts: List[str],
    answers: List[str] = [],
) -> Dict[str, torch.Tensor]:
    """
    Process a batch of prompts, answers, and images for model input.
    Returns a dict with padded input_ids, images, image_input_idx, image_masks, etc.
    """

    batch_size = len(prompts)
    # Ensure answers is either empty or same length as prompts
    assert (
        len(answers) == 0 or len(answers) == batch_size
    ), "Answers must have same length as prompts or be empty."

    # --- Prepare token IDs ---
    if len(answers) == 0:
        # No answers provided, only encode prompts
        tokens_list = []
        for prompt in prompts:
            tokens = processor.tokenizer.encode(
                "User: " + prompt + " Assistant:",
                add_special_tokens=False,
            )
            tokens_list.append(tokens)
    else:
        # Encode prompts + answers
        tokens_list = []
        for prompt, answer in zip(prompts, answers):
            tokens = processor.tokenizer.encode(
                "User: " + prompt + " Assistant: " + answer,
                add_special_tokens=False,
            )
            tokens_list.append(tokens)

    # --- Convert images from PIL to arrays ---
    images_arrays_list = []
    image_idxs_list = []
    for images in images_list:
        image_arrays = []
        image_idxs = []
        for idx, img in enumerate(images):
            if isinstance(img, Image.Image):
                img = ImageOps.exif_transpose(img)  # Fix orientation
                image_arrays.append(np.array(img.convert("RGB")))
                image_idxs.append(idx)  # Use actual index
            else:
                # Skip None images
                print("Warning: Encountered None image, skipping.")
                continue
        images_arrays_list.append(image_arrays)
        image_idxs_list.append(image_idxs)

    # --- Multimodal preprocess each example ---
    # You can tune these kwargs as needed
    images_kwargs = {
        "max_crops": 12,
        "overlap_margins": [4, 4],
        "base_image_input_size": [336, 336],
        "image_token_length_w": 12,
        "image_token_length_h": 12,
        "image_patch_size": 14,
        "image_padding_mask": True,
    }

    outputs_list = []
    for i in range(batch_size):
        tokens = tokens_list[i]
        imgs = images_arrays_list[i]
        idxs = image_idxs_list[i]
        out = processor.image_processor.multimodal_preprocess(
            images=imgs,
            image_idx=idxs,
            tokens=np.asarray(tokens).astype(np.int32),
            sequence_length=64,  # Adjust as needed
            image_patch_token_id=processor.special_token_ids["<im_patch>"],
            image_col_token_id=processor.special_token_ids["<im_col>"],
            image_start_token_id=processor.special_token_ids["<im_start>"],
            image_end_token_id=processor.special_token_ids["<im_end>"],
            **images_kwargs,
        )
        outputs_list.append(out)

    # --- Collate into padded batch tensors ---
    batch_outputs = {}
    for key in outputs_list[0].keys():
        tensors = [torch.from_numpy(out[key]) for out in outputs_list]
        batch_outputs[key] = torch.nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=-1,
        )

    # Prepend a BOS token (reusing the tokenizer's EOS as BOS, if desired)
    batch_outputs["input_ids"] = torch.nn.functional.pad(
        batch_outputs["input_ids"],
        (1, 0),
        value=processor.tokenizer.eos_token_id,  # or any special BOS token
    )

    # Shift image input indices because of the extra token we just added
    image_input_idx = batch_outputs["image_input_idx"]
    batch_outputs["image_input_idx"] = torch.where(
        image_input_idx < 0, image_input_idx, image_input_idx + 1
    )

    # Create labels, masking out padding/special tokens
    batch_outputs["labels"] = batch_outputs["input_ids"].clone()
    batch_outputs["labels"][batch_outputs["labels"] == -1] = -100
    special_token_ids = list(processor.special_token_ids.values())
    for special_id in special_token_ids:
        batch_outputs["labels"][batch_outputs["labels"] == special_id] = -100

    return batch_outputs


class DataCollator:
    """
    A data collator class for batching dataset examples into model inputs.
    Adapts your dataset to (images, prompts, answers) suitable for process_batch().
    """

    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, dataset: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1) Gather prompts and answers
        prompts = [row["prompt_text"] for row in dataset]
        answers = [row["answer_text"] for row in dataset]
        # 2) Gather images
        images_list = [row["image_list"] for row in dataset]

        # 3) Batch process
        batch_outputs = process_batch(
            processor=self.processor,
            images_list=images_list,
            prompts=prompts,
            answers=answers,
        )
        return batch_outputs


def train() -> None:
    """Trains the Molmo-7B-D model on your JSONL dataset."""
    # 1) Load the JSONL-based dataset
    #    Adjust the paths and splits to match your file structure
    train_dataset = load_dataset(
        "json", data_files=os.getenv("TRAIN_DATA_PATH", "train.jsonl"), split="train"
    )
    eval_dataset = load_dataset(
        "json", data_files=os.getenv("VAL_DATA_PATH", "val.jsonl"), split="train"
    )

    # 2) Preprocess them (turn image paths into PIL Images, store them in "image_list")
    train_dataset = train_dataset.map(preprocess_row)
    eval_dataset = eval_dataset.map(preprocess_row)

    print(f"Training dataset size after filtering: {len(train_dataset)}")
    print(f"Evaluation dataset size after filtering: {len(eval_dataset)}")

    # 4) Setup training args
    training_args = TrainingArguments(
        # Where to store checkpoints
        output_dir="./checkpoints",
        # Training
        per_device_train_batch_size=1,
        num_train_epochs=int(os.getenv("NUM_EPOCHS", "10")),
        bf16=True,  # or False if not using BF16
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # Evaluation
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=10,
        # FSDP (optional, depends on your environment)
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": "MolmoSequentialBlock"},
        # Logging
        logging_steps=1,
        report_to="wandb",
        save_steps=1000,
        save_total_limit=1,
        # Hub
        hub_private_repo=True,
        hub_model_id="agentsea/molmo-7b-ft-tideui",
        push_to_hub=False,  # set True if you want to push to HF Hub automatically
        # Optimizer settings
        learning_rate=1e-5,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.05,
        max_grad_norm=1.0,
        seed=3407,
    )

    # 5) Load the processor and model
    model_name = "allenai/Molmo-7B-D-0924"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # or torch.bfloat16 if your hardware supports it
    )

    # 6) (Optional) Update the AutoModelForCausalLM if needed
    model.config.auto_map["AutoModelForCausalLM"] = (
        "agentsea/molmo-7b-ft-tideui--modeling_molmo.MolmoForCausalLM"
    )

    # 7) Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(processor),
    )

    # 8) Train
    trainer.train()

    # 9) (Optional) Push model + processor to the Hub
    trainer.push_to_hub()  # Will only run if push_to_hub=True
    processor.push_to_hub("agentsea/molmo-7b-ft-tideui", private=True)

    # 10) (Optional) Upload custom code or model files if needed
    # E.g., if you have a custom modeling_molmo.py
    upload_file(
        path_or_fileobj="./modeling_molmo.py",
        path_in_repo="modeling_molmo.py",
        repo_id="agentsea/molmo-7b-ft-tideui",
    )


if __name__ == "__main__":
    train()
