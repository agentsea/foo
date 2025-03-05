from orign import GlobalConfig, ReplayBuffer, V1MSSwiftBufferParams


def create_actor_sft_buffer(
    name: str, skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Actor model for this skill"""

    actor_sft_buffer = ReplayBuffer(
        name=name,
        vram_request="40Gi",
        train_every=30,
        sample_n=100,
        sample_strategy="LatestWithRandom",
        adapter=name,
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3",
            torch_dtype="bfloat16",
            max_length=8192,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
            gradient_accumulation_steps_total=4,
        ),
        labels={"skill": skill_id},
        config=orign_config,
    )
    return actor_sft_buffer


def create_actor_dpo_buffer(
    name: str, skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Actor model DPO for this skill"""

    actor_dpo_buffer = ReplayBuffer(
        name=f"{name}-dpo",
        vram_request="80Gi",
        train_every=10000,
        sample_n=100,
        adapter=name,
        queue=name,
        sample_strategy="LatestWithRandom",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3",
            torch_dtype="bfloat16",
            max_length=8192,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
            rlhf_type="dpo",
        ),
        labels={"skill": skill_id},
        config=orign_config,
    )
    return actor_dpo_buffer


def create_base_actor_sft_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Base actor model"""

    actor_sft_buffer = ReplayBuffer(
        name="actor-base",
        vram_request="40Gi",
        train_every=100,
        sample_n=100,
        sample_strategy="LatestWithRandom",
        adapter="actor-base",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3",
            torch_dtype="bfloat16",
            max_length=8192,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
        ),
        labels={"skill": skill_id},
        config=orign_config,
    )
    return actor_sft_buffer


def create_base_actor_dpo_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Base actor model DPO"""

    actor_dpo_buffer = ReplayBuffer(
        name="actor-base-dpo",
        vram_request="80Gi",
        train_every=10000,
        sample_n=100,
        adapter="actor-base",
        queue="actor-base",
        sample_strategy="LatestWithRandom",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3",
            torch_dtype="bfloat16",
            max_length=8192,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
            rlhf_type="dpo",
        ),
        labels={"skill": skill_id},
        config=orign_config,
    )
    return actor_dpo_buffer


def create_val_sft_buffer(
    name: str, skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Validation model for this skill"""

    val_sft_buffer = ReplayBuffer(
        name=name,
        vram_request="40Gi",
        train_every=30,
        sample_n=100,
        sample_strategy="LatestWithRandom",
        adapter=name,
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3_offload",
            torch_dtype="bfloat16",
            max_length=16384,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
        ),
        config=orign_config,
        labels={"skill": skill_id},
    )
    return val_sft_buffer


def create_val_dpo_buffer(
    name: str, skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Validation model DPO for this skill"""

    val_dpo_buffer = ReplayBuffer(
        name=f"{name}-dpo",
        vram_request="80Gi",
        train_every=10000,
        sample_n=100,
        adapter=name,
        queue=name,
        sample_strategy="LatestWithRandom",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3",
            torch_dtype="bfloat16",
            max_length=16384,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
            rlhf_type="dpo",
        ),
        config=orign_config,
        labels={"skill": skill_id},
    )
    return val_dpo_buffer


def create_base_val_sft_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Base validation model"""

    val_sft_buffer = ReplayBuffer(
        name="val-base",
        vram_request="40Gi",
        train_every=100,
        sample_n=100,
        sample_strategy="LatestWithRandom",
        adapter="val-base",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3_offload",
            torch_dtype="bfloat16",
            max_length=16384,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
        ),
        config=orign_config,
        labels={"skill": skill_id},
    )
    return val_sft_buffer


def create_base_val_dpo_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Base validation model DPO"""

    val_dpo_buffer = ReplayBuffer(
        name="val-base-dpo",
        vram_request="80Gi",
        train_every=10000,
        sample_n=100,
        adapter="val-base",
        sample_strategy="LatestWithRandom",
        queue="val-base",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3",
            torch_dtype="bfloat16",
            max_length=16384,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
            rlhf_type="dpo",
        ),
        config=orign_config,
        labels={"skill": skill_id},
    )
    return val_dpo_buffer


def create_reason_annot_sft_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Reason annotation model for demonstrations"""

    val_sft_buffer = ReplayBuffer(
        name="reason-annot",
        vram_request="40Gi",
        train_every=100,
        sample_n=100,
        sample_strategy="LatestWithRandom",
        adapter="reason-annot",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3_offload",
            torch_dtype="bfloat16",
            max_length=16384,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
        ),
        config=orign_config,
        labels={"skill": skill_id},
    )
    return val_sft_buffer


def create_reason_annot_dpo_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Reason annotation model for demonstrations DPO"""

    val_dpo_buffer = ReplayBuffer(
        name="reason-annot-dpo",
        vram_request="80Gi",
        train_every=10000,
        sample_n=100,
        adapter="reason-annot",
        queue="reason-annot",
        sample_strategy="LatestWithRandom",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3",
            torch_dtype="bfloat16",
            max_length=16384,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
            rlhf_type="dpo",
        ),
        config=orign_config,
        labels={"skill": skill_id},
    )
    return val_dpo_buffer


def create_validation_annot_sft_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Validation annotation model for demonstrations"""

    val_sft_buffer = ReplayBuffer(
        name="validation-annot",
        vram_request="40Gi",
        train_every=100,
        sample_n=100,
        sample_strategy="LatestWithRandom",
        adapter="validation-annot",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3_offload",
            torch_dtype="bfloat16",
            max_length=16384,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
        ),
        config=orign_config,
        labels={"skill": skill_id},
    )
    return val_sft_buffer


def create_validation_annot_dpo_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Validation annotation model for demonstrations DPO"""

    val_dpo_buffer = ReplayBuffer(
        name="validation-annot-dpo",
        vram_request="80Gi",
        train_every=10000,
        sample_n=100,
        adapter="validation-annot",
        queue="validation-annot",
        sample_strategy="LatestWithRandom",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3",
            torch_dtype="bfloat16",
            max_length=16384,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
            rlhf_type="dpo",
        ),
        config=orign_config,
        labels={"skill": skill_id},
    )
    return val_dpo_buffer


def create_description_annot_sft_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Description annotation model for demonstrations"""

    val_sft_buffer = ReplayBuffer(
        name="description-annot",
        vram_request="40Gi",
        train_every=100,
        sample_n=100,
        sample_strategy="LatestWithRandom",
        adapter="description-annot",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3_offload",
            torch_dtype="bfloat16",
            max_length=16384,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
        ),
        config=orign_config,
        labels={"skill": skill_id},
    )
    return val_sft_buffer


def create_description_annot_dpo_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Description annotation model for demonstrations DPO"""

    val_dpo_buffer = ReplayBuffer(
        name="description-annot-dpo",
        vram_request="80Gi",
        train_every=10000,
        sample_n=100,
        adapter="description-annot",
        queue="description-annot",
        sample_strategy="LatestWithRandom",
        ms_swift_params=V1MSSwiftBufferParams(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type="qwen2_5_vl",
            train_type="lora",
            deepspeed="zero3",
            torch_dtype="bfloat16",
            max_length=16384,
            val_split_ratio=1.0,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            lora_rank=64,
            lora_alpha=128,
            size_factor=28,
            max_pixels=1025000,
            freeze_vit=True,
            rlhf_type="dpo",
        ),
        config=orign_config,
        labels={"skill": skill_id},
    )
    return val_dpo_buffer
