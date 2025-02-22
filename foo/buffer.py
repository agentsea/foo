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
            save_strategy="epoch",
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
            save_strategy="epoch",
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
            save_strategy="epoch",
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
            save_strategy="epoch",
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
            save_strategy="epoch",
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
            save_strategy="epoch",
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
            save_strategy="epoch",
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
