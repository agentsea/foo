from orign import ReplayBuffer
from orign.config import GlobalConfig


def create_actor_sft_buffer(
    name: str, skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Actor model for this skill"""

    actor_sft_buffer = ReplayBuffer(
        name=name,
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
        config=orign_config,
        labels={"skill": skill_id},
        train_every=100,
        owner="agentsea",
    )
    return val_sft_buffer


def create_validation_annot_sft_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Validation annotation model for demonstrations"""

    val_sft_buffer = ReplayBuffer(
        name="validation-annot",
        config=orign_config,
        labels={"skill": skill_id},
        train_every=100,
        owner="agentsea",
    )
    return val_sft_buffer


def create_description_annot_sft_buffer(
    skill_id: str, orign_config: GlobalConfig
) -> ReplayBuffer:
    """Description annotation model for demonstrations"""

    val_sft_buffer = ReplayBuffer(
        name="description-annot",
        config=orign_config,
        labels={"skill": skill_id},
        train_every=100,
        owner="agentsea",
    )
    return val_sft_buffer
