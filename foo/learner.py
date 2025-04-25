import logging
import os

import requests
from agentcore.models import V1UserProfile
from surfkit.auth.transport import get_current_user_sync
from surfkit.env import AGENTESEA_HUB_API_KEY_ENV
from surfkit.server.models import V1LearnTask
from surfkit.skill import Skill
from taskara import Task
from taskara.server.models import V1TaskUpdate
from tenacity import retry, stop_after_attempt, wait_fixed

from foo.agent import Foo as Agent

DEBUG_ENV_VAR = os.getenv("DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if DEBUG_ENV_VAR else logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


def learn_task(
    current_user: V1UserProfile,
    learn_model: V1LearnTask,
):
    task_model = learn_model.task
    logger.info(
        f"learning task: {task_model.model_dump()} with user {current_user.model_dump()}"
    )

    found = Task.find(
        remote=task_model.remote,
        id=task_model.id,
        owner_id=task_model.owner_id,
        auth_token=task_model.auth_token,
    )
    if not found:
        raise Exception(f"Task {task_model.id} not found")

    logger.info(f"found task: {found[0].to_v1().model_dump()}")

    task = found[0]
    task.remote = task_model.remote  # type: ignore
    task.auth_token = task_model.auth_token  # type: ignore

    skill_id = None
    if task.skill:
        skill_id = task.skill
    elif "skill" in task.labels:
        skill_id = task.labels["skill"]
    elif "skill_id" in task.labels:
        skill_id = task.labels["skill_id"]
    else:
        raise ValueError("Task skill or skill label not set")

    logger.info(f"finding skill_id: {skill_id}")
    print(f"current_user token: {task_model.auth_token}", flush=True)
    print(f"task.remote: {task.remote}", flush=True)
    skills = Skill.find(id=skill_id, remote=task.remote, token=task_model.auth_token)
    if not skills:
        raise ValueError(f"Skill not found: {skill_id}")
    skill = skills[0]
    logger.info(f"skill: {skill.to_v1().model_dump()}")
    v1_agent = learn_model.agent

    if v1_agent:
        config = Agent.config_type().model_validate(v1_agent.config)
        agent = Agent.from_config(config=config)
    else:
        agent = Agent.default()

    print(f"agent: {agent}", flush=True)

    if not task.remote or not task.auth_token:
        raise ValueError("Task remote and auth token must be set")

    try:
        print(f"labeling task as training: {task.id}", flush=True)
        _label_task(task.remote, task.auth_token, task, "foo/train/status", "training")  # type: ignore

        print("labeled task as training", flush=True)
        agent.learn_task(task, skill, current_user)

        print(f"labeling task as finished: {task.id}", flush=True)
        _label_task(task.remote, task.auth_token, task, "foo/train/status", "finished")  # type: ignore
        print("labeled task as finished", flush=True)
    except Exception as e:
        logger.error(f"error learning task: {e}")

        print(f"labeling task as error: {task.id}", flush=True)
        _label_task(task.remote, task.auth_token, task, "foo/train/status", "error")  # type: ignore

        _label_task(task.remote, task.auth_token, task, "foo/train/error", str(e))  # type: ignore
        print("labeled task as error", flush=True)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(10))
def get_remote_task(id: str, owner_id: str, server: str) -> Task:
    HUB_API_KEY = os.environ.get(AGENTESEA_HUB_API_KEY_ENV)
    if not HUB_API_KEY:
        raise Exception(f"${AGENTESEA_HUB_API_KEY_ENV} not set")

    logger.debug(f"connecting to remote task: {id} key: {HUB_API_KEY}")
    try:
        tasks = Task.find(
            id=id,
            remote=server,
            owner_id=owner_id,
        )
        if not tasks:
            raise Exception(f"Task {id} not found")
        logger.debug(f"got remote task: {tasks[0].__dict__}")
        return tasks[0]
    except Exception as e:
        logger.error(f"error getting remote task: {e}")
        raise e


def _label_task(remote: str, token: str, task: Task, key: str, value: str) -> None:
    """Label a task as trained

    Args:
        task (Task): The task
    """
    update = V1TaskUpdate(
        set_labels={key: value},
    )
    resp = requests.put(
        f"{remote}/v1/tasks/{task.id}",
        json=update.model_dump(),
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()


if __name__ == "__main__":
    import os

    task_json = os.getenv("LEARN_TASK_JSON")
    if not task_json:
        raise ValueError("LEARN_TASK_JSON not set")

    print(f"task_json: {task_json}", flush=True)

    v1learn = V1LearnTask.model_validate_json(task_json)
    token = os.getenv("AGENTSEA_API_KEY")
    if not token:
        raise ValueError("AGENTSEA_API_KEY not set")

    print(f"token: {token}", flush=True)

    user = get_current_user_sync(token)
    print(f"user: {user}", flush=True)

    learn_task(user, v1learn)
