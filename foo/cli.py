import argparse
import logging
import os
import sys
from typing import Dict, Optional

from namesgenerator import get_random_name
from rich.console import Console
from rich.table import Table

try:
    from agentdesk.device_v1 import Desktop, DesktopInstance
    from surfkit.cli.util import (  # If creating local trackers
        tracker_addr_agent,
        tracker_addr_local,
    )
    from surfkit.config import AGENTSEA_HUB_API_URL, GlobalConfig
    from taskara import (  # Assuming TaskStatus might be useful for checking results
        Task,
        TaskStatus,
    )
    from taskara.runtime.base import Tracker

    from foo.agent import Agent as FooAgent  # Import your Foo agent
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print(
        "Please ensure surfkit, taskara, agentdesk, namesgenerator, and foo.agent are accessible."
    )
    raise

# Configure basic logging
DEBUG_ENV_VAR = os.getenv("DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if DEBUG_ENV_VAR else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("foo_cli")

console = Console()


def _determine_owner_and_token(config: GlobalConfig) -> tuple[str, Optional[str]]:
    owner = "anonymous@agentsea.ai"  # Default owner
    task_token: Optional[str] = None

    if config.api_key:
        task_token = config.api_key
        try:
            from surfkit.hub import HubAuth

            hub = (
                HubAuth()
            )  # Assumes HubAuth can be initialized without args if API key is in config
            user_info = hub.get_user_info(config.api_key)
            owner = user_info.email
            logger.info(f"Using Hub user: {owner} and API key for task token.")
        except Exception as e:
            logger.warning(
                f"Failed to get user info from Hub with API key: {e}. Using default owner and API key as token."
            )
    else:
        logger.info(
            "No Hub API key found in global config. Task will use default owner and no token unless tracker provides one."
        )
    return owner, task_token


def _setup_tracker(
    tracker_name: Optional[str],
    tracker_remote: Optional[str],
    auth_enabled: bool,  # For creating a new tracker
) -> tuple[
    Optional[str], Optional[str], Optional[str]
]:  # (tracker_agent_addr, tracker_local_addr, task_token_from_tracker)
    _tracker_agent_addr: Optional[str] = None
    _tracker_local_addr: Optional[str] = None
    _task_token_override: Optional[str] = None  # Token specifically from a tracker

    # Simplified agent_runtime_name for local tracker context, assuming "docker" or "process" if one is created
    # This is primarily for `tracker_addr_agent` if a local tracker object is obtained.
    # If using HUB or remote, this specific runtime name for the agent side of tracker might be less critical.
    assumed_agent_runtime_for_tracker_addr = "process"

    if tracker_name:
        logger.info(f"Attempting to find tracker by name: {tracker_name}")
        trackers = Tracker.find(name=tracker_name)
        if not trackers:
            raise ValueError(f"Tracker with name '{tracker_name}' not found.")
        _tracker_obj = trackers[0]
        _tracker_agent_addr = tracker_addr_agent(
            _tracker_obj, assumed_agent_runtime_for_tracker_addr
        )
        _tracker_local_addr = tracker_addr_local(_tracker_obj)
        # If tracker has its own auth, _task_token_override might come from _tracker_obj.auth_token if applicable
        logger.info(
            f"Using tracker '{_tracker_obj.name}'. Agent-side addr: {_tracker_agent_addr}, Local addr: {_tracker_local_addr}"
        )

    elif tracker_remote:
        logger.info(f"Using remote tracker address: {tracker_remote}")
        _tracker_agent_addr = tracker_remote
        _tracker_local_addr = tracker_remote
    else:
        # Fallback: Check Hub config or create a local one (simplified from original)
        config = GlobalConfig.read()
        if config.api_key:
            logger.info(f"Using Agentsea Hub as tracker: {AGENTSEA_HUB_API_URL}")
            _tracker_agent_addr = AGENTSEA_HUB_API_URL
            _tracker_local_addr = AGENTSEA_HUB_API_URL
            _task_token_override = config.api_key  # Hub API key acts as the task token
        else:
            # Simplified: Try to create a default local Docker tracker if no other option
            # This part requires DockerTrackerRuntime and get_random_name
            logger.info(
                "No tracker specified and no Hub API key. Attempting to create a local Docker tracker."
            )
            try:
                from taskara.runtime.docker import DockerTrackerRuntime

                task_runt = DockerTrackerRuntime()
                local_tracker_name = get_random_name(sep="-")
                if not local_tracker_name:
                    raise SystemError("Failed to generate name for local tracker")

                _tracker_obj = task_runt.run(
                    name=local_tracker_name, auth_enabled=auth_enabled
                )
                logger.info(
                    f"Local tracker '{local_tracker_name}' created using Docker."
                )
                _tracker_agent_addr = tracker_addr_agent(
                    _tracker_obj, assumed_agent_runtime_for_tracker_addr
                )  # "docker" or "process"
                _tracker_local_addr = tracker_addr_local(_tracker_obj)
                # Potentially _task_token_override = _tracker_obj.auth_token if created with auth and it provides one
            except Exception as e:
                logger.error(
                    f"Failed to create local Docker tracker: {e}. Proceeding without a specific tracker."
                )
                # No specific tracker, tasks might be local / transient without a remote unless foo.agent handles it.
                # In this case, task.remote might be None or a default.

    if not _tracker_local_addr:
        logger.warning(
            "Tracker could not be determined. Task remote address will be None."
        )
        # Task objects handle remote=None

    return _tracker_agent_addr, _tracker_local_addr, _task_token_override


def _solve_with_foo_agent(
    description: str,
    device_name: str,
    max_steps: int = 30,
    tracker_name: Optional[str] = None,
    tracker_remote: Optional[str] = None,
    starting_url: Optional[str] = None,
    parent_id: Optional[str] = None,
    auth_enabled_tracker: bool = False,  # For creating a tracker if necessary
    skill_id: Optional[str] = None,
    # debug_foo_agent: bool = False # If FooAgent has its own debug handling
) -> Task:
    """
    Solves a task using the local Foo agent.
    Adapted from surfkit.client.solve
    """

    # 1. Setup Tracker
    # The 'agent_runtime' parameter for tracker_addr_agent isn't directly applicable here
    # as we're not running an external agent process. We'll assume a default or derive if needed.
    _tracker_agent_addr, _tracker_local_addr, _task_token_from_tracker = _setup_tracker(
        tracker_name, tracker_remote, auth_enabled_tracker
    )

    # 2. Determine Owner and Global Task Token
    global_config = GlobalConfig.read()
    owner, global_task_token = _determine_owner_and_token(global_config)

    # Prioritize token from tracker if available, else use global (e.g., Hub API key)
    final_task_token = (
        _task_token_from_tracker if _task_token_from_tracker else global_task_token
    )

    # 3. Setup Device
    logger.info(f"Finding device by name: '{device_name}'...")
    # Assuming Desktop.find returns a list of Desktop instances
    desktop_instances: list[DesktopInstance] = Desktop.find(name=device_name)
    if not desktop_instances:
        raise ValueError(f"Device '{device_name}' not found.")

    # Assuming the found instance is already a usable Desktop object or DesktopInstance
    # If Desktop.find() returns DesktopInstance, let's type it explicitly
    actual_device_instance: DesktopInstance = desktop_instances[
        0
    ]  # Keep as Desktop; DesktopInstance should be a subtype or compatible
    logger.info(
        f"Using device: {actual_device_instance.name if hasattr(actual_device_instance, 'name') else device_name}"
    )

    agent_device = Desktop.from_instance(actual_device_instance)

    # 4. Instantiate Foo Agent
    logger.info("Instantiating Foo agent...")
    # Assuming FooAgent is the class imported from foo.agent (e.g., class Agent(TaskAgent): ...)
    # And it has a default constructor or a ::default() classmethod
    try:
        if hasattr(FooAgent, "default") and callable(FooAgent.default):
            foo_agent_instance = FooAgent.default()
        else:
            foo_agent_instance = FooAgent()  # type: ignore
    except Exception as e:
        logger.error(f"Could not instantiate FooAgent: {e}")
        raise
    logger.info(f"Foo agent instantiated: {foo_agent_instance.name()}")

    # 5. Create Task Object
    params: Dict[str, str] = {}
    if starting_url:
        params["site"] = starting_url

    labels: Dict[str, str] = {}
    if skill_id:
        labels["skill"] = skill_id
        labels["skill_id"] = (
            skill_id  # Keep skill_id key for clarity or other potential uses
        )
    else:
        labels["skill"] = "test-skill"

    print("using skill_id: ", skill_id)

    logger.info(f"Creating Task object for description: '{description}'")
    task_obj = Task(
        description=description,
        parameters=params,
        max_steps=max_steps,
        assigned_to=foo_agent_instance.name(),  # Foo agent's name
        assigned_type=foo_agent_instance.__class__.__name__,  # Type of the Foo agent
        remote=_tracker_local_addr,  # Local access address for the tracker
        owner_id=owner,
        auth_token=final_task_token,
        parent_id=parent_id,
        labels=labels,
        skill=skill_id,
    )
    # If task needs agent-side tracker address, it's often set before sending to an agent
    # For a direct call, foo_agent_instance.solve_task might use task_obj.remote internally if needed
    # or one could set task_obj._remote = _tracker_agent_addr if that's a pattern foo.agent.solve_task expects
    if (
        _tracker_agent_addr
    ):  # Ensure agent has a way to reach tracker if it needs to (e.g. for task.save())
        task_obj._remote = _tracker_agent_addr

    # 6. Call Foo Agent's solve_task method
    logger.info(
        f"Calling {foo_agent_instance.name()}.solve_task for task ID {task_obj.id}..."
    )

    # Ensure the foo_agent_instance.solve_task signature matches:
    # def solve_task(self, task: Task, device: Optional[Device], max_steps: int) -> Task:
    # The 'device' passed should be the actual Desktop object, not its V1 representation.

    returned_task = foo_agent_instance.solve_task(
        task=task_obj,
        device=agent_device,  # Pass the live Desktop object
        max_steps=max_steps,
    )
    logger.info(
        f"Foo agent's solve_task method completed for task ID {returned_task.id}."
    )
    return returned_task


def solve(cli_args: argparse.Namespace):
    """
    Solves a task directly using the local Foo agent.
    """
    # Parameters are now accessed from cli_args
    task_description = cli_args.task
    device_name_arg = cli_args.device
    max_steps_arg = cli_args.max_steps
    tracker_name_arg = cli_args.tracker
    tracker_remote_arg = cli_args.tracker_remote
    starting_url_arg = cli_args.starting_url
    parent_id_arg = cli_args.parent_id
    auth_enabled_tracker_arg = cli_args.auth_tracker
    skill_id_arg = cli_args.skill_id
    # debug_foo_arg = cli_args.debug_foo # if you add this option to argparse

    logger.info(
        f"CLI invoked to solve task: '{task_description}' with device '{device_name_arg}'"
    )

    try:
        task_result = _solve_with_foo_agent(
            description=task_description,
            device_name=device_name_arg,
            max_steps=max_steps_arg,
            tracker_name=tracker_name_arg,
            tracker_remote=tracker_remote_arg,
            starting_url=starting_url_arg,
            parent_id=parent_id_arg,
            auth_enabled_tracker=auth_enabled_tracker_arg,
            skill_id=skill_id_arg,
            # debug_foo_agent=debug_foo_arg # Pass if _solve_with_foo_agent handles it
        )

        console.print("[bold green]Task Execution Result:[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="dim")
        table.add_column("Value")

        table.add_row("Task ID", task_result.id if task_result.id else "N/A")
        table.add_row("Description", task_result.description)
        status_val = "N/A"
        if task_result.status:
            status_val = (
                str(task_result.status.value)
                if hasattr(task_result.status, "value")
                else str(task_result.status)
            )
        table.add_row("Status", status_val)
        table.add_row(
            "Assigned To", task_result.assigned_to if task_result.assigned_to else "N/A"
        )
        table.add_row(
            "Remote Tracker", task_result.remote if task_result.remote else "N/A"
        )

        if task_result.error:
            table.add_row("[bold red]Error[/bold red]", str(task_result.error))

        console.print(table)

        if (
            task_result.error
            or (task_result.status and "fail" in status_val.lower())
            or (task_result.status == TaskStatus.ERROR)
        ):
            console.print(
                "[bold red]Task processing encountered an error or failed.[/bold red]"
            )
            sys.exit(1)
        else:
            console.print(
                "[bold green]Task processing by Foo agent completed.[/bold green]"
            )

    except ValueError as ve:
        logger.error(f"Configuration or input error: {ve}", exc_info=True)
        console.print(f"[bold red]Error: {ve}[/bold red]")
        sys.exit(1)
    except ImportError as ie:
        logger.error(f"Import error during execution: {ie}", exc_info=True)
        console.print(
            f"[bold red]Import Error: {ie}. Please ensure all dependencies are installed correctly.[/bold red]"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI to solve tasks using a local Foo agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t", "--task", required=True, help="Description of the task to solve."
    )
    parser.add_argument(
        "-d", "--device", required=True, help="Name of the existing device to use."
    )
    parser.add_argument(
        "--max-steps", type=int, default=30, help="Max steps allowed to solve the task."
    )
    parser.add_argument(
        "--tracker", default=None, help="Name of an existing tracker to use."
    )
    parser.add_argument(
        "--tracker-remote", default=None, help="Remote address of a tracker."
    )
    parser.add_argument(
        "--starting-url",
        default=None,
        help="Starting URL for the task (if applicable).",
    )
    parser.add_argument("--parent-id", default=None, help="Parent ID for the task.")
    parser.add_argument(
        "--auth-tracker",
        action="store_true",  # Defaults to False
        help="Enable auth if creating a local tracker.",
    )
    parser.add_argument("--skill-id", default=None, help="Skill ID for the task.")
    # The --debug-foo option was commented out in the original Typer app,
    # so it's omitted here unless requested.
    # parser.add_argument(
    #     "--debug-foo",
    #     action="store_true",
    #     help="Enable debug mode for the Foo agent (if supported)."
    # )

    args = parser.parse_args()
    solve(args)
