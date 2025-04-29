def create_description_prompt(
    action: str, image1_url: str, image2_url: str, answer: str | None = None
) -> dict:
    """
    Create an OpenAI format prompt describing what's happened between two images.
    """
    prompt_text = (
        f"The first image is the before image, and the second image is the after\n"
        f"image of a GUI interaction. The action that occurred is: {action}. "
        "Can you give a task description for what was accomplished?\n"
        "The goal would be for an agent to look at the first image and the task "
        "description which would result in the second image, for example "
        '"click on login button" would be a good description, or '
        '"move mouse to be over user icon", or "type text \'good fellas\'"\n'
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image1_url}},
                {"type": "image_url", "image_url": {"url": image2_url}},
            ],
        }
    ]

    if answer:
        messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}


def create_reason_prompt(
    action: str,
    task_description: str,
    image1_url: str,
    image2_url: str,
    answer: str | None = None,
) -> dict:
    """
    Create an OpenAI format prompt describing the reasoning chain needed to connect an action
    and a desired task outcome.
    """
    prompt_text = (
        f"The first image is the before image, and the second image is the after\n"
        f"image of a GUI interaction. The action that occurred is: {action}. "
        "Can you give a reasoning chain for what the user would need to think\n"
        "through in order to take the correct action with respect to the task? "
        f"The current task is: {task_description}\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image1_url}},
                {"type": "image_url", "image_url": {"url": image2_url}},
            ],
        }
    ]

    if answer:
        messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}


def create_validation_prompt(
    action: str,
    task_description: str,
    image1_url: str,
    image2_url: str,
    answer: str | None = None,
) -> dict:
    """
    Create an OpenAI format prompt asking the LLM to validate whether the action completed
    successfully for a given task.
    """
    prompt_text = (
        f"The first image is the before image, and the second image is the after\n"
        f"image of a GUI interaction. The action that occurred is: {action}. "
        "Considering the task we want to accomplish,\n"
        "please give me the reason why this action completed successfully or not. "
        f"The current task is: {task_description}\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image1_url}},
                {"type": "image_url", "image_url": {"url": image2_url}},
            ],
        }
    ]

    if answer:
        messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}


def create_actor_prompt(
    content: str, image_url: str, response: str | None = None
) -> dict:
    """
    Create an OpenAI format prompt for actor training.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    if response:
        messages.append({"role": "assistant", "content": response})

    return {"messages": messages}


def create_reason_actor_prompt(
    content: str, image_url: str, response: str | None = None
) -> dict:
    """
    Create an OpenAI format prompt for actor reasoning training.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    if response:
        messages.append({"role": "assistant", "content": response})

    return {"messages": messages}


def create_validation_actor_prompt(
    val_ctx: str, image1_url: str, image2_url: str, response: str | None = None
) -> dict:
    """
    Create an OpenAI format prompt for validation.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": val_ctx},
                {"type": "image_url", "image_url": {"url": image1_url}},
                {"type": "image_url", "image_url": {"url": image2_url}},
            ],
        }
    ]

    if response:
        messages.append({"role": "assistant", "content": response})

    return {"messages": messages}


def create_validation_reasoning_prompt(
    val_ctx_reason: str,
    image1_url: str,
    image2_url: str,
    response: str | None = None,
    rejected_response: str | None = None,
) -> dict:
    """
    Create an OpenAI format prompt for validation reasoning.
    """
    result: dict = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": val_ctx_reason},
                    {"type": "image_url", "image_url": {"url": image1_url}},
                    {"type": "image_url", "image_url": {"url": image2_url}},
                ],
            }
        ]
    }

    if response:
        result["messages"].append({"role": "assistant", "content": response})

    if rejected_response:
        result["rejected_response"] = rejected_response

    return result
