"""Self-recognition tasks."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, GenerateConfig

from src.helpers.model_names import inspect_model_name
from src.inspect.config import load_self_recognition_config
from src.inspect.scorer import logprob_scorer, answer_length_scorer
from src.inspect.data import load_dataset_pairwise, load_dataset_individual

from src.helpers.utils import (
    data_dir,
    load_json,
)


@task
def pairwise_query(
    model_name: str,
    treatment_name_1: str,
    treatment_name_2: str,
    dataset_name: str,
    dataset_file_name_1: str,
    dataset_file_name_2: str,
    config_name: str,
) -> Task:
    """
    Base comparison self-recognition task.

    Single message asking the model to identify which of two outputs it created.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name_1: Name of the first treatment to compare
        treatment_name_2: Name of the second treatment to compare
        dataset_name: Name of the dataset directory under data/
        dataset_file_name_1: File identifier for the first treatment
        dataset_file_name_2: File identifier for the second treatment
        config_name: name of pairwise config file with each required prompt

    Returns:
        Task object configured with logprobs enabled
    """
    config = load_self_recognition_config(config_name)

    inspect_model: str = inspect_model_name(model_name)

    # Load dataset
    dataset_samples = load_dataset_pairwise(
        treatment_name_1,
        treatment_name_2,
        dataset_name,
        dataset_file_name_1,
        dataset_file_name_2,
    )

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the prompt using the config template
        prompt = config.SR_task_prompt.format(
            content=sample_data["content"],
            output1=sample_data["output1"],
            output2=sample_data["output2"],
        )

        inspect_samples.append(
            Sample(
                input=prompt,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(
            logprobs=True, top_logprobs=2, system_message=config.system_prompt
        ),
    )


@task
def pairwise_conversation_assistant_tags(
    model_name: str,
    treatment_name_1: str,
    treatment_name_2: str,
    dataset_name: str,
    dataset_file_name_1: str,
    dataset_file_name_2: str,
    config_name: str,
) -> Task:
    """
    Base conversational self-recognition task.

    Creates a fake conversation history where both outputs appear as assistant
    messages, then asks which one the model actually wrote.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name_1: Name of the first treatment to compare
        treatment_name_2: Name of the second treatment to compare
        dataset_name: Name of the dataset directory under data/
        dataset_file_name_1: File identifier for the first treatment
        dataset_file_name_2: File identifier for the second treatment
        config_name: name of pairwise config file with each required prompt

    Returns:
        Task object configured with logprobs enabled
    """
    config = load_self_recognition_config(config_name)

    inspect_model: str = inspect_model_name(model_name)

    # Load dataset
    dataset_samples = load_dataset_pairwise(
        treatment_name_1,
        treatment_name_2,
        dataset_name,
        dataset_file_name_1,
        dataset_file_name_2,
    )

    # Create Inspect samples with conversation history
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the generation prompt
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )

        # Build conversation history
        messages = [
            # First interaction with output1
            ChatMessageUser(content=generation_prompt),
            ChatMessageAssistant(content=sample_data["output1"]),
            # Second interaction with output2 (same article/question)
            ChatMessageUser(content=generation_prompt),
            ChatMessageAssistant(content=sample_data["output2"]),
            # Final verification question
            ChatMessageUser(content=config.SR_task_prompt),
        ]

        inspect_samples.append(
            Sample(
                input=messages,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(
            logprobs=True, top_logprobs=2, system_message=config.system_prompt
        ),
    )


@task
def pairwise_conversation_user_tags(
    model_name: str,
    treatment_name_1: str,
    treatment_name_2: str,
    dataset_name: str,
    dataset_file_name_1: str,
    dataset_file_name_2: str,
    config_name: str,
) -> Task:
    """
    Base comparison self-recognition task.

    Creates a fake transcript of a model conversation structured like pairwise_conversation_assistant_tags, but where all content appears in the user messages. Informs the model that the transcript is of a previous conversation it had, and asks it to identify which of the two responses it originally produced.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name_1: Name of the first treatment to compare
        treatment_name_2: Name of the second treatment to compare
        dataset_name: Name of the dataset directory under data/
        dataset_file_name_1: File identifier for the first treatment
        dataset_file_name_2: File identifier for the second treatment
        config_name: name of pairwise config file with each required prompt

    Returns:
        Task object configured with logprobs enabled
    """
    config = load_self_recognition_config(config_name)

    inspect_model: str = inspect_model_name(model_name)

    # Load dataset
    dataset_samples = load_dataset_pairwise(
        treatment_name_1,
        treatment_name_2,
        dataset_name,
        dataset_file_name_1,
        dataset_file_name_2,
    )

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the prompt using the config template
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )
        prompt = config.SR_task_prompt.format(
            generation_prompt=generation_prompt,
            output1=sample_data["output1"],
            output2=sample_data["output2"],
        )

        inspect_samples.append(
            Sample(
                input=prompt,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(
            logprobs=True, top_logprobs=2, system_message=config.system_prompt
        ),
    )


@task
def individual_conversation_assistant_tags(
    model_name: str,
    treatment_name: str,
    dataset_name: str,
    dataset_file_name: str,
    config_name: str,
) -> Task:
    """
    Base conversational self-recognition task.

    Creates a fake conversation history where the model's response appears as an assistant message, then asks it to evaluate if it was originally produced by the model or not.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name: Name of the treatment being evaluated
        dataset_name: Name of the dataset directory under data/
        dataset_file_name: File identifier for the treatment
        config_name: name of self-recognition config file with each required prompt

    Returns:
        Task object configured with logprobs enabled
    """
    config = load_self_recognition_config(config_name)

    inspect_model: str = inspect_model_name(model_name)

    # Load dataset
    dataset_samples = load_dataset_pairwise(
        treatment_name,
        dataset_name,
        dataset_file_name,
    )

    # Create Inspect samples with conversation history
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the generation prompt
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )

        # Build conversation history
        messages = [
            # First interaction with output1
            ChatMessageUser(content=generation_prompt),
            ChatMessageAssistant(content=sample_data["output"]),
            # Final verification question
            ChatMessageUser(content=config.SR_task_prompt),
        ]

        inspect_samples.append(
            Sample(
                input=messages,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(
            logprobs=True, top_logprobs=2, system_message=config.system_prompt
        ),
    )


@task
def individual_conversation_user_tags(
    model_name: str,
    treatment_name: str,
    dataset_name: str,
    dataset_file_name: str,
    config_name: str,
) -> Task:
    """
    Base comparison self-recognition task.

    Creates a fake transcript of a model conversation structured like individual_conversation_assistant_tags, but where all content appears in the user messages. Informs the model that the transcript is of a previous conversation it had, and asks it to evaluate if its response was originally produced by the model or not.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name: Name of the treatment being evaluated
        dataset_name: Name of the dataset directory under data/
        dataset_file_name: File identifier for the treatment
        config_name: name of self-recognition config file with each required prompt

    Returns:
        Task object configured with logprobs enabled
    """
    config = load_self_recognition_config(config_name)

    inspect_model: str = inspect_model_name(model_name)

    # Load dataset
    dataset_samples = load_dataset_individual(
        treatment_name,
        dataset_name,
        dataset_file_name,
    )

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the prompt using the config template
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )
        prompt = config.SR_task_prompt.format(
            generation_prompt=generation_prompt,
            output=sample_data["output"],
        )

        inspect_samples.append(
            Sample(
                input=prompt,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(
            logprobs=True, top_logprobs=2, system_message=config.system_prompt
        ),
    )


@task
def generation(
    model_name: str,
    treatment_name: str,
    dataset_name: str,
    dataset_file_name: str,
    config_name: str,
) -> Task:
    """
    Base generation task.
    """
    contents = load_json(data_dir() / dataset_name / "input.json")
    config = load_self_recognition_config(config_name)
    model = inspect_model_name(model_name)

    # create base samples
    inspect_samples = []
    for uuid in contents.keys():
        sample_input = contents[uuid]
        generation_prompt = config.generation_prompt.format(
            content=sample_input,
        )
        metadata = {
            "uuid": uuid,
            "model_name": model_name,
            "config_name": config_name,
        }
        inspect_samples.append(
            Sample(
                input=generation_prompt,
                id=uuid,
                metadata=metadata,
            )
        )
        # TODO: optionally add a target later

    # Create GenerateConfig with only non-None parameters
    generate_config_params = {}
    if config.system_prompt is not None:
        generate_config_params["system_message"] = config.system_prompt
    if config.temperature is not None:
        generate_config_params["temperature"] = config.temperature
    if config.max_tokens is not None:
        generate_config_params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        generate_config_params["top_p"] = config.top_p
    if config.top_k is not None:
        generate_config_params["top_k"] = config.top_k
    if config.seed is not None:
        generate_config_params["seed"] = config.seed
    if config.stop_seqs is not None:
        generate_config_params["stop_seqs"] = config.stop_seqs

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=answer_length_scorer(),
        model=model,
        config=GenerateConfig(**generate_config_params)
        if generate_config_params
        else None,
    )
