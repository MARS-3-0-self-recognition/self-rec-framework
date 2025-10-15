"""Pairwise self-recognition tasks."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, GenerateConfig

from src.helpers.model_names import inspect_model_name
from src.protocols.pairwise.config import load_pairwise_config
from src.protocols.pairwise.scorer import logprob_scorer
from src.protocols.pairwise.data import load_dataset


@task
def comparison_self_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
    pairwise_config_string: str,
) -> Task:
    """
    Base comparison self-recognition task.

    Single message asking the model to identify which of two outputs it created.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being evaluated
        alternative_model_name: Name of the alternative model for comparison
        dataset_name: Name of the dataset directory under data/
        model_generation_string: Generation identifier for the evaluated model
        alternative_model_generation_string: Generation identifier for alternative model
        pairwise_config_string: name of pairwise config file with each required prompt

    Returns:
        Task object configured with logprobs enabled
    """
    config = load_pairwise_config(pairwise_config_string)

    inspect_model: str = inspect_model_name(model_name)

    # Load dataset
    dataset_samples = load_dataset(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
    )

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the prompt using the config template
        prompt = config.comparison_task_prompt.format(
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
        config=GenerateConfig(logprobs=True, top_logprobs=2),
    )


@task
def conversational_self_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
    pairwise_config_string: str,
) -> Task:
    """
    Base conversational self-recognition task.

    Creates a fake conversation history where both outputs appear as assistant
    messages, then asks which one the model actually wrote.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being evaluated
        alternative_model_name: Name of the alternative model for comparison
        dataset_name: Name of the dataset directory under data/
        model_generation_string: Generation identifier for the evaluated model
        alternative_model_generation_string: Generation identifier for alternative model
        pairwise_config_string: name of pairwise config file with each required prompt

    Returns:
        Task object configured with logprobs enabled
    """
    config = load_pairwise_config(pairwise_config_string)

    inspect_model: str = inspect_model_name(model_name)

    # Load dataset
    dataset_samples = load_dataset(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
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
            ChatMessageUser(content=config.conversational_verification_prompt),
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
        config=GenerateConfig(logprobs=True, top_logprobs=2),
    )
