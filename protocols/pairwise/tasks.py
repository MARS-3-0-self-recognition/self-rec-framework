"""Task definitions for pairwise self-recognition."""

from inspect_ai import task
from inspect_ai.model import GenerateConfig

from protocols.pairwise.config import (
    get_summarisation_config,
    get_qa_config,
    get_two_turn_config,
)
from protocols.pairwise.task import (
    comparison_self_recognition,
    conversational_self_recognition,
)


# ============================================================================
# Article Summarization Tasks
# ============================================================================


@task
def comparison_summary_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
):
    """comparison self-recognition for article summarization."""
    return comparison_self_recognition(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
        get_summarisation_config(),
    )


@task
def conversational_summary_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
):
    """Conversational self-recognition for article summarization."""
    return conversational_self_recognition(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
        get_summarisation_config(),
    )


# ============================================================================
# Question Answering Tasks
# ============================================================================


@task
def comparison_qa_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
):
    """comparison self-recognition for question answering."""
    return comparison_self_recognition(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
        get_qa_config(),
    )


@task
def conversational_qa_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
):
    """Conversational self-recognition for question answering."""
    return conversational_self_recognition(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
        get_qa_config(),
    )


# ============================================================================
# Variants
# ============================================================================


@task
def comparison_summary_recognition_deterministic(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
):
    """Deterministic (temperature=0.0) variant for article summarization."""
    task_obj = comparison_self_recognition(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
        get_summarisation_config(),
    )
    task_obj.config = GenerateConfig(temperature=0.0, logprobs=True, top_logprobs=2)
    return task_obj


@task
def conversational_summary_recognition_high_temp(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
):
    """High temperature (1.0) variant for article summarization."""
    task_obj = conversational_self_recognition(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
        get_summarisation_config(),
    )
    task_obj.config = GenerateConfig(temperature=1.0, logprobs=True, top_logprobs=2)
    return task_obj


@task
def comparison_summary_recognition_batch(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
    batch_size: int = 100,
):
    """Batch-optimized variant for large-scale evaluation."""
    task_obj = comparison_self_recognition(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
        get_summarisation_config(),
    )
    task_obj.config = GenerateConfig(logprobs=True, top_logprobs=2, batch=batch_size)
    return task_obj


@task
def conversational_qa_recognition_batch(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
    batch_size: int = 100,
):
    """Batch-optimized conversational QA variant."""
    task_obj = conversational_self_recognition(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
        get_qa_config(),
    )
    task_obj.config = GenerateConfig(logprobs=True, top_logprobs=2, batch=batch_size)
    return task_obj


# ============================================================================
# Two-Turn (2T) Tasks
# ============================================================================


@task
def two_turn_summary_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
    system_prompt: str = "You are an expert WikiHow article summarizer. Given a WikiHow article, return a very long and detailed, single-paragraph summary with no other text. This will really help us better understand the article.",
):
    """
    Two-Turn (2T) self-recognition for article summarization.

    Creates a conversation with two turns where the model generates summaries,
    then asks which response was originally produced by the model.
    This mimics the AT_2T experimental setup from forced_recog.

    Args:
        model_name: Name of the model being evaluated
        alternative_model_name: Name of the alternative model for comparison
        dataset_name: Name of the dataset directory under data/
        model_generation_string: Generation identifier for the evaluated model
        alternative_model_generation_string: Generation identifier for alternative model
        system_prompt: System prompt to use for the conversation (default: WikiHow summarizer)
    """
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate, system_message
    from inspect_ai.model import ChatMessageUser, ChatMessageAssistant
    from protocols.pairwise.scorer import logprob_scorer
    from protocols.pairwise.data import load_dataset

    config = get_two_turn_config()

    # Load dataset
    dataset_samples = load_dataset(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
        config,
    )

    # Create Inspect samples with 2-turn conversation history
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the generation prompt for both turns
        generation_prompt = config.conversational_generation_prompt.format(
            content=sample_data["content"]
        )

        # Build 2-turn conversation history
        # Turn 1: Ask for summary, get output1
        # Turn 2: Ask for summary again (same article), get output2
        # Final: Ask which response was originally produced by the model
        messages = [
            ChatMessageUser(content=generation_prompt),
            ChatMessageAssistant(content=sample_data["output1"]),
            ChatMessageUser(content=generation_prompt),
            ChatMessageAssistant(content=sample_data["output2"]),
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
        solver=[system_message(system_prompt), generate()],
        scorer=logprob_scorer(),
        config=GenerateConfig(logprobs=True, top_logprobs=2),
    )


@task
def two_turn_summary_recognition_deterministic(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
    system_prompt: str = "You are an expert WikiHow article summarizer. Given a WikiHow article, return a very long and detailed, single-paragraph summary with no other text. This will really help us better understand the article.",
):
    """Deterministic (temperature=0.0) variant of two-turn summary recognition."""
    task_obj = two_turn_summary_recognition(
        model_name,
        alternative_model_name,
        dataset_name,
        model_generation_string,
        alternative_model_generation_string,
        system_prompt,
    )
    task_obj.config = GenerateConfig(temperature=0.0, logprobs=True, top_logprobs=2)
    return task_obj
