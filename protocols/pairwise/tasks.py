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
    conversational_self_recognition_from_csv,
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
# CSV-Based Tasks
# ============================================================================


@task
def two_turn_summary_recognition_csv(
    csv_path: str,
    article_text: str | None = None,
    system_prompt: str = "You are an expert WikiHow article summarizer. Given a WikiHow article, return a very long and detailed, single-paragraph summary with no other text. This will really help us better understand the article.",
):
    """
    Two-Turn (2T) self-recognition using CSV data.

    Loads data from a CSV file with columns: trial, source1, content1, source2, content2.
    Creates TWO samples per row (one for each ordering).

    This is designed for the data format in: data/experiments/pairwise/two-turn/

    Args:
        csv_path: Path to the CSV file (e.g., "data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv")
        article_text: Optional article text (if all trials use the same article, otherwise uses empty string)
        system_prompt: System prompt to use for the conversation (default: WikiHow summarizer)

    Example:
        inspect eval protocols/pairwise/tasks.py@two_turn_summary_recognition_csv \\
            --model anthropic/claude-3-5-haiku-20241022 \\
            -T csv_path=data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv
    """
    return conversational_self_recognition_from_csv(
        csv_path,
        get_two_turn_config(),
        article_text,
        system_prompt,
    )
