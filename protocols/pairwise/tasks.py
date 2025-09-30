"""Task definitions for pairwise self-recognition."""

from inspect_ai import task
from inspect_ai.model import GenerateConfig

from .config import get_summarisation_config, get_qa_config
from .task import prospective_self_recognition, conversational_self_recognition


# ============================================================================
# Article Summarization Tasks
# ============================================================================

@task
def prospective_summary_recognition(model_name: str, alternative_model_name: str, dataset_name: str):
    """Prospective self-recognition for article summarization."""
    return prospective_self_recognition(model_name, alternative_model_name, dataset_name, get_summarisation_config())


@task
def conversational_summary_recognition(model_name: str, alternative_model_name: str, dataset_name: str):
    """Conversational self-recognition for article summarization."""
    return conversational_self_recognition(model_name, alternative_model_name, dataset_name, get_summarisation_config())


# ============================================================================
# Question Answering Tasks
# ============================================================================

@task
def prospective_qa_recognition(model_name: str, alternative_model_name: str, dataset_name: str):
    """Prospective self-recognition for question answering."""
    return prospective_self_recognition(model_name, alternative_model_name, dataset_name, get_qa_config())


@task
def conversational_qa_recognition(model_name: str, alternative_model_name: str, dataset_name: str):
    """Conversational self-recognition for question answering."""
    return conversational_self_recognition(model_name, alternative_model_name, dataset_name, get_qa_config())


# ============================================================================
# Variants
# ============================================================================

@task
def prospective_summary_recognition_deterministic(model_name: str, alternative_model_name: str, dataset_name: str):
    """Deterministic (temperature=0.0) variant for article summarization."""
    task_obj = prospective_self_recognition(model_name, alternative_model_name, dataset_name, get_summarisation_config())
    task_obj.config = GenerateConfig(temperature=0.0, logprobs=True, top_logprobs=5)
    return task_obj


@task
def conversational_summary_recognition_high_temp(model_name: str, alternative_model_name: str, dataset_name: str, temperature: float = 1.0):
    """High temperature (1.0) variant for article summarization."""
    task_obj = conversational_self_recognition(model_name, alternative_model_name, dataset_name, get_summarisation_config())
    task_obj.config = GenerateConfig(temperature=temperature, logprobs=True, top_logprobs=5)
    return task_obj


@task
def prospective_summary_recognition_batch(model_name: str, alternative_model_name: str, dataset_name: str, batch_size: int = 100):
    """Batch-optimized variant for large-scale evaluation."""
    task_obj = prospective_self_recognition(model_name, alternative_model_name, dataset_name, get_summarisation_config())
    task_obj.config = GenerateConfig(logprobs=True, top_logprobs=5, batch=batch_size)
    return task_obj


@task
def conversational_qa_recognition_batch(model_name: str, alternative_model_name: str, dataset_name: str, batch_size: int = 100):
    """Batch-optimized conversational QA variant."""
    task_obj = conversational_self_recognition(model_name, alternative_model_name, dataset_name, get_qa_config())
    task_obj.config = GenerateConfig(logprobs=True, top_logprobs=5, batch=batch_size)
    return task_obj