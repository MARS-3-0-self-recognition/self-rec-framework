"""High-level task definitions for pairwise self-recognition."""

from inspect_ai import task
from .config import get_summarisation_config, get_qa_config
from .task import prospective_self_recognition, conversational_self_recognition


# ============================================================================
# Article Summarization Tasks
# ============================================================================

@task
def prospective_summary_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str
):
    """
    Prospective self-recognition for article summarization.
    
    Single-message format where model is asked to identify its own summary.
    """
    return prospective_self_recognition(
        model_name=model_name,
        alternative_model_name=alternative_model_name,
        dataset_name=dataset_name,
        config=get_summarisation_config()
    )


@task
def conversational_summary_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str
):
    """
    Conversational self-recognition for article summarization.
    
    Multi-turn conversation format where both summaries appear as assistant
    messages before asking which one the model wrote.
    """
    return conversational_self_recognition(
        model_name=model_name,
        alternative_model_name=alternative_model_name,
        dataset_name=dataset_name,
        config=get_summarisation_config()
    )


# ============================================================================
# Question Answering Tasks
# ============================================================================

@task
def prospective_qa_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str
):
    """
    Prospective self-recognition for question answering.
    
    Single-message format where model is asked to identify its own answer.
    """
    return prospective_self_recognition(
        model_name=model_name,
        alternative_model_name=alternative_model_name,
        dataset_name=dataset_name,
        config=get_qa_config()
    )


@task
def conversational_qa_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str
):
    """
    Conversational self-recognition for question answering.
    
    Multi-turn conversation format where both answers appear as assistant
    messages before asking which one the model wrote.
    """
    return conversational_self_recognition(
        model_name=model_name,
        alternative_model_name=alternative_model_name,
        dataset_name=dataset_name,
        config=get_qa_config()
    )


# ============================================================================
# Example usage (for documentation):
# ============================================================================
# 
# inspect eval protocols/pairwise/tasks.py@prospective_summary_recognition \
#     --model openai/gpt-4 \
#     -T model_name=gpt-4 \
#     -T alternative_model_name=claude-3-5-sonnet \
#     -T dataset_name=news_articles
#
# inspect eval protocols/pairwise/tasks.py@conversational_qa_recognition \
#     --model anthropic/claude-3-5-sonnet \
#     -T model_name=claude-3-5-sonnet \
#     -T alternative_model_name=gpt-4 \
#     -T dataset_name=open_ended_questions