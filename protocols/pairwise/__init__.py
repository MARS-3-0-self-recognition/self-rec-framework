"""Pairwise self-recognition evaluation tasks."""

from .config import PairwiseConfig, load_config, get_summarisation_config, get_qa_config
from .scorer import logprob_scorer
from .task import prospective_self_recognition, conversational_self_recognition
from .tasks import (
    prospective_summary_recognition,
    conversational_summary_recognition,
    prospective_qa_recognition,
    conversational_qa_recognition
)

__all__ = [
    # Config
    "PairwiseConfig",
    "load_config",
    "get_summarisation_config",
    "get_qa_config",
    # Scorer
    "logprob_scorer",
    # Base tasks
    "prospective_self_recognition",
    "conversational_self_recognition",
    # Specific tasks
    "prospective_summary_recognition",
    "conversational_summary_recognition",
    "prospective_qa_recognition",
    "conversational_qa_recognition",
]