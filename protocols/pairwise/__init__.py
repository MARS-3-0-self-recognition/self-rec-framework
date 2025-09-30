"""Pairwise self-recognition evaluation tasks."""

from .config import PairwiseConfig, load_config, get_summarisation_config, get_qa_config
from .data import load_dataset
from .scorer import logprob_scorer
from .task import prospective_self_recognition, conversational_self_recognition
from .tasks import (
    prospective_summary_recognition,
    conversational_summary_recognition,
    prospective_qa_recognition,
    conversational_qa_recognition,
    prospective_summary_recognition_deterministic,
    conversational_summary_recognition_high_temp,
)

__all__ = [
    # Config
    "PairwiseConfig",
    "load_config",
    "get_summarisation_config",
    "get_qa_config",
    # Data
    "load_dataset",
    # Scorer
    "logprob_scorer",
    # Base tasks (not decorated, return Task objects)
    "prospective_self_recognition",
    "conversational_self_recognition",
    # Specific tasks (decorated with @task)
    "prospective_summary_recognition",
    "conversational_summary_recognition",
    "prospective_qa_recognition",
    "conversational_qa_recognition",
    # Example variants
    "prospective_summary_recognition_deterministic",
    "conversational_summary_recognition_high_temp",
]