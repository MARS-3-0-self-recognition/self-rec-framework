"""Custom exceptions for the framework.

Centralized here so they can be raised and caught from anywhere without needing to
know which module happens to define them. tasks.py re-exports these for backward
compatibility with existing `from ...inspect.tasks import ...` call sites.
"""


class AlwaysReasoningModelError(Exception):
    """Raised when trying to use an always-reasoning model without -thinking suffix."""

    pass


class ReasoningTokensRequiredError(Exception):
    """Raised when reasoning is enabled but the required CoT token budget is unset."""

    pass


class ContextWindowExceededError(Exception):
    """Raised when a model's context window can't fit the estimated input plus a
    minimal output budget — i.e. the evaluator genuinely can't run this task. The
    sweep catches this to skip the model gracefully rather than abort."""

    pass
