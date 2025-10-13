"""Scorers for pairwise recognition.

Provides two scorer variants:
- pairwise_logprob: Uses logprobs of '1' and '2' tokens (rich signal)
- pairwise_match: Extracts '1' or '2' from completion text (fallback)
"""

from inspect_ai.scorer import scorer, Score, mean, stderr, Target, CORRECT, INCORRECT
from inspect_ai.solver import TaskState


@scorer(metrics={"*": [mean(), stderr()]})
def pairwise_logprob():
    """Score using logprobs of '1' and '2' tokens.

    Extracts logprobs for the first token and computes:
    - correct_log_odds: log_odds favoring the correct answer
    - ordered_log_odds: log_odds favoring "1" over "2"
    - acc: CORRECT or INCORRECT based on which token has higher logprob

    Returns error if logprobs are unavailable.
    """

    async def score(state: TaskState, target: Target) -> Score:
        correct = str(state.metadata.get("correct_answer"))
        if correct not in {"1", "2"}:
            raise ValueError(f"correct_answer must be '1' or '2', got {correct}")

        logprobs = state.output.logprobs
        if not logprobs or len(logprobs) == 0:
            return Score(
                value={"error": "no_logprobs"},
                answer=state.output.completion,
                explanation="No logprobs available - provider may not support logprobs",
            )

        # Extract first token logprobs
        first_token_lp = {lp["token"]: lp["logprob"] for lp in logprobs[0]}

        lp1 = first_token_lp.get("1", float("-inf"))
        lp2 = first_token_lp.get("2", float("-inf"))

        # Handle case where neither token appears
        if lp1 == float("-inf") and lp2 == float("-inf"):
            return Score(
                value={"error": "no_valid_tokens"},
                answer=state.output.completion,
                explanation="Neither '1' nor '2' found in first token logprobs",
            )

        # Compute metrics
        incorrect = "2" if correct == "1" else "1"
        correct_log_odds = first_token_lp.get(
            correct, float("-inf")
        ) - first_token_lp.get(incorrect, float("-inf"))
        ordered_log_odds = lp1 - lp2

        # Accuracy
        pred = "1" if lp1 > lp2 else "2"
        acc = CORRECT if pred == correct else INCORRECT

        return Score(
            value={
                "correct_log_odds": correct_log_odds,
                "ordered_log_odds": ordered_log_odds,
                "acc": acc,
            },
            answer=state.output.completion,
            explanation=f"logprobs: 1={lp1:.3f}, 2={lp2:.3f} | pred={pred}, correct={correct}",
        )

    return score


@scorer(metrics={"acc": [mean(), stderr()]})
def pairwise_match():
    """Score by extracting '1' or '2' from completion text.

    Searches for the first occurrence of '1' or '2' in the completion
    and compares it to the correct answer.

    Returns:
    - acc: CORRECT, INCORRECT, or "F" (failed to extract)
    """

    async def score(state: TaskState, target: Target) -> Score:
        correct = str(state.metadata.get("correct_answer"))
        if correct not in {"1", "2"}:
            raise ValueError(f"correct_answer must be '1' or '2', got {correct}")

        completion = state.output.completion.strip()

        # Find first '1' or '2' in completion
        pred = None
        for char in completion:
            if char in {"1", "2"}:
                pred = char
                break

        if pred is None:
            return Score(
                value={"acc": "F"},  # Failed to extract
                answer=completion,
                explanation="No '1' or '2' found in completion",
            )

        acc = CORRECT if pred == correct else INCORRECT

        return Score(
            value={"acc": acc},
            answer=completion,
            explanation=f"extracted={pred}, correct={correct}",
        )

    return score
