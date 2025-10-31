# scorers.py

import sys
from inspect_ai.scorer import scorer, Score, mean, stderr, Target
from inspect_ai.solver import TaskState


@scorer(
    metrics={
        "*": [mean(), stderr()]  # apply mean + stderr to all metrics
    }
)
def logprob_scorer():
    """
    Custom scorer for binary-choice evals ("1" or "2").

    Reports:
      - correct_log_odds: logprob(correct) - logprob(incorrect)
      - ordered_log_odds: logprob("1") - logprob("2")
      - acc: "C" (correct) or "I" (incorrect)

    Fallback if logprobs are missing:
      - Look at the generated text and extract the first "1" or "2".
      - If none found, fail the sample.
    """

    async def score(state: TaskState, target: Target) -> Score:
        correct = str(state.metadata.get("correct_answer"))
        if correct not in {"1", "2"}:
            raise ValueError(f"correct_answer must be '1' or '2', got {correct}")

        # logprobs = state.output.logprobs

        # # Case 1: logprobs available
        # if logprobs and len(logprobs) > 0:
        #     first_token_logprobs = logprobs[0]
        #     token_lp = {lp["token"]: lp["logprob"] for lp in first_token_logprobs}

        #     lp1 = token_lp.get("1", float("-inf"))
        #     lp2 = token_lp.get("2", float("-inf"))

        #     incorrect = "2" if correct == "1" else "1"

        #     correct_log_odds = token_lp.get(correct, float("-inf")) - token_lp.get(incorrect, float("-inf"))
        #     ordered_log_odds = lp1 - lp2

        #     # Determine accuracy
        #     pred = "1" if lp1 > lp2 else "2"
        #     acc = "C" if pred == correct else "I"

        #     return Score(
        #         value={
        #             "correct_log_odds": correct_log_odds,
        #             "ordered_log_odds": ordered_log_odds,
        #             "acc": acc,
        #         },
        #         answer=state.output.completion,
        #         explanation=f"[logprobs] 1={lp1}, 2={lp2}, correct={correct}, pred={pred}"
        #     )

        # sys.stderr.write("[fallback] No logprobs available, using completion tokens.\n")

        # Case 2: fallback to generated text
        completion = state.output.completion.strip()

        # First check the very first token
        first_char = completion[0] if completion else ""
        if first_char in {"1", "2"}:
            pred = first_char
        else:
            # Scan for first occurrence of "1" or "2"
            pred = None
            for ch in completion:
                if ch in {"1", "2"}:
                    pred = ch
                    sys.stderr.write(
                        f"[fallback] Found answer later in completion: {pred}\n"
                    )
                    break

        if pred is None:
            sys.stderr.write(
                "[fallback] Could not find '1' or '2' in completion. Marking as failure.\n"
            )
            return Score(
                value={"acc": "F"}, answer=completion, explanation="No valid prediction"
            )

        acc = "C" if pred == correct else "I"

        return Score(
            value={"acc": acc},
            answer=completion,
            explanation=f"[fallback] pred={pred}, correct={correct}",
        )

    return score
