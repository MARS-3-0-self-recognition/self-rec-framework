# scorers.py

import sys
from inspect_ai.scorer import scorer, Score, mean, stderr, std, Target
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


@scorer(metrics=[mean(), std()])
def answer_length_scorer():
    """Scorer that records answer length and optionally captures reasoning/CoT."""

    def _extract_reasoning(state: TaskState) -> tuple[str | None, str | None]:
        """
        Return concatenated reasoning blocks and signature (if any) from the model output.

        Handles different model providers:
        - Anthropic: reasoning in message.content with signature
        - OpenAI o-series: reasoning in message.content (may be empty if reasoning_summary not enabled)
        - Together AI (Qwen, DeepSeek): reasoning in message.content

        Returns:
            Tuple of (reasoning_text, signature) where signature may be None
        """
        try:
            message = state.output.message
        except Exception:
            # Fallback: try accessing via choices (for OpenAI models)
            try:
                choices = getattr(state.output, "choices", None)
                if choices and len(choices) > 0:
                    message = getattr(choices[0], "message", None)
                else:
                    return None, None
            except Exception:
                return None, None

        if message is None:
            return None, None

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return None, None

        reasoning_chunks: list[str] = []
        signatures: list[str] = []
        if isinstance(content, list):
            for part in content:
                if getattr(part, "type", None) == "reasoning":
                    reasoning_text = getattr(part, "reasoning", None)
                    if reasoning_text:
                        reasoning_chunks.append(reasoning_text.strip())
                    # Extract signature if present (for Anthropic and OpenAI models)
                    signature = getattr(part, "signature", None)
                    if signature:
                        signatures.append(signature)

        reasoning_text = None
        if reasoning_chunks:
            reasoning_text = "\n\n".join(chunk for chunk in reasoning_chunks if chunk)

        # Use the first signature if available (Anthropic/OpenAI typically have one per message)
        signature = signatures[0] if signatures else None

        return reasoning_text, signature

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        cot, signature = _extract_reasoning(state)
        metadata = {"uuid": state.metadata["uuid"]}
        if cot:
            metadata["cot"] = cot
        if signature:
            metadata["cot_signature"] = signature

        return Score(
            value=len(answer),
            uuid=state.metadata["uuid"],
            answer=answer,
            metadata=metadata,
        )

    return score
