#!/usr/bin/env python3
"""
Rescore an eval file using the logprob_scorer.

This script reads an eval log file, applies the logprob_scorer to each sample,
and updates the scores. The scorer checks that the final character of the
completion is "1" or "2", marking as failure if not.

Usage:
    uv run experiments/scripts/rescoring/rescore_eval_file.py \\
        --eval_file data/results/.../experiment.eval \\
        [--output_file path/to/output.eval]  # Optional: defaults to overwriting input
"""

import argparse
import asyncio
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log, write_eval_log_async
from inspect_ai.scorer import Target
from inspect_ai.scorer._metric import SampleScore
from inspect_ai._eval.task.results import eval_results
from inspect_ai._eval.score import (
    metrics_from_log_header,
    reducers_from_log_header,
)
from inspect_ai._eval.loader import load_module

from src.inspect.scorer import logprob_scorer


class MockOutput:
    """Minimal mock of state.output for rescoring."""

    def __init__(self, completion: str):
        self.completion = completion


class MockTaskState:
    """Minimal mock of TaskState for rescoring."""

    def __init__(self, completion: str, metadata: dict[str, Any]):
        self.output = MockOutput(completion)
        self.metadata = metadata


async def rescore_eval_file(eval_file: Path, output_file: Path | None = None) -> None:
    """
    Rescore an eval file using logprob_scorer.

    Args:
        eval_file: Path to the input eval file
        output_file: Path to write the rescored eval file. If None, overwrites input.
    """
    # Read the eval log using inspect_ai's read_eval_log (handles ZIP format)
    print(f"Reading eval file: {eval_file}")
    log = read_eval_log(eval_file)

    if not log.samples:
        print("⚠ No samples found in eval file. Nothing to rescore.")
        return

    print(f"Found {len(log.samples)} samples to rescore")

    # Get the scorer function
    scorer_func = logprob_scorer()

    # Rescore each sample
    updated_count = 0
    failed_count = 0

    for i, sample in enumerate(log.samples):
        # Extract completion and metadata from sample
        completion = sample.output.completion if sample.output else ""
        metadata = sample.metadata if sample.metadata else {}

        # Create a minimal TaskState for the scorer
        state = MockTaskState(completion, metadata)

        # Create a minimal Target (scorer may not use it, but required by signature)
        target = Target(metadata.get("correct_answer", ""))

        try:
            # Run the scorer (it's async)
            new_score = await scorer_func(state, target)

            # Normalize the displayed answer to match inspect's expected format
            # (single-character "1"/"2" or "F" for failures), so the viewer renders correctly.
            completion_stripped = completion.strip()
            if completion_stripped and completion_stripped[-1] in {"1", "2"}:
                display_answer = completion_stripped[-1]
            else:
                display_answer = "F"
            new_score.answer = display_answer
            # Normalize failure marker to Inspect's expected "N" (no answer)
            # so the viewer uses pass/fail formatting correctly.
            if isinstance(new_score.value, dict) and new_score.value.get("acc") == "F":
                new_score.value["acc"] = "N"
                new_score.metadata = {**(new_score.metadata or {}), "failure": True}

            # Update the sample's scores
            if not sample.scores:
                sample.scores = {}
            sample.scores["logprob_scorer"] = new_score

            updated_count += 1
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(log.samples)} samples...")

        except Exception as e:
            failed_count += 1
            print(f"  ⚠ Error rescoring sample {i}: {e}")

    print("\n✓ Rescoring complete!")
    print(f"  Updated: {updated_count} samples")
    if failed_count > 0:
        print(f"  Failed: {failed_count} samples")

    # Recompute aggregated metrics, results, and reductions based on new scores
    print("\nRecomputing aggregated metrics and results...")
    try:
        # Load the task file to ensure scorers can be resolved
        if log.eval and log.eval.task_file:
            task_file = Path(log.eval.task_file)
            if task_file.exists():
                load_module(task_file)

        # Extract scores from all samples
        scores = []
        for sample in log.samples:
            if sample.scores:
                sample_scores = {}
                for score_name, score in sample.scores.items():
                    sample_scores[score_name] = SampleScore(
                        score=score,
                        sample_id=sample.id,
                        sample_metadata=sample.metadata,
                    )
                scores.append(sample_scores)

        reducers = reducers_from_log_header(log)
        metrics = metrics_from_log_header(log)

        # Get scorer names from log
        scorer_names = (
            [s.name for s in log.eval.scorers]
            if log.eval and log.eval.scorers
            else None
        )

        # Pre-populate scorers_info by creating ScorerInfo objects for each scorer name
        # This is necessary because eval_results only creates ScorerInfo for scores not in
        # resolved_scorer_names, but when all scores match scorer_names, scorers_info stays empty
        if scorer_names:
            # Create ScorerInfo objects that will be used by eval_results
            # We pass scorer_names but also need to ensure scorers_info gets populated
            # The workaround: Don't pass scorer_names, let eval_results auto-detect from scores
            # But ensure the scorer can be loaded by loading the task file above

            # Actually, the issue is that when scorer_names is provided, the code expects
            # scorers to be provided OR for scores to have names not in scorer_names.
            # Since our scores have names IN scorer_names, we need to pass scorer_names=None
            # to let it auto-detect, OR we need to ensure the scorer can be resolved.
            # Let's use the auto-detect approach by not passing scorer_names:
            scorer_names_for_eval = None
        else:
            scorer_names_for_eval = None

        # Recompute results and reductions
        # By not passing scorer_names, eval_results will auto-detect from scores
        # and create ScorerInfo.from_name for each unique score name
        results, reductions = eval_results(
            samples=len(log.samples),
            scores=scores,
            reducers=reducers,
            scorers=None,  # Don't pass scorers to avoid registry info issues
            metrics=metrics,
            scorer_names=scorer_names_for_eval,  # None = auto-detect from scores
            early_stopping=log.results.early_stopping if log.results else None,
        )

        # Update the log's results and reductions
        log.results = results
        log.reductions = reductions

        print("  ✓ Metrics recomputed successfully")
    except Exception as e:
        print(f"  ⚠ Warning: Could not recompute metrics: {e}")
        print("  Continuing with sample-level scores only...")
        import traceback

        traceback.print_exc()

    # Always ensure status is set to "success" after rescoring (if it wasn't already)
    # This is required for the eval viewer to display correctly with proper formatting
    if log.status != "success":
        log.status = "success"
        log.error = None
        log.invalidated = False

    # Determine output path
    if output_file is None:
        output_file = eval_file
        print(f"\nWriting to: {output_file} (overwriting input)")
    else:
        print(f"\nWriting to: {output_file}")

    # Use inspect_ai's native write function to properly write the entire EvalLog
    # This ensures all aggregated results, reductions, summaries, etc. are updated
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Set the location on the log so write_eval_log_async knows where to write
    log.location = str(output_file)

    # Write using the native async function (handles all ZIP structure, summaries, etc.)
    await write_eval_log_async(log, str(output_file), format="eval")

    print(f"✓ Successfully wrote rescored eval file to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Rescore an eval file using logprob_scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to the input eval file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to write the rescored eval file (default: overwrites input)",
    )

    args = parser.parse_args()

    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        print(f"Error: Eval file not found: {eval_file}")
        return

    output_file = Path(args.output_file) if args.output_file else None

    # Run the async rescoring function
    asyncio.run(rescore_eval_file(eval_file, output_file))


if __name__ == "__main__":
    main()
