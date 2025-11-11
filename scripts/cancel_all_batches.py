#!/usr/bin/env python3
"""
Cancel all active batch jobs across all providers.

This script cancels all in-progress batch jobs from:
- OpenAI
- Anthropic
- Google (if any exist)

Use this to clean up orphaned batch jobs when a script is interrupted.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


def cancel_openai_batches():
    """Cancel all active OpenAI batch jobs."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        batches = client.batches.list(limit=100)

        active_statuses = ["validating", "in_progress", "finalizing"]
        cancelled = []

        for batch in batches.data:
            if batch.status in active_statuses:
                try:
                    client.batches.cancel(batch.id)
                    cancelled.append(
                        (
                            batch.id,
                            batch.status,
                            f"{batch.request_counts.completed}/{batch.request_counts.total}",
                        )
                    )
                except Exception as e:
                    print(f"  ⚠ Failed to cancel {batch.id}: {e}")

        return cancelled

    except Exception as e:
        print(f"⚠ Error accessing OpenAI: {e}")
        return []


def cancel_anthropic_batches():
    """Cancel all active Anthropic batch jobs."""
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        batches = client.messages.batches.list(limit=100)

        cancelled = []

        for batch in batches.data:
            if batch.processing_status == "in_progress":
                try:
                    client.messages.batches.cancel(batch.id)
                    total = (
                        batch.request_counts.processing
                        + batch.request_counts.succeeded
                        + batch.request_counts.errored
                        + batch.request_counts.canceled
                        + batch.request_counts.expired
                    )
                    completed = (
                        batch.request_counts.processing + batch.request_counts.succeeded
                    )
                    cancelled.append(
                        (batch.id, batch.processing_status, f"{completed}/{total}")
                    )
                except Exception as e:
                    print(f"  ⚠ Failed to cancel {batch.id}: {e}")

        return cancelled

    except Exception as e:
        print(f"⚠ Error accessing Anthropic: {e}")
        return []


def main():
    print(f"\n{'='*70}")
    print("CANCEL ALL BATCH JOBS")
    print(f"{'='*70}\n")

    print("Checking for active batch jobs...\n")

    # Cancel OpenAI batches
    print("OpenAI:")
    openai_cancelled = cancel_openai_batches()
    if openai_cancelled:
        print(f"  ✓ Cancelled {len(openai_cancelled)} batches:")
        for batch_id, status, progress in openai_cancelled:
            print(f"    • {batch_id} [{status}, {progress} completed]")
    else:
        print("  ⊘ No active batches found")

    print()

    # Cancel Anthropic batches
    print("Anthropic:")
    anthropic_cancelled = cancel_anthropic_batches()
    if anthropic_cancelled:
        print(f"  ✓ Cancelled {len(anthropic_cancelled)} batches:")
        for batch_id, status, progress in anthropic_cancelled:
            print(f"    • {batch_id} [{status}, {progress} completed]")
    else:
        print("  ⊘ No active batches found")

    print()

    # Summary
    total_cancelled = len(openai_cancelled) + len(anthropic_cancelled)

    print(f"{'='*70}")
    print(f"SUMMARY: Cancelled {total_cancelled} total batch jobs")
    print(f"  • OpenAI: {len(openai_cancelled)}")
    print(f"  • Anthropic: {len(anthropic_cancelled)}")
    print(f"{'='*70}\n")

    if total_cancelled > 0:
        print("⚠ Note: You may still be charged for partial processing")
        print("  of cancelled batches. Check provider billing for details.\n")


if __name__ == "__main__":
    main()
