#!/usr/bin/env python3
"""
List all active batch jobs and optionally save their IDs for later retrieval.

This helps recover from interrupted batch runs by showing what's still processing.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")


def list_openai_batches(save_to_file=False):
    """List and optionally save OpenAI batch job info."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        batches = client.batches.list(limit=100)

        active_statuses = ["validating", "in_progress", "finalizing"]
        active_batches = []

        for batch in batches.data:
            if batch.status in active_statuses:
                batch_info = {
                    "id": batch.id,
                    "status": batch.status,
                    "created_at": batch.created_at,
                    "request_counts": {
                        "total": batch.request_counts.total,
                        "completed": batch.request_counts.completed,
                        "failed": batch.request_counts.failed,
                    },
                }
                active_batches.append(batch_info)

        return "openai", active_batches

    except Exception as e:
        print(f"⚠ Error accessing OpenAI: {e}")
        return "openai", []


def list_anthropic_batches(save_to_file=False):
    """List and optionally save Anthropic batch job info."""
    try:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        batches = client.messages.batches.list(limit=100)

        active_batches = []

        for batch in batches.data:
            if batch.processing_status == "in_progress":
                batch_info = {
                    "id": batch.id,
                    "status": batch.processing_status,
                    "created_at": batch.created_at,
                    "request_counts": {
                        "processing": batch.request_counts.processing,
                        "succeeded": batch.request_counts.succeeded,
                        "errored": batch.request_counts.errored,
                        "canceled": batch.request_counts.canceled,
                        "expired": batch.request_counts.expired,
                    },
                }
                active_batches.append(batch_info)

        return "anthropic", active_batches

    except Exception as e:
        print(f"⚠ Error accessing Anthropic: {e}")
        return "anthropic", []


def main():
    parser = argparse.ArgumentParser(
        description="List active batch jobs across all providers"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save batch IDs to JSON file for later retrieval",
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("ACTIVE BATCH JOBS")
    print(f"{'='*70}\n")

    all_batches = {}

    # Check OpenAI
    provider, openai_batches = list_openai_batches()
    all_batches[provider] = openai_batches

    print(f"OpenAI: {len(openai_batches)} active batches")
    for batch in openai_batches:
        progress = (
            f"{batch['request_counts']['completed']}/{batch['request_counts']['total']}"
        )
        print(f"  • {batch['id']}")
        print(f"    Status: {batch['status']}, Progress: {progress}")

    print()

    # Check Anthropic
    provider, anthropic_batches = list_anthropic_batches()
    all_batches[provider] = anthropic_batches

    print(f"Anthropic: {len(anthropic_batches)} active batches")
    for batch in anthropic_batches:
        total = sum(batch["request_counts"].values())
        completed = (
            batch["request_counts"]["processing"] + batch["request_counts"]["succeeded"]
        )
        progress = f"{completed}/{total}"
        print(f"  • {batch['id']}")
        print(f"    Status: {batch['status']}, Progress: {progress}")

    print()

    total_active = len(openai_batches) + len(anthropic_batches)

    print(f"{'='*70}")
    print(f"TOTAL ACTIVE: {total_active} batch jobs")
    print(f"{'='*70}\n")

    # Save to file if requested
    if args.save:
        save_path = Path(args.save)
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "batches": all_batches,
            "total_active": total_active,
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"✓ Saved batch IDs to: {save_path}\n")
        print("  To retrieve later, keep this file and use the batch IDs")
        print("  to check status or retrieve results manually.\n")


if __name__ == "__main__":
    main()
