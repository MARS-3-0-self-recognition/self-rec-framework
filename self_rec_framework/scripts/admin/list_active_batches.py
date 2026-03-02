#!/usr/bin/env python3
"""
List all active batch jobs and optionally save their IDs for later retrieval.

This helps recover from interrupted batch runs by showing what's still processing.
Supports OpenAI, Anthropic, and Together AI batch jobs.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from together import Together

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


def list_together_batches(save_to_file=False):
    """List and optionally save Together AI batch job info."""
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            print("⚠ TOGETHER_API_KEY not found in environment")
            return "together", []

        client = Together(api_key=api_key)
        batches = client.batches.list_batches()

        # Together AI batch statuses: likely "pending", "processing", "completed", "failed", "cancelled"
        # We'll treat "pending" and "processing" as active
        active_statuses = ["pending", "processing", "in_progress"]
        active_batches = []

        for batch in batches:
            # Check status - Together AI batches might have status as an attribute
            status = getattr(batch, "status", None) or getattr(batch, "processing_status", None)
            if status and status.lower() in [s.lower() for s in active_statuses]:
                batch_info = {
                    "id": getattr(batch, "id", None) or getattr(batch, "batch_job_id", None),
                    "status": status,
                    "created_at": getattr(batch, "created_at", None),
                    "request_counts": {},
                }

                # Try to get request counts if available
                request_counts = getattr(batch, "request_counts", None)
                if request_counts:
                    batch_info["request_counts"] = {
                        "total": getattr(request_counts, "total", None),
                        "completed": getattr(request_counts, "completed", None)
                        or getattr(request_counts, "succeeded", None),
                        "failed": getattr(request_counts, "failed", None)
                        or getattr(request_counts, "errored", None),
                        "pending": getattr(request_counts, "pending", None)
                        or getattr(request_counts, "processing", None),
                    }

                active_batches.append(batch_info)

        return "together", active_batches

    except Exception as e:
        print(f"⚠ Error accessing Together AI: {e}")
        return "together", []


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

    # Check Together AI
    provider, together_batches = list_together_batches()
    all_batches[provider] = together_batches

    print(f"Together AI: {len(together_batches)} active batches")
    for batch in together_batches:
        request_counts = batch.get("request_counts", {})
        total = request_counts.get("total")
        completed = request_counts.get("completed") or request_counts.get("succeeded", 0)
        if total is not None:
            progress = f"{completed}/{total}"
        else:
            progress = "unknown"
        print(f"  • {batch['id']}")
        print(f"    Status: {batch['status']}, Progress: {progress}")

    print()

    total_active = len(openai_batches) + len(anthropic_batches) + len(together_batches)

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
