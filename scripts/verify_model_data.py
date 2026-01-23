#!/usr/bin/env python3
"""
Verify that models are paired with the correct data (COT-R vs COT-I).

For each model, reports:
- Data model name used for lookup (get_data_model_name)
- Classification: COT-R (own data) vs COT-I (base model data)
- Whether the data path exists for the given dataset

Usage:
    uv run scripts/verify_model_data.py --model_names -set eval_cot-r \\
        --dataset_dir_path data/input/wikisum/training_set_1-20

    uv run scripts/verify_model_data.py --model_names haiku-3.5 gemini-2.5-pro-thinking \\
        --dataset_dir_path data/input/wikisum/training_set_1-20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.helpers.model_names import (  # noqa: E402
    get_data_model_name,
    is_native_reasoning_model,
)
from src.helpers.model_sets import get_model_set  # noqa: E402


def _expand_model_names(raw: list[str]) -> list[str]:
    """Expand -set <name> to model list; pass through individual names."""
    out: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "-set" and i + 1 < len(raw):
            name = raw[i + 1]
            models = get_model_set(name)
            if not models:
                raise ValueError(f"Unknown or empty model set: {name}")
            out.extend(models)
            i += 2
        else:
            out.append(raw[i])
            i += 1
    return out


def parse_dataset_dir_path(path: str) -> tuple[str, str]:
    """Parse dataset_dir_path into (dataset_name, data_subset)."""
    p = Path(path)
    parts = p.parts
    if "input" in parts:
        idx = parts.index("input")
        rest = parts[idx + 1 :]
    else:
        rest = list(parts)
    if len(rest) < 2:
        raise ValueError(
            f"Expected data/input/{{dataset_name}}/{{data_subset}}; got {path}"
        )
    return rest[0], rest[1]


def main() -> None:
    # Preprocess -set for argparse
    if "--model_names" in sys.argv:
        i = sys.argv.index("--model_names")
        for j in range(i + 1, len(sys.argv)):
            if (
                sys.argv[j] == "-set"
                and j + 1 < len(sys.argv)
                and not sys.argv[j + 1].startswith("--")
            ):
                sys.argv[j] = "SET_PLACEHOLDER"

    ap = argparse.ArgumentParser(
        description="Verify models are paired with the correct data (COT-R vs COT-I)."
    )
    ap.add_argument(
        "--model_names",
        nargs="+",
        required=True,
        help="Models to check (supports -set <name>).",
    )
    ap.add_argument(
        "--dataset_dir_path",
        type=str,
        required=True,
        help="Dataset dir, e.g. data/input/wikisum/training_set_1-20",
    )
    args = ap.parse_args()

    args.model_names = [x.replace("SET_PLACEHOLDER", "-set") for x in args.model_names]
    try:
        model_list = _expand_model_names(args.model_names)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if "-set" in args.model_names:
        idx = args.model_names.index("-set")
        set_name = args.model_names[idx + 1] if idx + 1 < len(args.model_names) else "?"
        print(f"Expanded '-set {set_name}' → {len(model_list)} models\n")

    try:
        dataset_name, data_subset = parse_dataset_dir_path(args.dataset_dir_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    data_root = ROOT / "data" / "input" / dataset_name / data_subset

    print("=" * 72)
    print("Model ↔ data verification")
    print("=" * 72)
    print(f"Dataset: {dataset_name} / {data_subset}")
    print(f"Models:  {len(model_list)}")
    print()

    all_ok = True
    for model in model_list:
        data_model = get_data_model_name(model)
        is_cot_r = model.endswith("-thinking") and is_native_reasoning_model(model)
        kind = (
            "COT-R (own data)"
            if is_cot_r
            else ("COT-I (base data)" if model != data_model else "DR (no -thinking)")
        )

        path = data_root / data_model / "data.json"
        exists = path.is_file()

        status = "✓" if exists else "✗"
        if not exists:
            all_ok = False

        print(f"  {status} {model}")
        print(f"      → data model: {data_model}  [{kind}]")
        print(f"      → path: {path}")
        if not exists:
            print("        (missing)")
        print()

    print("=" * 72)
    if all_ok:
        print("All models paired with existing data.")
    else:
        print("Some models have missing data paths.")
        sys.exit(1)


if __name__ == "__main__":
    main()
