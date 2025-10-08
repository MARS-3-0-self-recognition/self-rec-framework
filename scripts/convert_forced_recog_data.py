#!/usr/bin/env python3
"""
Convert forced_recog CSV data to pairwise JSON format

This utility converts data from the forced_recog pipeline (CSV format with control/treatment)
to the new protocols/pairwise format (JSON with model-specific outputs).

Usage:
    # Convert a single experiment directory
    python scripts/convert_forced_recog_data.py \
        --input forced_recog/results_and_data/experiments/WikiSum/model_vs_treatment \
        --output data/wikisum \
        --model-name claude-3-5-sonnet \
        --alternative-model-name gpt-4 \
        --control-generation control \
        --treatment-generation typo

    # Convert with custom field names
    python scripts/convert_forced_recog_data.py \
        --input path/to/experiment \
        --output data/my_dataset \
        --model-name my_model \
        --alternative-model-name other_model \
        --passage-field passage \
        --response-field response
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict
import sys


def load_csv_data(
    csv_path: Path, passage_field: str = "passage", response_field: str = "response"
) -> pd.DataFrame:
    """Load CSV file and validate required columns exist."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check for required columns
    required_cols = [passage_field, response_field]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"CSV missing required columns: {missing_cols}. Found: {df.columns.tolist()}"
        )

    return df


def create_uuid_mapping(
    df: pd.DataFrame, passage_field: str = "passage"
) -> Dict[str, str]:
    """
    Create UUID -> content mapping.
    If CSV has a 'uuid' column, use it. Otherwise, create UUIDs from row index.
    """
    if "uuid" in df.columns:
        # Use existing UUIDs
        uuid_map = {row["uuid"]: row[passage_field] for _, row in df.iterrows()}
    else:
        # Create UUIDs from index
        uuid_map = {f"item_{idx:04d}": row[passage_field] for idx, row in df.iterrows()}

    return uuid_map


def extract_outputs(
    df: pd.DataFrame,
    uuid_mapping: Dict[str, str],
    passage_field: str = "passage",
    response_field: str = "response",
) -> Dict[str, str]:
    """
    Extract outputs (responses) matched to UUIDs.

    Returns dict mapping UUID -> response text
    """
    # If we have explicit UUIDs
    if "uuid" in df.columns:
        outputs = {row["uuid"]: row[response_field] for _, row in df.iterrows()}
    else:
        # Match by passage content
        reverse_uuid_map = {content: uuid for uuid, content in uuid_mapping.items()}
        outputs = {}
        for _, row in df.iterrows():
            passage = row[passage_field]
            if passage in reverse_uuid_map:
                uuid = reverse_uuid_map[passage]
                outputs[uuid] = row[response_field]

    return outputs


def convert_experiment(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    alternative_model_name: str,
    control_generation: str,
    treatment_generation: str,
    passage_field: str = "passage",
    response_field: str = "response",
    content_filename: str = "articles.json",
    output_filename: str = "summaries.json",
) -> None:
    """
    Convert a forced_recog experiment directory to pairwise JSON format.

    Args:
        input_dir: Directory containing control.csv and treatment.csv
        output_dir: Output directory for JSON files
        model_name: Name of the model (for directory structure)
        alternative_model_name: Name of alternative model
        control_generation: Generation string for control (e.g., 'control')
        treatment_generation: Generation string for treatment (e.g., 'typo')
        passage_field: Name of the passage/content column in CSV
        response_field: Name of the response/output column in CSV
        content_filename: Name for the content file (default: articles.json)
        output_filename: Base name for output files (default: summaries.json)
    """
    print(f"\n{'='*80}")
    print(f"Converting: {input_dir.name}")
    print(f"{'='*80}\n")

    # Load CSV files
    control_path = input_dir / "control.csv"
    treatment_path = input_dir / "treatment.csv"

    print(f"Loading {control_path}...")
    control_df = load_csv_data(control_path, passage_field, response_field)
    print(f"  Found {len(control_df)} rows")

    print(f"Loading {treatment_path}...")
    treatment_df = load_csv_data(treatment_path, passage_field, response_field)
    print(f"  Found {len(treatment_df)} rows")

    # Create UUID mapping from control data
    print("\nCreating UUID mapping...")
    uuid_mapping = create_uuid_mapping(control_df, passage_field)
    print(f"  Created {len(uuid_mapping)} UUIDs")

    # Extract outputs
    print("\nExtracting outputs...")
    control_outputs = extract_outputs(
        control_df, uuid_mapping, passage_field, response_field
    )
    treatment_outputs = extract_outputs(
        treatment_df, uuid_mapping, passage_field, response_field
    )
    print(f"  Control outputs: {len(control_outputs)}")
    print(f"  Treatment outputs: {len(treatment_outputs)}")

    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    alt_model_dir = output_dir / alternative_model_name
    alt_model_dir.mkdir(parents=True, exist_ok=True)

    # Save content file (articles/questions)
    content_path = output_dir / content_filename
    print(f"\nSaving content to {content_path}...")
    with open(content_path, "w", encoding="utf-8") as f:
        json.dump(uuid_mapping, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {len(uuid_mapping)} items")

    # Save control outputs (model outputs)
    control_output_path = model_dir / f"{control_generation}_{output_filename}"
    print(f"\nSaving control outputs to {control_output_path}...")
    with open(control_output_path, "w", encoding="utf-8") as f:
        json.dump(control_outputs, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {len(control_outputs)} outputs")

    # Save treatment outputs (alternative model outputs)
    treatment_output_path = alt_model_dir / f"{treatment_generation}_{output_filename}"
    print(f"\nSaving treatment outputs to {treatment_output_path}...")
    with open(treatment_output_path, "w", encoding="utf-8") as f:
        json.dump(treatment_outputs, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {len(treatment_outputs)} outputs")

    print(f"\n{'='*80}")
    print("Conversion complete!")
    print(f"{'='*80}")
    print("\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── {content_filename}")
    print(f"    ├── {model_name}/")
    print(f"    │   └── {control_generation}_{output_filename}")
    print(f"    └── {alternative_model_name}/")
    print(f"        └── {treatment_generation}_{output_filename}")
    print("\nYou can now run experiments with:")
    print(f"  dataset_name: {output_dir.name}")
    print(f"  model_name: {model_name}")
    print(f"  alternative_model_name: {alternative_model_name}")
    print(f"  model_generation_string: {control_generation}")
    print(f"  alternative_model_generation_string: {treatment_generation}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert forced_recog CSV data to pairwise JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python scripts/convert_forced_recog_data.py \\
        --input forced_recog/results_and_data/experiments/WikiSum/claude_vs_typo \\
        --output data/wikisum \\
        --model-name claude-3-5-sonnet \\
        --alternative-model-name claude-3-5-sonnet \\
        --control-generation control \\
        --treatment-generation typo

    # Custom field names
    python scripts/convert_forced_recog_data.py \\
        --input path/to/experiment \\
        --output data/my_dataset \\
        --model-name my_model \\
        --alternative-model-name other_model \\
        --passage-field article_text \\
        --response-field generated_summary
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing control.csv and treatment.csv",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for JSON files"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model (creates subdirectory)",
    )

    parser.add_argument(
        "--alternative-model-name",
        type=str,
        required=True,
        help="Name of the alternative model (creates subdirectory)",
    )

    parser.add_argument(
        "--control-generation",
        type=str,
        default="control",
        help="Generation string for control outputs (default: control)",
    )

    parser.add_argument(
        "--treatment-generation",
        type=str,
        default="treatment",
        help="Generation string for treatment outputs (default: treatment)",
    )

    parser.add_argument(
        "--passage-field",
        type=str,
        default="passage",
        help="Name of passage/content column in CSV (default: passage)",
    )

    parser.add_argument(
        "--response-field",
        type=str,
        default="response",
        help="Name of response/output column in CSV (default: response)",
    )

    parser.add_argument(
        "--content-filename",
        type=str,
        default="articles.json",
        help="Filename for content file (default: articles.json)",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="summaries.json",
        help="Base filename for output files (default: summaries.json)",
    )

    args = parser.parse_args()

    # Convert paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Run conversion
    try:
        convert_experiment(
            input_dir=input_dir,
            output_dir=output_dir,
            model_name=args.model_name,
            alternative_model_name=args.alternative_model_name,
            control_generation=args.control_generation,
            treatment_generation=args.treatment_generation,
            passage_field=args.passage_field,
            response_field=args.response_field,
            content_filename=args.content_filename,
            output_filename=args.output_filename,
        )
    except Exception as e:
        print(f"\nError during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
