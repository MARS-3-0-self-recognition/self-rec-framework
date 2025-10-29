"""
Unified treatment application interface for caps and typos.

This module provides a high-level API to apply typo or capitalization treatments
to data files without using the command-line interface.
"""

import argparse
import random
from pathlib import Path

from src.data_gen.procedural_editing.caps import apply_caps_treatment
from src.data_gen.procedural_editing.typos import apply_typo_treatment
from src.helpers.utils import load_json, save_json


def apply_treatment(
    treatment_type: str,
    strength: str,
    input_path: str,
    output_path: str = None,
    seed: int = None,
) -> Path:
    """
    Apply typo or capitalization treatment to a data file.

    Args:
        treatment_type (str): Type of treatment to apply. Must be "caps" or "typos".
        strength (str): Treatment strength. Must be "S1", "S2", "S3", or "S4".
        input_path (str): Path to input JSON file (UUID->text format).
        output_path (str, optional): Path to output JSON file. If not provided,
                                     will be automatically generated based on input path.
        seed (int, optional): Random seed for reproducible results.

    Returns:
        Path: Path to the output file.

    Raises:
        ValueError: If treatment_type is not "caps" or "typos", or if strength is invalid.
        FileNotFoundError: If input file does not exist.

    Example:
        uv run src/data_gen/procedural_editing/treatment.py --treatment_type=typos --strength=S2 --input_path=data/wikisum_debug/haiku-3-5/wikisum_config.json
    """
    # Validate treatment type
    if treatment_type not in ["caps", "typos"]:
        raise ValueError(
            f"Invalid treatment type '{treatment_type}'. Must be 'caps' or 'typos'."
        )

    # Validate strength
    valid_strengths = ["S1", "S2", "S3", "S4"]
    if strength not in valid_strengths:
        raise ValueError(
            f"Invalid strength '{strength}'. Must be one of: {valid_strengths}"
        )

    # Load input data
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Determine output path
    if output_path is None:
        # Generate output path: data/wikisum_train_1-20/haiku-3-5/wikisum_config.json
        #                        -> data/wikisum_train_1-20/haiku-3-5_caps_S2/wikisum_config.json
        # Get the parent directory name (e.g., "haiku-3-5")
        parent_dir_name = input_file.parent.name

        # Create new directory name with treatment suffix (e.g., "haiku-3-5_caps_S2")
        new_dir_name = f"{parent_dir_name}_{treatment_type}_{strength}"

        # Create new path
        output_dir = input_file.parent.parent / new_dir_name
        output_file = output_dir / input_file.name
    else:
        output_file = Path(output_path)

    # Set random seed if provided (must be before applying treatment)
    if seed is not None:
        random.seed(seed)

    # Load input data
    input_data = load_json(input_file)

    # Apply treatment
    if treatment_type == "caps":
        treated_data = apply_caps_treatment(input_data, strength=strength)
    elif treatment_type == "typos":
        treated_data = apply_typo_treatment(input_data, strength=strength)

    # Save treated data
    save_json(treated_data, output_file)

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Apply typo or capitalization treatments to data files",
        epilog="Examples:\n"
        "  %(prog)s --treatment_type=caps --strength=S2 --input_path=data/wikisum_train_1-20/haiku-3-5/wikisum_config.json\n"
        "  %(prog)s --treatment_type=typos --strength=S1 --input_path=data/cnn_debug/3-5-haiku/simple_config.json --seed=42",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--treatment_type",
        type=str,
        required=True,
        choices=["caps", "typos"],
        help="Type of treatment to apply: 'caps' or 'typos'",
    )
    parser.add_argument(
        "--strength",
        type=str,
        required=True,
        choices=["S1", "S2", "S3", "S4"],
        help="Treatment strength: S1-S4 (see treatment docs for details)",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSON file (UUID->text format)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output JSON file (default: auto-generated from input path)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )

    args = parser.parse_args()

    try:
        output_path = apply_treatment(
            treatment_type=args.treatment_type,
            strength=args.strength,
            input_path=args.input_path,
            output_path=args.output_path,
            seed=args.seed,
        )

        print(
            f"✓ Successfully applied {args.treatment_type} treatment ({args.strength})"
        )
        print(f"  Output saved to: {output_path}")
        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
