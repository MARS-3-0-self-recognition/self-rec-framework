#!/usr/bin/env python3
"""
Apply capitalization treatments to model outputs in JSON format.

Takes a JSON file with UUID->text mappings and applies S1-S4 capitalization treatments,
saving the results as {treatment_name}_config.json files.

Usage:
    # Apply S1 treatment (25% caps) to simple_config.json
    uv run data_loader/caps.py --input_file=data/cnn_debug/3-5-sonnet/simple_config.json --treatment_name=caps_s1 --strength=S1

    # Apply S4 treatment (100% caps) to all outputs
    uv run data_loader/caps.py --input_file=data/cnn_debug/3-5-haiku/simple_config.json --treatment_name=caps_s4 --strength=S4

    # Apply S2 treatment with custom output directory
    uv run data_loader/caps.py --input_file=data/wikisum_debug/input.json --treatment_name=caps_s2 --strength=S2 --output_dir=data/wikisum_debug

Strength levels:
    S1: 25% capitalization (light)
    S2: 50% capitalization (medium)
    S3: 75% capitalization (heavy)
    S4: 100% capitalization (all caps)
"""

import argparse
import random
from pathlib import Path

from src.helpers.utils import load_json, save_json


def randomly_capitalize_string(input_string, percentage=50):
    """
    Randomly capitalize letters in a string based on the specified percentage.

    Args:
        input_string (str): The input string to modify
        percentage (int): Percentage of letters to capitalize (0-100)

    Returns:
        str: The modified string with randomly capitalized letters

    Raises:
        ValueError: If percentage is not between 0 and 100
    """
    if not isinstance(percentage, (int, float)) or percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be a number between 0 and 100")

    if not input_string:
        return input_string

    # Convert percentage to decimal (e.g., 50% -> 0.5)
    probability = percentage / 100.0

    # Create a list of characters to modify
    result = list(input_string)

    # Iterate through each character and randomly capitalize based on probability
    for i in range(len(result)):
        char = result[i]
        # Only process alphabetic characters
        if char.isalpha():
            # Randomly decide whether to capitalize this letter
            if random.random() < probability:
                result[i] = char.upper()
            else:
                result[i] = char.lower()

    return "".join(result)


def apply_caps_treatment(input_data, strength=None, percentage=None):
    """
    Apply capitalization treatment to input data.

    Args:
        input_data (dict): Dictionary with UUID->text mappings
        strength (str, optional): Treatment strength (S1, S2, S3, S4)
        percentage (float, optional): Custom capitalization percentage (0-100)

    Returns:
        dict: Dictionary with same UUIDs but treated text
    """
    strength_settings = {
        "S1": 25,  # Light: 25% capitalization
        "S2": 50,  # Medium: 50% capitalization
        "S3": 75,  # Heavy: 75% capitalization
        "S4": 100,  # Major: 100% capitalization (all caps)
    }

    if strength is not None:
        if strength not in strength_settings:
            raise ValueError(
                f"Invalid strength '{strength}'. Must be one of: {list(strength_settings.keys())}"
            )
        percentage = strength_settings[strength]
    elif percentage is not None:
        if not (0 <= percentage <= 100):
            raise ValueError(f"Percentage must be between 0 and 100, got {percentage}")
    else:
        raise ValueError("Either strength or percentage must be specified")

    result = {}
    for uuid, text in input_data.items():
        if percentage == 100:
            # 100% is special case: all caps
            result[uuid] = str(text).upper()
        else:
            result[uuid] = randomly_capitalize_string(str(text), percentage)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Apply capitalization treatments to model outputs",
        epilog="Examples:\n"
        "  %(prog)s --input_file=data/cnn_debug/3-5-sonnet/simple_config.json --treatment_name=caps_s1 --strength=S1\n"
        "  %(prog)s --input_file=data/wikisum_debug/input.json --treatment_name=caps_s4 --strength=S4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSON file (UUID->text format)",
    )
    parser.add_argument(
        "--treatment_name",
        type=str,
        required=True,
        help="Name for the treatment (e.g., 'caps_s1', 'caps_s2')",
    )
    # Create mutually exclusive group for strength vs custom parameters
    treatment_group = parser.add_mutually_exclusive_group(required=True)
    treatment_group.add_argument(
        "--strength",
        type=str,
        choices=["S1", "S2", "S3", "S4"],
        help="Treatment strength: S1 (25%% caps), S2 (50%% caps), S3 (75%% caps), S4 (100%% caps)",
    )
    treatment_group.add_argument(
        "--percentage",
        type=float,
        help="Custom capitalization percentage (0-100). Requires --strength to be omitted.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input file directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Load input data
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    print(f"Loading input data from: {input_path}")
    input_data = load_json(input_path)
    print(f"  Loaded {len(input_data)} samples")

    # Apply treatment
    if args.strength:
        print(f"Applying {args.strength} capitalization treatment...")
        treated_data = apply_caps_treatment(input_data, strength=args.strength)
    else:
        print(f"Applying custom capitalization treatment ({args.percentage}%)...")
        treated_data = apply_caps_treatment(input_data, percentage=args.percentage)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save treated data
    output_filename = f"{args.treatment_name}_config.json"
    output_path = output_dir / output_filename

    save_json(treated_data, output_path)

    print(f"✓ Successfully saved treated data to: {output_path}")
    if args.strength:
        print(f"  Treatment: {args.treatment_name} ({args.strength})")
    else:
        print(f"  Treatment: {args.treatment_name} ({args.percentage}%)")
    print(f"  Samples: {len(treated_data)}")

    return 0


if __name__ == "__main__":
    exit(main())
