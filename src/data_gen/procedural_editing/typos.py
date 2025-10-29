#!/usr/bin/env python3
"""
Apply typo treatments to model outputs in JSON format.

Takes a JSON file with UUID->text mappings and applies S1-S4 typo treatments,
saving the results as {treatment_name}_config.json files.

Usage:
    # Apply S1 treatment (0.1 typos per word) to simple_config.json
    uv run src/data_gen/procedural_editing/apply_typo_treatment.py --input_file=data/cnn_debug/3-5-sonnet/simple_config.json --treatment_name=typo_s1 --strength=S1

    # Apply S4 treatment (1.2 typos per word) to all outputs
    uv run src/data_gen/procedural_editing/apply_typo_treatment.py --input_file=data/cnn_debug/3-5-haiku/simple_config.json --treatment_name=typo_s4 --strength=S4

    # Apply S2 treatment with custom output directory
    uv run src/data_gen/procedural_editing/apply_typo_treatment.py --input_file=data/wikisum_debug/input.json --treatment_name=typo_s2 --strength=S2 --output_dir=data/wikisum_debug

Strength levels:
    S1: 0.1 typos per word (light)
    S2: 0.3 typos per word (medium)
    S3: 0.6 typos per word (heavy)
    S4: 1.2 typos per word (major)
"""

import argparse
import random
import string
from pathlib import Path

from src.helpers.utils import load_json, save_json


def get_keyboard_substitution(char):
    """
    Get a realistic keyboard substitution for a given character.
    Based on QWERTY keyboard layout and common typing errors.

    Args:
        char (str): The character to find a substitution for

    Returns:
        str: A realistic substitution character, or the original if no substitution available
    """
    # QWERTY keyboard layout for common substitutions
    keyboard_map = {
        "q": ["w", "a"],
        "w": ["q", "e", "a", "s"],
        "e": ["w", "r", "s", "d"],
        "r": ["e", "t", "d", "f"],
        "t": ["r", "y", "f", "g"],
        "y": ["t", "u", "g", "h"],
        "u": ["y", "i", "h", "j"],
        "i": ["u", "o", "j", "k"],
        "o": ["i", "p", "k", "l"],
        "p": ["o", "l"],
        "a": ["q", "w", "s", "z"],
        "s": ["a", "w", "e", "d", "x", "z"],
        "d": ["s", "e", "r", "f", "c", "x"],
        "f": ["d", "r", "t", "g", "v", "c"],
        "g": ["f", "t", "y", "h", "b", "v"],
        "h": ["g", "y", "u", "j", "n", "b"],
        "j": ["h", "u", "i", "k", "m", "n"],
        "k": ["j", "i", "o", "l", "m"],
        "l": ["k", "o", "p"],
        "z": ["a", "s", "x"],
        "x": ["z", "s", "d", "c"],
        "c": ["x", "d", "f", "v"],
        "v": ["c", "f", "g", "b"],
        "b": ["v", "g", "h", "n"],
        "n": ["b", "h", "j", "m"],
        "m": ["n", "j", "k"],
        # Common number substitutions
        "1": ["2", "q"],
        "2": ["1", "3", "w"],
        "3": ["2", "4", "e"],
        "4": ["3", "5", "r"],
        "5": ["4", "6", "t"],
        "6": ["5", "7", "y"],
        "7": ["6", "8", "u"],
        "8": ["7", "9", "i"],
        "9": ["8", "0", "o"],
        "0": ["9", "p"],
    }

    char_lower = char.lower()
    if char_lower in keyboard_map:
        substitution = random.choice(keyboard_map[char_lower])
        # Preserve the original case
        if char.isupper():
            return substitution.upper()
        else:
            return substitution
    return char


def introduce_typos(
    input_string, flip_rate=5, drop_rate=3, add_rate=2, substitute_rate=4
):
    """
    Introduce common typos into a string with configurable rates.

    The modifications are applied in order: drops, substitutions, flips, additions
    to avoid compounding effects (e.g., dropping 100% eliminates need for further processing).

    Args:
        input_string (str): The input string to modify
        flip_rate (int): Percentage of adjacent letter pairs to flip (0-100)
        drop_rate (int): Percentage of characters to drop (0-100)
        add_rate (int): Percentage of positions to add random characters (0-100)
        substitute_rate (int): Percentage of characters to substitute with keyboard neighbors (0-100)

    Returns:
        str: The modified string with introduced typos

    Raises:
        ValueError: If any rate is not between 0 and 100
    """
    # Validate input parameters
    for rate, name in [
        (flip_rate, "flip_rate"),
        (drop_rate, "drop_rate"),
        (add_rate, "add_rate"),
        (substitute_rate, "substitute_rate"),
    ]:
        if not isinstance(rate, (int, float)) or rate < 0 or rate > 100:
            raise ValueError(f"{name} must be a number between 0 and 100")

    if not input_string:
        return input_string

    # Convert rates to probabilities
    flip_prob = flip_rate / 100.0
    drop_prob = drop_rate / 100.0
    add_prob = add_rate / 100.0
    substitute_prob = substitute_rate / 100.0

    # Step 1: Drop characters (highest priority to avoid compounding)
    result = list(input_string)
    if drop_prob > 0:
        # Process from end to beginning to avoid index issues
        for i in range(len(result) - 1, -1, -1):
            if random.random() < drop_prob:
                result.pop(i)

    # Step 2: Substitute characters
    if substitute_prob > 0:
        for i in range(len(result)):
            if random.random() < substitute_prob:
                char = result[i]
                substitution = get_keyboard_substitution(char)
                if substitution != char:
                    result[i] = substitution

    # Step 3: Flip adjacent characters
    if flip_prob > 0 and len(result) >= 2:
        for i in range(len(result) - 1):
            if random.random() < flip_prob:
                # Swap adjacent characters
                result[i], result[i + 1] = result[i + 1], result[i]

    # Step 4: Add random characters
    if add_prob > 0:
        # Find positions to add characters (avoid adding at the very beginning/end)
        positions_to_add = []
        for i in range(1, len(result)):
            if random.random() < add_prob:
                positions_to_add.append(i)

        # Add characters (process in reverse order to maintain indices)
        for pos in reversed(positions_to_add):
            # Generate a random character (letter or number)
            if random.random() < 0.7:  # 70% letters, 30% numbers
                char = random.choice(string.ascii_letters)
            else:
                char = random.choice(string.digits)
            result.insert(pos, char)

    return "".join(result)


def introduce_typos_per_word(input_string, typos_per_word=1.0, typo_types=None):
    """
    Introduce typos into a string with a specified rate per word.

    Args:
        input_string (str): The input string to modify
        typos_per_word (float): Number of typos per word. Values < 1 indicate probability
                               (e.g., 0.5 = 50% chance of typo per word)
        typo_types (set): Set of typo types to use. Default is all types.
                         Options: {'substitute_rate', 'flip_rate', 'drop_rate', 'add_rate'}

    Returns:
        str: The modified string with introduced typos

    Raises:
        ValueError: If typos_per_word is negative or typo_types is invalid
    """
    if not input_string:
        return input_string

    if typos_per_word < 0:
        raise ValueError("typos_per_word must be non-negative")

    # Default typo types if none specified
    if typo_types is None:
        typo_types = {"substitute_rate", "flip_rate", "drop_rate", "add_rate"}

    # Validate typo types
    valid_types = {"substitute_rate", "flip_rate", "drop_rate", "add_rate"}
    if not typo_types.issubset(valid_types):
        raise ValueError(f"Invalid typo types. Must be subset of {valid_types}")

    if not typo_types:
        return input_string  # No typo types specified, return original

    # Split into words (preserve whitespace)
    import re

    words = re.split(r"(\s+)", input_string)
    result_words = []

    for word in words:
        if not word.strip():  # Whitespace
            result_words.append(word)
            continue

        # Skip very short words (1-2 characters)
        if len(word.strip()) <= 2:
            result_words.append(word)
            continue

        # Calculate probability of introducing each typo type in this word
        # Each typo type has rate/4 chance of occurring per word
        char_list = list(word)

        for typo_type in typo_types:
            if random.random() < typos_per_word / 4:
                if typo_type == "substitute_rate":
                    # Substitute one character with keyboard neighbor
                    # Find valid characters to substitute
                    valid_indices = []
                    for i, char in enumerate(char_list):
                        if char.isalnum():  # Only substitute alphanumeric characters
                            valid_indices.append(i)

                    if valid_indices:
                        idx = random.choice(valid_indices)
                        char = char_list[idx]
                        substitution = get_keyboard_substitution(char)
                        if substitution != char:
                            char_list[idx] = substitution

                elif typo_type == "flip_rate":
                    # Flip one pair of adjacent characters
                    if len(char_list) >= 2:
                        # Find pairs of adjacent non-space characters
                        valid_pairs = []
                        for i in range(len(char_list) - 1):
                            if char_list[i] != " " and char_list[i + 1] != " ":
                                valid_pairs.append(i)

                        if valid_pairs:
                            idx = random.choice(valid_pairs)
                            char_list[idx], char_list[idx + 1] = (
                                char_list[idx + 1],
                                char_list[idx],
                            )

                elif typo_type == "drop_rate":
                    # Drop one character
                    if len(char_list) >= 2:
                        # Find non-space characters
                        non_space_indices = [
                            i for i, char in enumerate(char_list) if char != " "
                        ]
                        if non_space_indices:
                            idx = random.choice(non_space_indices)
                            char_list.pop(idx)

                elif typo_type == "add_rate":
                    # Add one random character
                    if char_list:
                        # Find positions next to non-space characters
                        valid_positions = []
                        for i in range(len(char_list) + 1):
                            # Check if position is adjacent to a non-space character
                            if (i > 0 and char_list[i - 1] != " ") or (
                                i < len(char_list) and char_list[i] != " "
                            ):
                                valid_positions.append(i)

                        if valid_positions:
                            pos = random.choice(valid_positions)
                            # Generate a keyboard-adjacent character based on nearby characters
                            nearby_chars = []

                            # Look at characters around the insertion point
                            if pos > 0 and char_list[pos - 1].isalnum():
                                nearby_chars.append(char_list[pos - 1])
                            if pos < len(char_list) and char_list[pos].isalnum():
                                nearby_chars.append(char_list[pos])

                            # Generate a keyboard-adjacent character
                            if nearby_chars:
                                # Use keyboard substitution logic to find adjacent characters
                                base_char = random.choice(nearby_chars)
                                keyboard_adjacent = get_keyboard_substitution(base_char)
                                if keyboard_adjacent != base_char:
                                    char = keyboard_adjacent
                                else:
                                    # Fallback to random alphanumeric
                                    char = random.choice(
                                        string.ascii_letters + string.digits
                                    )
                            else:
                                # Fallback to random alphanumeric
                                char = random.choice(
                                    string.ascii_letters + string.digits
                                )

                            char_list.insert(pos, char)

        result_words.append("".join(char_list))

    return "".join(result_words)


def apply_typo_treatment(
    input_data,
    strength=None,
    typos_per_word=None,
    flip_rate=None,
    drop_rate=None,
    add_rate=None,
    substitute_rate=None,
):
    """
    Apply typo treatment to input data.

    Args:
        input_data (dict): Dictionary with UUID->text mappings
        strength (str, optional): Treatment strength (S1, S2, S3, S4)
        typos_per_word (float, optional): Custom typos per word rate (>=0)
        flip_rate (float, optional): Percentage of adjacent letter pairs to flip (0-100)
        drop_rate (float, optional): Percentage of characters to drop (0-100)
        add_rate (float, optional): Percentage of positions to add random characters (0-100)
        substitute_rate (float, optional): Percentage of characters to substitute (0-100)

    Returns:
        dict: Dictionary with same UUIDs but treated text
    """
    strength_settings = {
        "S1": 0.1,  # Light: 0.1 typos per word
        "S2": 0.3,  # Medium: 0.3 typos per word
        "S3": 0.6,  # Heavy: 0.6 typos per word
        "S4": 1.2,  # Major: 1.2 typos per word
    }

    if strength is not None:
        if strength not in strength_settings:
            raise ValueError(
                f"Invalid strength '{strength}'. Must be one of: {list(strength_settings.keys())}"
            )
        typos_per_word = strength_settings[strength]
        # Use default rates for strength-based treatments
        flip_rate = 5
        drop_rate = 3
        add_rate = 2
        substitute_rate = 4
    elif typos_per_word is not None:
        if typos_per_word < 0:
            raise ValueError(f"Typos per word must be >= 0, got {typos_per_word}")
        # Check that all custom parameters are provided
        if any(
            param is None for param in [flip_rate, drop_rate, add_rate, substitute_rate]
        ):
            raise ValueError(
                "When using custom typos_per_word, all rate parameters (flip_rate, drop_rate, add_rate, substitute_rate) must be specified"
            )
    else:
        raise ValueError("Either strength or typos_per_word must be specified")

    result = {}
    for uuid, text in input_data.items():
        if strength is not None:
            # Use the per-word approach for strength-based treatments
            result[uuid] = introduce_typos_per_word(str(text), typos_per_word)
        else:
            # Use the direct rate approach for custom parameters
            result[uuid] = introduce_typos(
                str(text), flip_rate, drop_rate, add_rate, substitute_rate
            )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Apply typo treatments to model outputs",
        epilog="Examples:\n"
        "  %(prog)s --input_file=data/cnn_debug/3-5-sonnet/simple_config.json --treatment_name=typo_s1 --strength=S1\n"
        "  %(prog)s --input_file=data/wikisum_debug/input.json --treatment_name=typo_s4 --strength=S4",
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
        help="Name for the treatment (e.g., 'typo_s1', 'typo_s2')",
    )
    # Create mutually exclusive group for strength vs custom parameters
    treatment_group = parser.add_mutually_exclusive_group(required=True)
    treatment_group.add_argument(
        "--strength",
        type=str,
        choices=["S1", "S2", "S3", "S4"],
        help="Treatment strength: S1 (0.1 typos/word), S2 (0.3 typos/word), S3 (0.6 typos/word), S4 (1.2 typos/word)",
    )
    treatment_group.add_argument(
        "--custom",
        action="store_true",
        help="Use custom parameters (requires --typos_per_word, --flip_rate, --drop_rate, --add_rate, --substitute_rate)",
    )

    # Custom parameters - all required when using custom mode
    parser.add_argument(
        "--typos_per_word",
        type=float,
        help="Custom typos per word rate (>=0). Required when using --custom.",
    )
    parser.add_argument(
        "--flip_rate",
        type=float,
        help="Percentage of adjacent letter pairs to flip (0-100). Required when using --custom.",
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        help="Percentage of characters to drop (0-100). Required when using --custom.",
    )
    parser.add_argument(
        "--add_rate",
        type=float,
        help="Percentage of positions to add random characters (0-100). Required when using --custom.",
    )
    parser.add_argument(
        "--substitute_rate",
        type=float,
        help="Percentage of characters to substitute with keyboard neighbors (0-100). Required when using --custom.",
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

    # Validate custom parameters
    if args.custom:
        custom_params = [
            args.typos_per_word,
            args.flip_rate,
            args.drop_rate,
            args.add_rate,
            args.substitute_rate,
        ]
        if any(param is None for param in custom_params):
            parser.error(
                "When using --custom, all custom parameters (--typos_per_word, --flip_rate, --drop_rate, --add_rate, --substitute_rate) must be specified"
            )

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
        print(f"Applying {args.strength} typo treatment...")
        treated_data = apply_typo_treatment(input_data, strength=args.strength)
    else:
        print(
            f"Applying custom typo treatment ({args.typos_per_word} typos per word, {args.flip_rate}% flip, {args.drop_rate}% drop, {args.add_rate}% add, {args.substitute_rate}% substitute)..."
        )
        treated_data = apply_typo_treatment(
            input_data,
            typos_per_word=args.typos_per_word,
            flip_rate=args.flip_rate,
            drop_rate=args.drop_rate,
            add_rate=args.add_rate,
            substitute_rate=args.substitute_rate,
        )

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
        print(
            f"  Treatment: {args.treatment_name} ({args.typos_per_word} typos/word, {args.flip_rate}% flip, {args.drop_rate}% drop, {args.add_rate}% add, {args.substitute_rate}% substitute)"
        )
    print(f"  Samples: {len(treated_data)}")

    return 0


if __name__ == "__main__":
    exit(main())
