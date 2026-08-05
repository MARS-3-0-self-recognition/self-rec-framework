import argparse
import json
from pathlib import Path
from typing import Dict, Any


def project_root() -> Path:
    """Returns the path to the project root (current working directory)."""
    return Path.cwd()


def package_root() -> Path:
    """Returns the path to the self_rec_framework package root."""
    return Path(__file__).parent.parent.parent


def data_dir() -> Path:
    """Returns the path to the data directory."""
    return project_root() / "data"


def load_json(file_path: Path) -> Dict[str, Any]:
    """Read the json file at the given path."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data: Any, file_path: Path, create_dir: bool = True):
    """Save the data to the given path."""
    if create_dir:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def rollout_json_file_path(
    dataset_name: str, model_name: str, generation_string: str
) -> Path:
    """
    Returns the path to the rollout json file for a given dataset, model, and generation string.
    """
    return data_dir() / dataset_name / model_name / f"{generation_string}.json"


def rollout_eval_log_dir(
    dataset_name: str, model_name: str, generation_string: str
) -> Path:
    """Returns the path to the Inspect eval file for base rollout corresponding to a given dataset, model, and generation string."""
    return data_dir() / dataset_name / model_name / generation_string


def parse_range(range_str: str) -> tuple[int, int]:
    """Parse a range string like '5-15' into a tuple of (start, end)."""
    try:
        parts = range_str.split("-")
        if len(parts) != 2:
            raise ValueError
        start, end = int(parts[0]), int(parts[1])
        return (start, end)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"Range must be in format 'START-END' (e.g., '5-15'), got: {range_str}"
        )


def load_rollout_json(
    dataset_name: str, model_name: str, generation_string: str
) -> Dict[str, Any]:
    """Read the rollout json file for a given dataset, model, and generation string."""
    return load_json(
        rollout_json_file_path(dataset_name, model_name, generation_string)
    )
