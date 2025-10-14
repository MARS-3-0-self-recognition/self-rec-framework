import json
from pathlib import Path
from typing import Dict, Any


def project_root() -> Path:
    """Returns the path to the project root."""
    return Path(__file__).parent.parent.parent


def data_dir() -> Path:
    """Returns the path to the data directory."""
    return project_root() / "data"


def load_json(file_path: Path) -> Dict[str, Any]:
    """Read the json file at the given path."""
    with open(file_path, "r") as f:
        return json.load(f)


def rollout_json_file_path(
    dataset_name: str, model_name: str, generation_string: str
) -> Path:
    """
    Returns the path to the rollout json file for a given dataset, model, and generation string.
    """
    return data_dir() / dataset_name / model_name / f"{generation_string}.json"


def load_rollout_json(
    dataset_name: str, model_name: str, generation_string: str
) -> Dict[str, Any]:
    """Read the rollout json file for a given dataset, model, and generation string."""
    return load_json(
        rollout_json_file_path(dataset_name, model_name, generation_string)
    )
