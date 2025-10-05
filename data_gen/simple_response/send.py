"""Send batch inference requests."""

import argparse
import uuid
from pathlib import Path
from datetime import datetime
import yaml

from data_gen.utils.api import get_api_client
from data_gen.utils.file_utils import (
    load_content_file,
    write_jsonl,
    ensure_output_directory,
    save_json
)


def parse_model_string(model: str) -> tuple[str, str]:
    """
    Parse model string into (api_name, model_path).
    
    Args:
        model: Format {api}/{model_path}
        
    Returns:
        (api_name, model_path) tuple
    """
    if '/' not in model:
        raise ValueError(f"Model must be in format {{api}}/{{model_path}}, got {model}")
    
    parts = model.split('/', 1)
    return parts[0], parts[1]


def model_to_dirname(model: str) -> str:
    """
    Convert model string to directory name.
    
    Args:
        model: Format {api}/{model_path}
        
    Returns:
        Directory name with -- instead of /
    """
    return model.replace('/', '--')


def load_config(config_path: Path) -> dict:
    """Load and validate YAML config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required = ["temperature", "output_type", "system_prompt", "prompt"]
    missing = [f for f in required if f not in config]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")
    
    if config["output_type"] not in ["summaries", "answers"]:
        raise ValueError(f"output_type must be 'summaries' or 'answers', got {config['output_type']}")
    
    if "{request}" not in config["prompt"]:
        raise ValueError("prompt must contain {request} placeholder")
    
    return config


def check_existing_request(generation_dir: Path):
    """Throw error if request_status.json already exists."""
    status_file = generation_dir / "request_status.json"
    if status_file.exists():
        raise RuntimeError(
            f"Batch request already exists at {status_file}\n"
            "Delete or move this file to create a new batch request."
        )


def create_request_data(content_data: dict, config: dict, model_path: str, api_client) -> tuple[list, dict]:
    """
    Create request data using API client.
    
    Returns:
        (requests_list, custom_id_map) tuple
    """
    requests = []
    custom_id_map = {}
    
    for content_uuid, content_text in content_data.items():
        # Generate random custom_id
        custom_id = f"req-{uuid.uuid4().hex[:16]}"
        custom_id_map[content_uuid] = custom_id
        
        # Format request using API client
        request = api_client.prepare_request_format(
            custom_id=custom_id,
            content=content_text,
            config=config,
            model=model_path
        )
        requests.append(request)
    
    return requests, custom_id_map


def main():
    parser = argparse.ArgumentParser(description="Send batch inference request")
    parser.add_argument("--model", required=True, help="Model in format {api}/{model_path}")
    parser.add_argument("--data", required=True, help="Dataset name (directory under data/)")
    parser.add_argument("--generation_config", required=True, help="Path to generation config YAML")
    
    args = parser.parse_args()
    
    # Parse inputs
    api_name, model_path = parse_model_string(args.model)
    model_dirname = model_to_dirname(args.model)
    config_path = Path(args.generation_config)
    generation_name = config_path.stem  # e.g., "simple_config" from "simple_config.yaml"
    
    # Load config
    config = load_config(config_path)
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / args.data
    model_dir = ensure_output_directory(data_dir, model_dirname)
    generation_dir = model_dir / generation_name
    generation_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing request
    check_existing_request(generation_dir)
    
    print(f"Creating batch request for {args.model} on {args.data}")
    print(f"Generation name: {generation_name}")
    print(f"Output directory: {generation_dir}")
    
    # Load content
    content_data = load_content_file(data_dir, config["output_type"])
    print(f"Loaded {len(content_data)} items")
    
    # Get API client
    api_client = get_api_client(api_name)
    
    # Create request data
    print("Creating request data...")
    requests, custom_id_map = create_request_data(content_data, config, model_path, api_client)
    
    # Save request data locally
    request_data_path = generation_dir / "request_data.jsonl"
    if api_client.needs_file_upload():
        write_jsonl(requests, request_data_path)
        print(f"Saved request data to {request_data_path}")
    
    # Upload and create batch
    print("Uploading and creating batch...")
    upload_id = None
    if api_client.needs_file_upload():
        upload_id = api_client.upload_requests(request_data_path, model_path)
        print(f"Uploaded file, ID: {upload_id}")
    
    batch_info = api_client.create_batch(
        upload_id=upload_id,
        model=model_path,
        requests=requests if not api_client.needs_file_upload() else None
    )
    print(f"Created batch: {batch_info['batch_id']}")
    print(f"Status: {batch_info['status']}")
    
    # Save request status
    request_status = {
        "api_name": api_name,
        "batch_id": batch_info["batch_id"],
        "model": args.model,
        "model_path": model_path,
        "dataset_name": args.data,
        "generation_name": generation_name,
        "custom_id_map": custom_id_map,
        "received": False,
        "failed": False,
        "created_at": datetime.utcnow().isoformat(),
        "config": config,
        **{k: v for k, v in batch_info.items() if k != "batch_id"}
    }
    
    if upload_id:
        request_status["upload_id"] = upload_id
    
    status_file = generation_dir / "request_status.json"
    save_json(request_status, status_file)
    print(f"Saved request status to {status_file}")
    print("\nBatch request submitted successfully!")
    print(f"Run receive script to check status and download results.")


if __name__ == "__main__":
    main()