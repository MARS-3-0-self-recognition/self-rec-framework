"""Check batch status and download results."""

import argparse
from pathlib import Path
import shutil

from data_gen.utils.api import get_api_client
from data_gen.utils.file_utils import load_json, save_json, read_jsonl
from data_gen.utils.schema import get_response_parser, responses_to_output_dict


def model_to_dirname(model: str) -> str:
    """Convert model string to directory name."""
    return model.replace('/', '--')


def load_request_status(generation_dir: Path) -> dict:
    """Load request_status.json (error if not found)."""
    status_file = generation_dir / "request_status.json"
    
    if not status_file.exists():
        raise FileNotFoundError(
            f"No request_status.json found at {status_file}\n"
            "Make sure you've run the send script first."
        )
    
    return load_json(status_file)


def mark_as_failed(generation_dir: Path, metadata: dict):
    """Move request_status.json to request_status.failed.json with failed=True."""
    status_file = generation_dir / "request_status.json"
    failed_file = generation_dir / "request_status.failed.json"
    
    metadata["failed"] = True
    save_json(metadata, failed_file)
    
    # Remove original
    if status_file.exists():
        status_file.unlink()
    
    print(f"Batch failed. Status saved to {failed_file}")


def check_and_download_batch(api_client, metadata: dict, raw_path: Path, generation_dir: Path) -> bool:
    """
    Check batch status and download if complete.
    
    Returns:
        True if successfully downloaded, False if still processing
    """
    batch_id = metadata["batch_id"]
    
    print(f"Checking batch status: {batch_id}")
    status = api_client.get_status(batch_id)
    
    print(f"Status: {status['status']}")
    
    if status["failed"]:
        print("Batch failed!")
        mark_as_failed(generation_dir, metadata)
        return False
    
    if not status["completed"]:
        print("Batch still processing. Check again later.")
        return False
    
    print("Batch completed! Downloading results...")
    
    # Update metadata with any new fields from status
    for key, value in status.items():
        if key not in metadata and value is not None:
            metadata[key] = value
    
    # Download results
    api_client.download_results(batch_id, raw_path, metadata)
    print(f"Downloaded results to {raw_path}")
    
    return True


def transform_to_output(raw_path: Path, output_path: Path, metadata: dict, api_name: str):
    """Parse raw.jsonl and convert to {uuid: output} format."""
    print("Transforming results to output format...")
    
    # Get parser for this API
    parser = get_response_parser(api_name)
    
    # Read and parse raw results
    raw_results = read_jsonl(raw_path)
    parsed_responses = []
    failed_count = 0
    
    for line in raw_results:
        try:
            custom_id, output_text = parser(line)
            parsed_responses.append((custom_id, output_text))
        except ValueError as e:
            print(f"Warning: {e}")
            failed_count += 1
    
    print(f"Successfully parsed {len(parsed_responses)} responses")
    if failed_count > 0:
        print(f"Failed to parse {failed_count} responses")
    
    # Convert to {uuid: output} format
    custom_id_map = metadata["custom_id_map"]
    output_dict = responses_to_output_dict(parsed_responses, custom_id_map)
    
    # Save output
    save_json(output_dict, output_path)
    print(f"Saved final output to {output_path}")
    
    # Print summary
    expected_count = len(custom_id_map)
    actual_count = len(output_dict)
    print(f"Output contains {actual_count}/{expected_count} items")
    
    if actual_count < expected_count:
        missing = expected_count - actual_count
        print(f"Warning: {missing} items missing from output")


def main():
    parser = argparse.ArgumentParser(description="Receive batch inference results")
    parser.add_argument("--model", required=True, help="Model in format {api}/{model_path}")
    parser.add_argument("--data", required=True, help="Dataset name (directory under data/)")
    parser.add_argument("--generation_config", required=True, help="Path to generation config YAML")
    
    args = parser.parse_args()
    
    # Parse inputs
    model_dirname = model_to_dirname(args.model)
    config_path = Path(args.generation_config)
    generation_name = config_path.stem
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / args.data
    model_dir = data_dir / model_dirname
    generation_dir = model_dir / generation_name
    
    if not generation_dir.exists():
        raise FileNotFoundError(f"Generation directory not found: {generation_dir}")
    
    print(f"Checking batch for {args.model} on {args.data}")
    print(f"Generation name: {generation_name}")
    
    # Load request status
    metadata = load_request_status(generation_dir)
    
    # Check if already received
    if metadata.get("received"):
        print("Batch already received!")
        output_file = model_dir / f"{generation_name}.json"
        if output_file.exists():
            print(f"Output file already exists at {output_file}")
        else:
            print("Warning: received=True but output file not found. Re-transforming...")
            raw_path = generation_dir / "raw.jsonl"
            if raw_path.exists():
                transform_to_output(raw_path, output_file, metadata, metadata["api_name"])
            else:
                print("Error: raw.jsonl not found. Cannot re-transform.")
        return
    
    # Get API client
    api_client = get_api_client(metadata["api_name"])
    
    # Check and download batch
    raw_path = generation_dir / "raw.jsonl"
    success = check_and_download_batch(api_client, metadata, raw_path, generation_dir)
    
    if not success:
        return
    
    # Transform to output format
    output_file = model_dir / f"{generation_name}.json"
    transform_to_output(raw_path, output_file, metadata, metadata["api_name"])
    
    # Update request status
    metadata["received"] = True
    status_file = generation_dir / "request_status.json"
    save_json(metadata, status_file)
    print(f"Updated request status: received=True")
    
    print("\nBatch processing complete!")


if __name__ == "__main__":
    main()