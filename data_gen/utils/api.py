"""Batch API client implementations."""

import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class BatchAPIClient(ABC):
    """Abstract base class for batch API implementations."""
    
    @abstractmethod
    def prepare_request_format(self, custom_id: str, content: str, config: dict, model: str) -> dict:
        """
        Format a single request for this API.
        
        Args:
            custom_id: Unique identifier for this request
            content: The article/question text
            config: Config dict with system_prompt, prompt template, temperature
            model: Model identifier
            
        Returns:
            Dict in API-specific format
        """
        pass
    
    @abstractmethod
    def needs_file_upload(self) -> bool:
        """Whether this API requires uploading a file before creating batch."""
        pass
    
    @abstractmethod
    def upload_requests(self, request_data_path: Path, model: str) -> Optional[str]:
        """
        Upload requests file to API (if needed).
        
        Args:
            request_data_path: Path to local request_data.jsonl
            model: Model identifier
            
        Returns:
            file_id/dataset_id or None if upload not needed
        """
        pass
    
    @abstractmethod
    def create_batch(self, upload_id: Optional[str], model: str, requests: Optional[List[dict]] = None) -> Dict[str, Any]:
        """
        Create batch job.
        
        Args:
            upload_id: file_id/dataset_id from upload (or None for APIs without upload)
            model: Model identifier
            requests: Request list for APIs that don't need file upload (Anthropic)
            
        Returns:
            Dict with at minimum: {"batch_id": str, "status": str}
        """
        pass
    
    @abstractmethod
    def get_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get batch status.
        
        Returns:
            Dict with at minimum: {"status": str, "completed": bool}
        """
        pass
    
    @abstractmethod
    def download_results(self, batch_id: str, output_path: Path, metadata: dict):
        """
        Download batch results to output_path.
        
        Args:
            batch_id: Batch identifier
            output_path: Where to save results
            metadata: Full request_status dict (may contain output_file_id, etc)
        """
        pass


class OpenAIBatchClient(BatchAPIClient):
    """OpenAI Batch API implementation."""
    
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def prepare_request_format(self, custom_id: str, content: str, config: dict, model: str) -> dict:
        user_prompt = config["prompt"].format(request=content)
        
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": config.get("temperature", 1.0),
                "max_tokens": config.get("max_tokens", 4096)
            }
        }
    
    def needs_file_upload(self) -> bool:
        return True
    
    def upload_requests(self, request_data_path: Path, model: str) -> str:
        with open(request_data_path, 'rb') as f:
            file_obj = self.client.files.create(file=f, purpose="batch")
        return file_obj.id
    
    def create_batch(self, upload_id: str, model: str, requests: Optional[List[dict]] = None) -> Dict[str, Any]:
        batch = self.client.batches.create(
            input_file_id=upload_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return {
            "batch_id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at
        }
    
    def get_status(self, batch_id: str) -> Dict[str, Any]:
        batch = self.client.batches.retrieve(batch_id)
        completed = batch.status in ["completed", "failed", "expired", "cancelled"]
        failed = batch.status in ["failed", "expired", "cancelled"]
        
        return {
            "status": batch.status,
            "completed": completed,
            "failed": failed,
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None)
        }
    
    def download_results(self, batch_id: str, output_path: Path, metadata: dict):
        output_file_id = metadata.get("output_file_id")
        if not output_file_id:
            # Refresh to get output_file_id
            status = self.get_status(batch_id)
            output_file_id = status.get("output_file_id")
        
        if not output_file_id:
            raise ValueError(f"No output_file_id found for batch {batch_id}")
        
        file_response = self.client.files.content(output_file_id)
        with open(output_path, 'wb') as f:
            f.write(file_response.read())


class AnthropicBatchClient(BatchAPIClient):
    """Anthropic Message Batches API implementation."""
    
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def prepare_request_format(self, custom_id: str, content: str, config: dict, model: str) -> dict:
        user_prompt = config["prompt"].format(request=content)
        
        return {
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": config.get("max_tokens", 4096),
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "system": config["system_prompt"],
                "temperature": config.get("temperature", 1.0)
            }
        }
    
    def needs_file_upload(self) -> bool:
        return False
    
    def upload_requests(self, request_data_path: Path, model: str) -> None:
        return None
    
    def create_batch(self, upload_id: None, model: str, requests: List[dict]) -> Dict[str, Any]:
        if not requests:
            raise ValueError("Anthropic API requires requests list")
        
        batch = self.client.messages.batches.create(requests=requests)
        return {
            "batch_id": batch.id,
            "status": batch.processing_status,
            "created_at": batch.created_at.isoformat() if batch.created_at else None
        }
    
    def get_status(self, batch_id: str) -> Dict[str, Any]:
        batch = self.client.messages.batches.retrieve(batch_id)
        completed = batch.processing_status == "ended"
        failed = False  # Anthropic doesn't have a failed status, individual requests can fail
        
        return {
            "status": batch.processing_status,
            "completed": completed,
            "failed": failed,
            "results_url": batch.results_url if completed else None
        }
    
    def download_results(self, batch_id: str, output_path: Path, metadata: dict):
        # Stream results to file
        with open(output_path, 'w') as f:
            for result in self.client.messages.batches.results(batch_id):
                f.write(json.dumps(result.model_dump()) + '\n')


class FireworksBatchClient(BatchAPIClient):
    """Fireworks Batch Inference API implementation."""
    
    def __init__(self):
        import requests
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        self.account_id = os.getenv("FIREWORKS_ACCOUNT_ID")
        if not self.account_id:
            raise ValueError("FIREWORKS_ACCOUNT_ID environment variable required")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
    
    def prepare_request_format(self, custom_id: str, content: str, config: dict, model: str) -> dict:
        user_prompt = config["prompt"].format(request=content)
        
        return {
            "custom_id": custom_id,
            "body": {
                "messages": [
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": config.get("temperature", 1.0),
                "max_tokens": config.get("max_tokens", 4096)
            }
        }
    
    def needs_file_upload(self) -> bool:
        return True
    
    def upload_requests(self, request_data_path: Path, model: str) -> str:
        dataset_id = f"batch-{int(time.time())}"
        
        # Create dataset
        create_url = f"https://api.fireworks.ai/v1/accounts/{self.account_id}/datasets"
        create_resp = self.session.post(create_url, json={
            "datasetId": dataset_id,
            "dataset": {"userUploaded": {}}
        })
        create_resp.raise_for_status()
        
        # Upload file
        upload_url = f"https://api.fireworks.ai/v1/accounts/{self.account_id}/datasets/{dataset_id}:upload"
        with open(request_data_path, 'rb') as f:
            upload_resp = self.session.post(upload_url, files={"file": f})
        upload_resp.raise_for_status()
        
        return dataset_id
    
    def create_batch(self, upload_id: str, model: str, requests: Optional[List[dict]] = None) -> Dict[str, Any]:
        job_id = f"batch-job-{int(time.time())}"
        output_dataset_id = f"{upload_id}-output"
        
        url = f"https://api.fireworks.ai/v1/accounts/{self.account_id}/batchInferenceJobs?batchInferenceJobId={job_id}"
        resp = self.session.post(url, json={
            "model": model,
            "inputDatasetId": f"accounts/{self.account_id}/datasets/{upload_id}",
            "outputDatasetId": f"accounts/{self.account_id}/datasets/{output_dataset_id}"
        })
        resp.raise_for_status()
        
        return {
            "batch_id": job_id,
            "status": "PENDING",
            "output_dataset_id": output_dataset_id
        }
    
    def get_status(self, batch_id: str) -> Dict[str, Any]:
        url = f"https://api.fireworks.ai/v1/accounts/{self.account_id}/batchInferenceJobs/{batch_id}"
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        
        status = data.get("state", "UNKNOWN")
        # Fireworks uses JOB_STATE_ prefix
        completed = status in ["JOB_STATE_COMPLETED", "COMPLETED", "JOB_STATE_FAILED", "FAILED", "JOB_STATE_EXPIRED", "EXPIRED"]
        failed = status in ["JOB_STATE_FAILED", "FAILED", "JOB_STATE_EXPIRED", "EXPIRED"]
        
        return {
            "status": status,
            "completed": completed,
            "failed": failed,
            "output_dataset_id": data.get("outputDatasetId")
        }
    
    def download_results(self, batch_id: str, output_path: Path, metadata: dict):
        output_dataset_id = metadata.get("output_dataset_id")
        if not output_dataset_id:
            raise ValueError(f"No output_dataset_id found for batch {batch_id}")
        
        # Extract dataset name from full ID
        dataset_name = output_dataset_id.split("/")[-1]
        
        # Get download endpoint
        url = f"https://api.fireworks.ai/v1/accounts/{self.account_id}/datasets/{dataset_name}:getDownloadEndpoint"
        resp = self.session.get(url)
        resp.raise_for_status()
        download_data = resp.json()
        
        # Download first file (assuming single output file)
        filename_to_urls = download_data.get("filenameToSignedUrls", {})
        if not filename_to_urls:
            raise ValueError(f"No download URLs found for dataset {dataset_name}")
        
        # Get first signed URL
        signed_url = next(iter(filename_to_urls.values()))
        
        # Download file
        file_resp = self.session.get(signed_url)
        file_resp.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(file_resp.content)


class TogetherBatchClient(BatchAPIClient):
    """Together Batch API implementation."""
    
    def __init__(self):
        from together import Together
        self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    
    def prepare_request_format(self, custom_id: str, content: str, config: dict, model: str) -> dict:
        user_prompt = config["prompt"].format(request=content)
        
        return {
            "custom_id": custom_id,
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": config.get("temperature", 1.0),
                "max_tokens": config.get("max_tokens", 4096)
            }
        }
    
    def needs_file_upload(self) -> bool:
        return True
    
    def upload_requests(self, request_data_path: Path, model: str) -> str:
        file_resp = self.client.files.upload(file=str(request_data_path), purpose="batch-api")
        return file_resp.id
    
    def create_batch(self, upload_id: str, model: str, requests: Optional[List[dict]] = None) -> Dict[str, Any]:
        batch = self.client.batches.create_batch(upload_id, endpoint="/v1/chat/completions")
        return {
            "batch_id": batch.id,
            "status": batch.status,
            "created_at": getattr(batch, "created_at", None)
        }
    
    def get_status(self, batch_id: str) -> Dict[str, Any]:
        batch = self.client.batches.get_batch(batch_id)
        completed = batch.status in ["COMPLETED", "FAILED", "EXPIRED", "CANCELLED"]
        failed = batch.status in ["FAILED", "EXPIRED", "CANCELLED"]
        
        return {
            "status": batch.status,
            "completed": completed,
            "failed": failed,
            "output_file_id": getattr(batch, "output_file_id", None)
        }
    
    def download_results(self, batch_id: str, output_path: Path, metadata: dict):
        output_file_id = metadata.get("output_file_id")
        if not output_file_id:
            status = self.get_status(batch_id)
            output_file_id = status.get("output_file_id")
        
        if not output_file_id:
            raise ValueError(f"No output_file_id found for batch {batch_id}")
        
        self.client.files.retrieve_content(id=output_file_id, output=str(output_path))


def get_api_client(api_name: str) -> BatchAPIClient:
    """
    Factory function to get the appropriate API client.
    
    Args:
        api_name: One of "openai", "anthropic", "fireworks", "together"
        
    Returns:
        BatchAPIClient instance
    """
    clients = {
        "openai": OpenAIBatchClient,
        "anthropic": AnthropicBatchClient,
        "fireworks": FireworksBatchClient,
        "together": TogetherBatchClient
    }
    
    if api_name not in clients:
        raise ValueError(f"Unknown API: {api_name}. Must be one of {list(clients.keys())}")
    
    return clients[api_name]()