"""Data transformation utilities."""

from typing import Dict, Tuple


def parse_openai_response(response_line: dict) -> Tuple[str, str]:
    """
    Extract (custom_id, completion_text) from OpenAI batch response.
    
    Args:
        response_line: Single line from OpenAI batch output
        
    Returns:
        (custom_id, completion_text) tuple
    """
    custom_id = response_line["custom_id"]
    
    # Check if errored
    if response_line.get("error"):
        raise ValueError(f"Request {custom_id} failed: {response_line['error']}")
    
    # Extract completion text
    response_body = response_line["response"]["body"]
    completion = response_body["choices"][0]["message"]["content"]
    
    return custom_id, completion


def parse_anthropic_response(response_line: dict) -> Tuple[str, str]:
    """
    Extract (custom_id, completion_text) from Anthropic batch response.
    
    Args:
        response_line: Single line from Anthropic batch output
        
    Returns:
        (custom_id, completion_text) tuple
    """
    custom_id = response_line["custom_id"]
    result = response_line["result"]
    
    # Check result type
    if result["type"] == "errored":
        raise ValueError(f"Request {custom_id} failed: {result['error']}")
    elif result["type"] == "expired":
        raise ValueError(f"Request {custom_id} expired")
    elif result["type"] == "canceled":
        raise ValueError(f"Request {custom_id} was canceled")
    
    # Extract completion text
    message = result["message"]
    completion = message["content"][0]["text"]
    
    return custom_id, completion


def parse_fireworks_response(response_line: dict) -> Tuple[str, str]:
    """
    Extract (custom_id, completion_text) from Fireworks batch response.
    
    Args:
        response_line: Single line from Fireworks batch output
        
    Returns:
        (custom_id, completion_text) tuple
    """
    custom_id = response_line["custom_id"]
    
    # Check if errored
    if "error" in response_line and response_line["error"]:
        raise ValueError(f"Request {custom_id} failed: {response_line['error']}")
    
    # Extract completion text (similar to OpenAI format)
    response_body = response_line.get("response", {})
    if not response_body:
        raise ValueError(f"No response body for {custom_id}")
    
    completion = response_body["choices"][0]["message"]["content"]
    
    return custom_id, completion


def parse_together_response(response_line: dict) -> Tuple[str, str]:
    """
    Extract (custom_id, completion_text) from Together batch response.
    
    Args:
        response_line: Single line from Together batch output
        
    Returns:
        (custom_id, completion_text) tuple
    """
    custom_id = response_line["custom_id"]
    
    # Check if errored
    if "error" in response_line and response_line["error"]:
        raise ValueError(f"Request {custom_id} failed: {response_line['error']}")
    
    # Extract completion text
    response_body = response_line.get("body", {})
    if not response_body:
        raise ValueError(f"No response body for {custom_id}")
    
    completion = response_body["choices"][0]["message"]["content"]
    
    return custom_id, completion


def get_response_parser(api_name: str):
    """
    Get the appropriate response parser for an API.
    
    Args:
        api_name: One of "openai", "anthropic", "fireworks", "together"
        
    Returns:
        Parser function
    """
    parsers = {
        "openai": parse_openai_response,
        "anthropic": parse_anthropic_response,
        "fireworks": parse_fireworks_response,
        "together": parse_together_response
    }
    
    if api_name not in parsers:
        raise ValueError(f"Unknown API: {api_name}")
    
    return parsers[api_name]


def responses_to_output_dict(responses: list, custom_id_map: Dict[str, str]) -> Dict[str, str]:
    """
    Convert list of (custom_id, output) tuples to {uuid: output} dict.
    
    Args:
        responses: List of (custom_id, output_text) tuples
        custom_id_map: Dict mapping uuid -> custom_id
        
    Returns:
        Dict mapping uuid -> output_text
    """
    # Invert the map: custom_id -> uuid
    id_to_uuid = {custom_id: uuid for uuid, custom_id in custom_id_map.items()}
    
    result = {}
    for custom_id, output_text in responses:
        if custom_id not in id_to_uuid:
            print(f"Warning: Unknown custom_id {custom_id}, skipping")
            continue
        uuid = id_to_uuid[custom_id]
        result[uuid] = output_text
    
    return result