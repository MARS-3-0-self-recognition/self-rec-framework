import argparse
import re
import sys
import os


def generate_shorthand(display_name):
    """
    Generate a shorthand name from a display name.

    Examples:
        "DeepSeek R1-0528" -> "deepseek-r1-0528"
        "DeepSeek V3.1" -> "deepseek-v3.1"
        "Meta Llama 3.1 8B Instruct Turbo" -> "meta-llama-3.1-8b"
        "Qwen 2.5 7B Instruct Turbo" -> "qwen-2.5-7b"
        "Kimi K2 Thinking" -> "kimi-k2-thinking"
    """
    if not display_name:
        return ""

    name = display_name.strip()

    # Strip common suffixes (order matters — strip longer patterns first)
    strip_suffixes = [
        "Instruct Turbo",
        "Instruct",
        "Turbo",
    ]
    for suffix in strip_suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()

    # Remove parenthetical notes like "(original)"
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name).strip()

    # Lowercase
    name = name.lower()

    # Normalize separators: replace spaces with hyphens, collapse multiples
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-")

    return name


def filter_models(models, search_term=None):
    """Filter models by search term (case-insensitive)."""
    if not search_term:
        return models
    search_lower = search_term.lower()
    return [m for m in models if search_lower in m.lower()]


def list_anthropic_models(search=None):
    try:
        import anthropic
    except ImportError:
        print("anthropic package not found. Please install it first.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    try:
        response = client.models.list()
        model_ids = [m.id for m in response.data]
        return filter_models(model_ids, search)
    except Exception as e:
        print(f"Error fetching Anthropic models: {e}", file=sys.stderr)
        return []


def check_together_availability(model_ids, api_key, max_workers=10):
    """
    Check which Together AI models are available for serverless inference.

    Probes each model with a minimal 1-token request to determine availability.
    Returns a dict mapping model_id -> status string.
    """
    import concurrent.futures
    import requests

    def _check_one(model_id):
        try:
            resp = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                return model_id, "serverless"
            data = resp.json()
            err = data.get("error", {})
            code = err.get("code", "")
            if code == "model_not_available":
                return model_id, "dedicated-only"
            return model_id, f"error ({resp.status_code})"
        except Exception as e:
            return model_id, f"error ({e})"

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_check_one, mid): mid for mid in model_ids}
        for future in concurrent.futures.as_completed(futures):
            model_id, status = future.result()
            results[model_id] = status
    return results


def list_together_models(search=None, model_type=None):
    """
    List Together AI models.

    Args:
        search: Search term to filter models
        model_type: Filter by model type (e.g., 'chat', 'language', 'image', 'embedding', 'moderation')
                   If None, includes all types that can be successfully parsed

    Returns:
        List of dicts with keys: 'id', 'display_name'
    """
    try:
        import requests
    except ImportError:
        print("requests package not found. Please install it first.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("TOGETHER_API_KEY not found in environment.", file=sys.stderr)
        return []

    try:
        # Use direct API call to avoid Pydantic validation issues
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        response = requests.get("https://api.together.xyz/v1/models", headers=headers)
        response.raise_for_status()

        models_data = response.json()
        results = []

        for model in models_data:
            # Extract model ID
            model_id = model.get("id") or model.get("name")
            if not model_id:
                continue

            # Filter by model type if specified
            if model_type:
                model_model_type = model.get("type", "")
                if model_model_type != model_type:
                    continue

            # Filter by search term
            if search and search.lower() not in model_id.lower():
                continue

            results.append({
                "id": model_id,
                "display_name": model.get("display_name", ""),
            })

        return results
    except Exception as e:
        print(f"Error fetching Together models: {e}", file=sys.stderr)
        return []


def list_fireworks_models(search=None):
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not found. Please install it first.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=os.getenv("FIREWORKS_API_KEY"),
        base_url="https://api.fireworks.ai/inference/v1",
    )
    try:
        response = client.models.list()
        model_ids = [m.id for m in response]
        return filter_models(model_ids, search)
    except Exception as e:
        print(f"Error fetching Fireworks models: {e}", file=sys.stderr)
        return []


def list_openai_models(search=None):
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not found. Please install it first.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.models.list()
        model_ids = [m.id for m in response]
        return filter_models(model_ids, search)
    except Exception as e:
        print(f"Error fetching OpenAI models: {e}", file=sys.stderr)
        return []


def list_google_models(search=None):
    try:
        import google.generativeai as genai
    except ImportError:
        print(
            "google-generativeai package not found. Please install it first.",
            file=sys.stderr,
        )
        sys.exit(1)

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    try:
        models = genai.list_models()
        model_ids = [m.name for m in models]
        return filter_models(model_ids, search)
    except Exception as e:
        print(f"Error fetching Google models: {e}", file=sys.stderr)
        return []


def main():
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        print(
            "python-dotenv not installed. Please install it to load .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="List provider models.")
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default=None,
        help="Provider (e.g., 'anthropic', 'together', 'fireworks'). Required unless --local is used.",
    )
    parser.add_argument(
        "-n", "--number", type=int, default=None, help="Number of models to list (required unless --local)"
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        default=False,
        help="List local shorthand model name mappings from INSPECT_MODEL_NAMES. "
        "Optionally filter by -p (provider) and -s (search).",
    )
    parser.add_argument(
        "-s",
        "--search",
        type=str,
        default=None,
        help="Search term to filter models (case-insensitive)",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default=None,
        help="Model type to filter (Together only: 'chat', 'language', 'image', 'embedding', 'moderation', 'video', etc.)",
    )
    parser.add_argument(
        "-c",
        "--check-availability",
        action="store_true",
        default=False,
        help="Check serverless availability for each model (Together only). "
        "Probes each model with a minimal request to detect dedicated-only models.",
    )
    args = parser.parse_args()

    # Handle --local mode: list shorthand mappings from INSPECT_MODEL_NAMES
    if args.local:
        from self_rec_framework.src.helpers.model_names import INSPECT_MODEL_NAMES

        # Map provider flag values to inspect name prefixes
        provider_prefixes = {
            "anthropic": "anthropic/",
            "together": "together/",
            "fireworks": "fireworks/",
            "openai": "openai/",
            "google": "google/",
        }

        # Collect existing entries from INSPECT_MODEL_NAMES
        entries = []  # list of (shorthand, inspect_name, is_added)
        added_inspect_names = set()
        for shorthand, inspect_name in INSPECT_MODEL_NAMES.items():
            # Filter by provider if specified
            if args.provider:
                prefix = provider_prefixes.get(args.provider)
                if prefix and not inspect_name.startswith(prefix):
                    continue

            # Filter by search term (matches against both shorthand and inspect name)
            if args.search:
                search_lower = args.search.lower()
                if search_lower not in shorthand.lower() and search_lower not in inspect_name.lower():
                    continue

            entries.append((shorthand, inspect_name, True))
            added_inspect_names.add(inspect_name)

        # For Together provider, also fetch from API and suggest shorthands for unadded models
        if args.provider == "together":
            # Collect all Together inspect names (not just filtered ones) for dedup
            all_together_inspect_names = {
                v for v in INSPECT_MODEL_NAMES.values() if v.startswith("together/")
            }

            together_models = list_together_models(args.search, args.type)
            for model in together_models:
                inspect_name = f"together/{model['id']}"
                if inspect_name in all_together_inspect_names:
                    continue  # already shown above (or filtered out)
                display = model.get("display_name", "")
                suggested = generate_shorthand(display) if display else model["id"].split("/")[-1].lower()
                entries.append((suggested, f"{inspect_name}  ({display})" if display else inspect_name, False))

        if args.number:
            entries = entries[: args.number]

        for shorthand, inspect_name, is_added in entries:
            status = "[ADDED]" if is_added else "[NOT ADDED]"
            print(f"{status}  {shorthand}: {inspect_name}")
        return

    # Validate required args for non-local mode
    if not args.provider:
        parser.error("-p/--provider is required unless --local is used")
    if args.number is None:
        parser.error("-n/--number is required unless --local is used")

    provider = args.provider

    # Provider configuration
    providers = {
        "anthropic": {
            "env_key": "ANTHROPIC_API_KEY",
            "func": list_anthropic_models,
        },
        "together": {
            "env_key": "TOGETHER_API_KEY",
            "func": list_together_models,
        },
        "fireworks": {
            "env_key": "FIREWORKS_API_KEY",
            "func": list_fireworks_models,
        },
        "openai": {
            "env_key": "OPENAI_API_KEY",
            "func": list_openai_models,
        },
        "google": {
            "env_key": "GOOGLE_API_KEY",
            "func": list_google_models,
        },
    }

    assert (
        provider in providers
    ), f"Provider '{provider}' not supported. Supported providers: {', '.join(providers.keys())}"

    provider_config = providers[provider]
    if not os.getenv(provider_config["env_key"]):
        print(
            f"{provider_config['env_key']} not found in environment.", file=sys.stderr
        )
        sys.exit(1)

    # Fetch and filter models
    if provider == "together":
        from self_rec_framework.src.helpers.model_names import INSPECT_MODEL_NAMES

        # Build reverse lookup: together inspect_name -> list of shorthands
        inspect_reverse = {}
        for shorthand, inspect_name in INSPECT_MODEL_NAMES.items():
            if inspect_name.startswith("together/"):
                inspect_reverse.setdefault(inspect_name, []).append(shorthand)

        # Together returns list of dicts with 'id' and 'display_name'
        together_models = provider_config["func"](args.search, args.type)
        together_models = together_models[: args.number]

        model_ids = [m["id"] for m in together_models]
        display_names = {m["id"]: m["display_name"] for m in together_models}

        # Check availability if requested
        availability = {}
        if args.check_availability and model_ids:
            api_key = os.getenv(provider_config["env_key"])
            print(f"Checking serverless availability for {len(model_ids)} models...", file=sys.stderr)
            availability = check_together_availability(model_ids, api_key)

        for model_id in model_ids:
            parts = [model_id]
            display = display_names.get(model_id, "")
            if display:
                parts.append(f"  ({display})")

            # Show shorthand status
            inspect_key = f"together/{model_id}"
            existing = inspect_reverse.get(inspect_key)
            if existing:
                parts.append(f"  [ADDED: {', '.join(existing)}]")
            else:
                suggested = generate_shorthand(display) if display else ""
                if suggested:
                    parts.append(f"  [NOT ADDED — suggested: {suggested}]")
                else:
                    parts.append("  [NOT ADDED]")

            if args.check_availability:
                status = availability.get(model_id, "unknown")
                if status == "dedicated-only":
                    parts.append("  [DEDICATED-ONLY]")
            print("".join(parts))
    else:
        # Other providers return plain lists of model IDs
        if args.type:
            print(
                "Warning: --type is only supported for Together provider, ignoring.",
                file=sys.stderr,
            )
        if args.check_availability:
            print(
                "Warning: --check-availability is only supported for Together provider, ignoring.",
                file=sys.stderr,
            )
        models = provider_config["func"](args.search)
        for model_id in models[: args.number]:
            print(model_id)


if __name__ == "__main__":
    main()
