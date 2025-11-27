import argparse
import sys
import os


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


def list_together_models(search=None, model_type=None):
    """
    List Together AI models.

    Args:
        search: Search term to filter models
        model_type: Filter by model type (e.g., 'chat', 'language', 'image', 'embedding', 'moderation')
                   If None, includes all types that can be successfully parsed
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
        model_ids = []

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

            model_ids.append(model_id)

        return filter_models(model_ids, search)
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
        required=True,
        help="Provider (e.g., 'anthropic', 'together', 'fireworks')",
    )
    parser.add_argument(
        "-n", "--number", type=int, required=True, help="Number of models to list"
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
    args = parser.parse_args()

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
        # Together supports type filtering
        models = provider_config["func"](args.search, args.type)
    else:
        # Other providers don't support type filtering
        if args.type:
            print(
                "Warning: --type is only supported for Together provider, ignoring.",
                file=sys.stderr,
            )
        models = provider_config["func"](args.search)

    # Truncate and print
    for model_id in models[: args.number]:
        print(model_id)


if __name__ == "__main__":
    main()
