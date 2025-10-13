import argparse
import sys
import os


def list_anthropic_models(n):
    try:
        import anthropic
    except ImportError:
        print("anthropic package not found. Please install it first.", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    try:
        response = client.models.list()
        models = response.data
        print([m.id for m in models])
    except Exception as e:
        print(f"Error fetching Anthropic models: {e}", file=sys.stderr)


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
        help="Provider (e.g., 'anthropic', 'together')",
    )
    parser.add_argument(
        "-n", "--number", type=int, required=True, help="Number of models to list"
    )
    args = parser.parse_args()

    provider = args.provider.lower()
    n = args.number

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not found in environment.", file=sys.stderr)
            sys.exit(1)
        list_anthropic_models(n)
    else:
        print(f"Provider '{provider}' not supported.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
