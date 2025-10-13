"""Protocol solvers for pairwise recognition.

Protocols construct the recognition prompt from templates and metadata,
then call generate() to get the model's response.
"""

from pathlib import Path

from inspect_ai.solver import solver, Generate, TaskState
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant


def load_template(template_path: str) -> str:
    """Load prompt template from file.

    Args:
        template_path: Path to template file

    Returns:
        Template string with {placeholder} format
    """
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return path.read_text().strip()


@solver
def pairwise_comparison(
    template_path: str, content_field: str = "Article", output_field: str = "Summary"
):
    """Comparison protocol: single message with both outputs.

    Constructs a user message presenting two outputs and asks the model
    to identify which one it generated.

    Args:
        template_path: Path to prompt template file
        content_field: Name for content field in template (e.g., "Article", "Question")
        output_field: Name for output field in template (e.g., "Summary", "Answer")

    Returns:
        Solver that constructs comparison prompt and generates response
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        template = load_template(template_path)

        # Format template with metadata
        prompt = template.format(
            content_field=content_field,
            output_field=output_field,
            content=state.metadata["content"],
            output1=state.metadata["output1"],
            output2=state.metadata["output2"],
        )

        # Add as user message
        state.messages.append(ChatMessageUser(content=prompt))

        # Generate response
        return await generate(state)

    return solve


@solver
def pairwise_conversational(
    generation_template_path: str,
    verification_template_path: str,
    content_field: str = "Article",
    output_field: str = "Summary",
):
    """Conversational protocol: fake conversation history.

    Creates a fake conversation where both outputs appear as assistant
    messages, then asks the model which one it actually wrote.

    This simulates the model encountering its own outputs in context
    (e.g., in conversation history or when monitoring assistant messages).

    Args:
        generation_template_path: Template for generation requests
        verification_template_path: Template for final verification question
        content_field: Content field name (e.g., "Article", "Question")
        output_field: Output field name (e.g., "Summary", "Answer")

    Returns:
        Solver that constructs conversational context and generates response
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        gen_template = load_template(generation_template_path)
        verify_template = load_template(verification_template_path)

        # First turn: request and output1
        gen_prompt_1 = gen_template.format(
            content_field=content_field, content=state.metadata["content"]
        )
        state.messages.extend(
            [
                ChatMessageUser(content=gen_prompt_1),
                ChatMessageAssistant(content=state.metadata["output1"]),
            ]
        )

        # Second turn: same request and output2
        gen_prompt_2 = gen_template.format(
            content_field=content_field,
            content=state.metadata["content"],  # Same content
        )
        state.messages.extend(
            [
                ChatMessageUser(content=gen_prompt_2),
                ChatMessageAssistant(content=state.metadata["output2"]),
            ]
        )

        # Verification question
        verify_prompt = verify_template.format(output_field=output_field)
        state.messages.append(ChatMessageUser(content=verify_prompt))

        # Generate response
        return await generate(state)

    return solve
