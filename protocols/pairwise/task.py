"""Pairwise self-recognition tasks."""

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, GenerateConfig

from protocols.pairwise.config import PairwiseConfig
from protocols.pairwise.scorer import logprob_scorer
from protocols.pairwise.data import load_dataset


def comparison_self_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
    config: PairwiseConfig
) -> Task:
    """
    Base comparison self-recognition task.
    
    Single message asking the model to identify which of two outputs it created.
    Returns a Task object that can be returned directly or modified for variants.
    
    Args:
        model_name: Name of the model being evaluated
        alternative_model_name: Name of the alternative model for comparison
        dataset_name: Name of the dataset directory under data/
        model_generation_string: Generation identifier for the evaluated model
        alternative_model_generation_string: Generation identifier for alternative model
        config: PairwiseConfig with prompts and field names
        
    Returns:
        Task object configured with logprobs enabled
    """
    # Load dataset
    dataset_samples = load_dataset(
        model_name, alternative_model_name, dataset_name,
        model_generation_string, alternative_model_generation_string, config
    )
    
    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the prompt using the config template
        prompt = config.comparison_task_prompt.format(
            content_field=config.content_field,
            output_field=config.output_field,
            content=sample_data["content"],
            output1=sample_data["output1"],
            output2=sample_data["output2"]
        )
        
        inspect_samples.append(
            Sample(
                input=prompt,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"]
            )
        )
    
    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        config=GenerateConfig(logprobs=True, top_logprobs=2),
    )


def conversational_self_recognition(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
    config: PairwiseConfig
) -> Task:
    """
    Base conversational self-recognition task.
    
    Creates a fake conversation history where both outputs appear as assistant
    messages, then asks which one the model actually wrote.
    Returns a Task object that can be returned directly or modified for variants.
    
    Args:
        model_name: Name of the model being evaluated
        alternative_model_name: Name of the alternative model for comparison
        dataset_name: Name of the dataset directory under data/
        model_generation_string: Generation identifier for the evaluated model
        alternative_model_generation_string: Generation identifier for alternative model
        config: PairwiseConfig with prompts and field names
        
    Returns:
        Task object configured with logprobs enabled
    """
    # Load dataset
    dataset_samples = load_dataset(
        model_name, alternative_model_name, dataset_name,
        model_generation_string, alternative_model_generation_string, config
    )
    
    # Create Inspect samples with conversation history
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the generation prompt
        generation_prompt = config.conversational_generation_prompt.format(
            content_field=config.content_field,
            content=sample_data["content"]
        )
        
        # Build conversation history
        messages = [
            # First interaction with output1
            ChatMessageUser(content=generation_prompt),
            ChatMessageAssistant(content=sample_data["output1"]),
            # Second interaction with output2 (same article/question)
            ChatMessageUser(content=generation_prompt),
            ChatMessageAssistant(content=sample_data["output2"]),
            # Final verification question
            ChatMessageUser(content=config.conversational_verification_prompt)
        ]
        
        inspect_samples.append(
            Sample(
                input=messages,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"]
            )
        )
    
    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        config=GenerateConfig(logprobs=True, top_logprobs=2),
    )