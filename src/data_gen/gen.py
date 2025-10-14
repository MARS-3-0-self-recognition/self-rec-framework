from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from src.helpers.model_names import inspect_model_name
from src.helpers.utils import data_dir, load_json
from src.protocols.pairwise.config import load_pairwise_config


@task
def base_generation(
    model_name: str,
    model_generation_string: str,
    dataset_name: str,
    pairwise_config_string: str,
) -> Task:
    """
    Base generation task.
    """
    contents = load_json(data_dir() / dataset_name / "input.json")
    config = load_pairwise_config(pairwise_config_string)
    model = inspect_model_name(model_name)

    # create base samples
    inspect_samples = []
    for uuid in contents.keys():
        sample_input = contents[uuid]
        generation_prompt = config.generation_prompt.format(
            content=sample_input,
        )
        metadata = {
            "uuid": uuid,
            "model_name": model_name,
            "model_generation_string": model_generation_string,
        }
        inspect_samples.append(
            Sample(
                input=generation_prompt,
                metadata=metadata,
            )
        )
        # TODO: optionally add a target later

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        model=model,
    )
