from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.solver import generate
from inspect_ai.scorer import Target, scorer, Score, mean, std
from inspect_ai.solver import TaskState
from inspect_ai.model import GenerateConfig

from src.helpers.model_names import inspect_model_name
from src.helpers.utils import (
    data_dir,
    load_json,
    rollout_eval_log_dir,
    rollout_json_file_path,
    save_json,
)
from src.data_gen.config import load_generation_config


@scorer(metrics=[mean(), std()])
def answer_length_scorer():
    """Scorer that makes model output as 'answer' such that it shows in .eval table and also counts number of characters in answer."""

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        return Score(
            value=len(answer),
            uuid=state.metadata["uuid"],
            answer=answer,
        )

    return score


@task
def base_generation(
    model_name: str,
    model_generation_string: str,
    dataset_name: str,
) -> Task:
    """
    Base generation task.
    """
    contents = load_json(data_dir() / dataset_name / "input.json")
    config = load_generation_config(model_generation_string)
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
                id=uuid,
                metadata=metadata,
            )
        )
        # TODO: optionally add a target later

    # Create GenerateConfig with only non-None parameters
    generate_config_params = {}
    if config.system_prompt is not None:
        generate_config_params["system_message"] = config.system_prompt
    if config.temperature is not None:
        generate_config_params["temperature"] = config.temperature
    if config.max_tokens is not None:
        generate_config_params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        generate_config_params["top_p"] = config.top_p
    if config.top_k is not None:
        generate_config_params["top_k"] = config.top_k
    if config.seed is not None:
        generate_config_params["seed"] = config.seed
    if config.stop_seqs is not None:
        generate_config_params["stop_seqs"] = config.stop_seqs

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=answer_length_scorer(),
        model=model,
        config=GenerateConfig(**generate_config_params)
        if generate_config_params
        else None,
    )


def construct_rollout_dict(eval_log: EvalLog) -> dict[str, str]:
    """Parse the model outputs and uuid's into a dict"""
    rollout_dict = {}
    for sample in eval_log.samples:
        rollout_dict[sample.id] = sample.output.completion
    return rollout_dict


def run_base_generation(
    model_name: str,
    model_generation_string: str,
    dataset_name: str,
):
    task = base_generation(
        model_name=model_name,
        model_generation_string=model_generation_string,
        dataset_name=dataset_name,
    )
    log_dir = str(
        rollout_eval_log_dir(dataset_name, model_name, model_generation_string)
    )
    print(f"Log directory: {log_dir}")
    eval_logs = eval(task, log_dir=log_dir)
    assert len(eval_logs) == 1, "Expected only one eval log"
    eval_log = eval_logs[0]
    rollout_dict = construct_rollout_dict(eval_log)

    save_dir = rollout_json_file_path(dataset_name, model_name, model_generation_string)
    save_json(rollout_dict, save_dir)

    print(f"Saved rollout json to {save_dir}")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Run base generation task")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--model_generation_string",
        type=str,
        required=True,
        help="Model generation string",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")

    args = parser.parse_args()

    run_base_generation(
        model_name=args.model_name,
        model_generation_string=args.model_generation_string,
        dataset_name=args.dataset_name,
    )
