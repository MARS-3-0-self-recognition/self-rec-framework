from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.solver import generate
from inspect_ai.scorer import Target, scorer, Score, mean, std
from inspect_ai.solver import TaskState
from src.helpers.model_names import inspect_model_name
from src.helpers.utils import (
    data_dir,
    load_json,
    rollout_eval_log_dir,
    rollout_json_file_path,
    save_json,
)
from src.protocols.pairwise.config import load_pairwise_config


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
                id=uuid,
                metadata=metadata,
            )
        )
        # TODO: optionally add a target later

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=answer_length_scorer(),
        model=model,
    )


def construct_rollout_dict(eval_log: EvalLog) -> dict[str, str]:
    """Parse the model outputs and uuid's into a dict"""
    rollout_dict = {}
    for sample in eval_log.samples:
        rollout_dict[sample.id] = sample.output.completion
    return rollout_dict


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
    parser.add_argument(
        "--pairwise_config_string",
        type=str,
        required=True,
        help="Pairwise config string",
    )
    ## possible TODO: add in 'process-only' capability
    # parser.add_argument(
    #     "--only_process",
    #     action="store_true",
    #     default=False,
    #     help="Only process existing eval log without rerunning evaluation",
    # )

    args = parser.parse_args()

    task = base_generation(
        model_name=args.model_name,
        model_generation_string=args.model_generation_string,
        dataset_name=args.dataset_name,
        pairwise_config_string=args.pairwise_config_string,
    )
    log_dir = str(
        rollout_eval_log_dir(
            args.dataset_name, args.model_name, args.model_generation_string
        )
    )
    print(f"Log directory: {log_dir}")
    eval_logs = eval(task, log_dir=log_dir)
    assert len(eval_logs) == 1, "Expected only one eval log"
    eval_log = eval_logs[0]
    rollout_dict = construct_rollout_dict(eval_log)

    save_dir = rollout_json_file_path(
        args.dataset_name, args.model_name, args.model_generation_string
    )
    save_json(rollout_dict, save_dir)

    print(f"Saved rollout json to {save_dir}")
