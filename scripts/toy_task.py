import json
from inspect_ai import task, Task
from inspect_ai import solver
from inspect_ai.scorer import Target, Score, accuracy, stderr, scorer
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import TaskState, generate


@scorer(metrics=[accuracy(), stderr()])
def identity_scorer():
    """Scorer that puts model output as 'answer' such that it shows in .eval table."""

    async def score(state: TaskState, target: Target) -> Score:
        return Score(
            value=0,
            answer=state.output.completion,
        )

    return score


@solver
def save_output(jsonfile_path: str):
    """Saves output to a json file, where uuid is the key and output is the value."""

    async def save(state: TaskState, target: Target) -> TaskState:
        with open(jsonfile_path, "w") as f:
            json.dump({state.metadata["uuid"]: state.output.completion}, f)
        return state

    return save


@task
def toy_task():
    dataset = MemoryDataset(
        [
            Sample(input="What is your name?", id="name"),
            Sample(input="What is your favourite colour?", id="colour"),
            Sample(input="Please tell me a story about your childhood.", id="story"),
            Sample(input="Write a haiku about Autumn in Shoreditch.", id="haiku"),
        ]
    )

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=identity_scorer(),
    )
