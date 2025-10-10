from inspect_ai import task, Task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import generate


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
        scorer=None,
    )
