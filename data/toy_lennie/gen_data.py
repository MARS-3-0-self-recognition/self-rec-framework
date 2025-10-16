# Sample(input="What is your name?", id="name"),
#             Sample(input="What is your favourite colour?", id="colour"),
#             Sample(input="Please tell me a story about your childhood.", id="story"),
#             Sample(input="Write a haiku about Autumn in Shoreditch.", id="haiku"),

import uuid
from pathlib import Path

from src.helpers.utils import save_json

# generated once manually by Lennie in the terminal at start of project
MY_DATASET_NAMESPACE = uuid.UUID("7bae975c-d5fb-416b-a803-4582e5ef1a2c")

if __name__ == "__main__":
    samples = [
        "What is your name?",
        "What is your favourite colour?",
        "Please tell me a story about your childhood.",
        "Write a haiku about Autumn in Shoreditch.",
    ]

    samples_dict = {
        str(uuid.uuid3(MY_DATASET_NAMESPACE, sample)): sample for sample in samples
    }

    dir_path = Path(__file__).parent
    save_path = dir_path / "questions.json"

    save_json(samples_dict, save_path)
