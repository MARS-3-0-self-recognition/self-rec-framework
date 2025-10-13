"""Main entry point using Hydra for config management.

Usage:
    python run.py model=claude alternative_model=gpt4
    python run.py model=claude alternative_model=gpt4 protocol=conversational
    python run.py experiment=hypothesis1_assistant_tags
"""

import hydra
from omegaconf import DictConfig
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.model import GenerateConfig

from src.protocols.pairwise.data_loader import load_pairwise_dataset
from src.utils.config import resolve_model_name


@hydra.main(
    config_path="configs", config_name="experiment/pairwise_base", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Run pairwise recognition experiment with Hydra config."""

    # Resolve model names
    model_name = resolve_model_name(cfg.model.short_name)
    alt_model_name = resolve_model_name(cfg.alternative_model.short_name)

    # Load dataset
    samples = load_pairwise_dataset(
        dataset_name=cfg.dataset.name,
        model_name=cfg.model.short_name,
        alternative_model_name=cfg.alternative_model.short_name,
        model_gen_string=cfg.model_gen_string,
        alternative_gen_string=cfg.alternative_gen_string,
        content_file=cfg.dataset.content_file,
        output_file=cfg.dataset.output_file,
    )

    # Instantiate protocol and scorer from config
    protocol_solver = hydra.utils.instantiate(cfg.protocol)
    task_scorer = hydra.utils.instantiate(cfg.scorer)

    # Create task
    task = Task(
        dataset=samples,
        solver=protocol_solver,
        scorer=task_scorer,
        config=GenerateConfig(
            logprobs=cfg.get("logprobs", True),
            top_logprobs=cfg.get("top_logprobs", 2),
            temperature=cfg.get("temperature", None),
        ),
    )

    # Run evaluation
    log_dir = Path(cfg.get("log_dir", "logs"))

    print(f"\n{'='*60}")
    print("Running Pairwise Recognition Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Alternative: {alt_model_name}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Protocol: {cfg.protocol._target_.split('.')[-1]}")
    print(f"Scorer: {cfg.scorer._target_.split('.')[-1]}")
    print(f"Samples: {len(samples)}")
    print(f"{'='*60}\n")

    eval(
        task,
        model=model_name,
        log_dir=str(log_dir),
    )


if __name__ == "__main__":
    main()
