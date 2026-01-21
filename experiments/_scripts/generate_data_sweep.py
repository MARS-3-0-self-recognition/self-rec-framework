"""
Sweep data generation script for multiple models.

This script uses Inspect AI's multi-model parallelism to generate data for
multiple models simultaneously, then post-processes the results into separate
model directories.
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

from inspect_ai import eval

from src.inspect.tasks import generation
from src.inspect.config import create_generation_config
from src.helpers.utils import data_dir, save_json
from src.helpers.model_names import inspect_model_name, short_model_name
from utils import expand_model_names
from generate_data import (
    apply_treatments,
    construct_data_dicts,
    load_generation_config,
)

from inspect_ai.log import read_eval_log
import shutil


def _collect_partial_results(log_dir: Path, eval_logs: list, model_names: list[str]):
    """
    Try to collect any partial results from completed eval logs in the log directory.
    
    This is called when an exception occurs during eval() to recover any completed
    model generations.
    
    Args:
        log_dir: Directory where eval logs are saved
        eval_logs: List to append recovered logs to
        model_names: List of model names that were being processed
    """
    try:
        # Find all .eval files in the log directory
        eval_files = list(log_dir.glob("*.eval"))
        if not eval_files:
            return
        
        # Sort by modification time (newest first)
        eval_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Get the most recent eval files (up to the number of models)
        recent_files = eval_files[: len(model_names)]
        
        recovered = 0
        for eval_file in recent_files:
            try:
                log = read_eval_log(str(eval_file))
                # Check if this log was successful and has samples
                if log.status == "success" and log.samples:
                    # Check if we already have this log
                    existing_models = {getattr(l.eval, 'model', None) for l in eval_logs}
                    if log.eval.model not in existing_models:
                        eval_logs.append(log)
                        recovered += 1
            except Exception:
                continue
        
        if recovered > 0:
            print(f"  ✓ Recovered {recovered} completed model(s) from partial run")
    except Exception as e:
        print(f"  ⚠ Could not recover partial results: {e}")


def run_sweep_generation(
    model_names: list[str],
    dataset_path: str,
    dataset_config: str,
    overwrite: bool = False,
    batch: bool | int | str = False,
):
    """
    Generate data for multiple models using Inspect AI's multi-model parallelism.

    Uses a single eval() call with multiple models, then post-processes results
    to separate data by model and apply treatments.

    Args:
        model_names: List of model names to use for generation
        dataset_path: Path to input.json (e.g., 'data/wikisum/debug/input.json')
        dataset_config: Path to generation config YAML with temperature, treatments, etc.
        overwrite: If True, regenerate/reapply even if data exists
        batch: Enable batch mode for supported providers (OpenAI, Anthropic, Google, Together AI).
               Can be True (default config), int (batch size), or str (path to config file)
    """
    # Parse dataset path
    dataset_path_obj = Path(dataset_path)
    parts = dataset_path_obj.parts

    if parts[0] == "data" and parts[1] == "input":
        dataset_name = parts[2]
        data_subset = parts[3]
    elif parts[0] == "input":
        dataset_name = parts[1]
        data_subset = parts[2]
    else:
        raise ValueError(
            f"Invalid dataset path format: {dataset_path}. "
            f"Expected: data/input/dataset_name/data_subset/input.json"
        )

    total_models = len(model_names)

    print(f"\n{'=' * 70}")
    print("SWEEP DATA GENERATION (Multi-Model Parallel)")
    print(f"{'=' * 70}")
    print(f"Dataset: {dataset_name}/{data_subset}")
    print(f"Models to process: {total_models}")
    print(f"  {', '.join(model_names)}")
    print(f"Config: {dataset_config}")
    if batch:
        print("Batch mode: ENABLED")
    if overwrite:
        print("Mode: OVERWRITE (regenerating existing data)")
    else:
        print("Mode: SKIP (checking existing data)")
    print(f"{'=' * 70}\n")

    # Load generation config
    gen_config = load_generation_config(dataset_config)

    # Create ExperimentConfig for generation
    exp_config = create_generation_config(
        dataset_name=dataset_name,
        temperature=gen_config.get("temperature"),
        max_final_answer_tokens=gen_config.get("max_final_answer_tokens")
        or gen_config.get("max_tokens"),  # Backward compat
        seed=gen_config.get("seed"),
    )

    # Check which models need generation (skip if exists and not overwrite)
    models_to_generate = []
    skipped_models = []

    for model_name in model_names:
        output_path = (
            data_dir() / "input" / dataset_name / data_subset / model_name / "data.json"
        )
        if output_path.exists() and not overwrite:
            print(f"  ⊘ {model_name}: data already exists, skipping")
            skipped_models.append(model_name)
        else:
            models_to_generate.append(model_name)
            if output_path.exists():
                print(f"  → {model_name}: will overwrite existing data")
            else:
                print(f"  ✓ {model_name}: will generate")

    if not models_to_generate:
        print("\n⊘ All models already have data, nothing to generate")
        return

    # Pre-filter models that can't run in instruct mode (always-reasoning models without -thinking)
    # Import the helper here to avoid circular imports
    from src.inspect.tasks import _is_always_reasoning_model
    from src.helpers.model_names import is_thinking_model
    
    valid_models = []
    invalid_models = []
    
    for model_name in models_to_generate:
        if _is_always_reasoning_model(model_name) and not is_thinking_model(model_name):
            invalid_models.append(model_name)
            print(f"  ⚠ {model_name}: Cannot use in instruct mode (always uses reasoning)")
            print(f"     Use '{model_name}-thinking' instead.")
        else:
            valid_models.append(model_name)
    
    if invalid_models:
        print(f"\n⚠ Skipped {len(invalid_models)} always-reasoning model(s) without -thinking suffix")
    
    models_to_generate = valid_models
    
    if not models_to_generate:
        print("\n⊘ No valid models to generate (all require -thinking suffix)")
        return

    print(f"\n{'=' * 70}")
    print(f"Generating data for {len(models_to_generate)} models in parallel...")
    print(f"{'=' * 70}\n")

    # Create a separate task for each model
    # This is necessary because the generation() task bakes the model into the Task object
    # Separate models that don't support batch mode (Google Gemini, GPT-5/o-series)
    no_batch_tasks = []
    no_batch_models = []
    batch_tasks = []
    batch_models = []

    for model_name in models_to_generate:
        task = generation(
            model_name=model_name,
            dataset_name=dataset_name,
            data_subset=data_subset,
            exp_config=exp_config,
        )

        # Check if it's a model that doesn't support batch mode
        # Google/Gemini models have bugs in Inspect AI batch mode
        # GPT-5.1 and o3 models return unsupported_value errors in batch mode
        # GPT-4o-mini takes unreasonably long in batch mode
        # Grok models via XAI have batch API issues
        # ll-3.3-70b-dsR1-thinking (DeepSeek R1 Distill) has batch rejection issues
        # DeepSeek R1 full model has batch parsing issues (status_code returned as string)
        # Note: gpt-5 (without .1) is allowed to try batch mode
        # Note: Together AI batch mode is supported and works well (except DeepSeek models)
        is_gemini = "gemini" in model_name.lower()
        is_gpt5_1 = model_name.lower() == "gpt-5.1" or model_name.lower().startswith(
            "gpt-5.1"
        )
        is_gpt4o_mini = model_name.lower() in ["gpt-4o-mini", "gpt4o-mini"]
        is_o3 = model_name.lower().startswith("o3")
        is_grok = "grok" in model_name.lower()
        is_dsr1_distill = "dsr1" in model_name.lower()
        is_deepseek_r1 = "deepseek-r1" in model_name.lower()

        if is_gemini or is_gpt5_1 or is_gpt4o_mini or is_o3 or is_grok or is_dsr1_distill or is_deepseek_r1:
            no_batch_tasks.append(task)
            no_batch_models.append(model_name)
        else:
            batch_tasks.append(task)
            batch_models.append(model_name)

    # Set up shared log directory for generation
    log_dir = (
        data_dir() / "input" / dataset_name / data_subset / "_generation_logs_sweep"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # Run generation with multiple tasks in parallel
    # Use max_tasks to limit parallelism (default to number of models, max 16)
    total_tasks = len(no_batch_tasks) + len(batch_tasks)
    max_tasks = min(total_tasks, 16)

    if batch and no_batch_tasks:
        print(
            f"\n\033[91m⚠ WARNING: Batch mode disabled for {len(no_batch_tasks)} models"
        )
        print(
            "  (Google Gemini batch mode has bugs; GPT-5.1 returns unsupported_value errors)\033[0m"
        )
        print(f"  • Batch-compatible models: {len(batch_tasks)} WITH batch mode")
        print(f"  • Non-batch models: {len(no_batch_tasks)} WITHOUT batch mode\n")

    print(
        f"Running multi-model generation with max_tasks={max_tasks} (this may take a while with batch mode)...\n"
    )

    # Run evaluations - pass all tasks to a single eval() call for parallel execution
    # Inspect AI handles parallelism internally via max_tasks
    # Note: Multiple concurrent eval() calls from different threads are NOT allowed,
    # but a single eval() call with multiple tasks IS parallel
    eval_logs = []
    failed_models = []

    if batch and no_batch_tasks:
        # Run non-batch models first, then batch-compatible models with batch
        if no_batch_tasks:
            print(f"Running {len(no_batch_tasks)} models WITHOUT batch mode...")
            try:
                no_batch_logs = eval(
                    no_batch_tasks,
                    log_dir=str(log_dir),
                    max_tasks=max_tasks,
                    batch=False,
                )
                eval_logs.extend(no_batch_logs)
            except Exception as e:
                error_str = str(e)
                if "AlwaysReasoningModelError" in error_str or "always uses reasoning" in error_str:
                    print(f"\n⚠ One or more models cannot be used in instruct mode.")
                    print(f"   Use '-thinking' suffix for always-reasoning models.\n")
                else:
                    print(f"\n⚠ Error in non-batch generation: {error_str[:200]}\n")
                # Try to collect any partial results from log directory
                _collect_partial_results(log_dir, eval_logs, no_batch_models)

        if batch_tasks:
            print(f"\nRunning {len(batch_tasks)} models WITH batch mode...")
            try:
                batch_logs = eval(
                    batch_tasks,
                    log_dir=str(log_dir),
                    max_tasks=max_tasks,
                    batch=batch,
                )
                eval_logs.extend(batch_logs)
            except Exception as e:
                error_str = str(e)
                # Check for OpenAI batch API "missing field `name`" error
                if "missing field `name`" in error_str or "UnprocessableEntityError" in error_str:
                    print(f"\n{'=' * 70}")
                    print("⚠ BATCH API ERROR: Missing 'name' field in batch creation")
                    print(f"{'=' * 70}")
                    print(
                        "This appears to be an Inspect AI bug where the 'name' field is"
                        " sometimes missing when creating OpenAI batches."
                    )
                    print("\nSuggested actions:")
                    print("  1. Re-run the script - this error is often intermittent")
                    print(
                        "  2. Check for active batches that may have been created: "
                        "scripts/list_active_batches.py"
                    )
                    print(
                        "  3. If batches were created, wait for completion or cancel them: "
                        "scripts/cancel_all_batches.py"
                    )
                    print(f"{'=' * 70}\n")
                elif "AlwaysReasoningModelError" in error_str or "always uses reasoning" in error_str:
                    print(f"\n⚠ One or more models cannot be used in instruct mode.")
                    print(f"   Use '-thinking' suffix for always-reasoning models.\n")
                else:
                    print(f"\n⚠ Error in batch generation: {error_str[:200]}\n")
                # Try to collect any partial results from log directory
                _collect_partial_results(log_dir, eval_logs, batch_models)
    else:
        # Run all together
        all_tasks = no_batch_tasks + batch_tasks
        all_models = no_batch_models + batch_models
        try:
            eval_logs = eval(
                all_tasks,
                log_dir=str(log_dir),
                max_tasks=max_tasks,
                batch=batch,
            )
        except Exception as e:
            error_str = str(e)
            # Check for OpenAI batch API "missing field `name`" error
            if "missing field `name`" in error_str or "UnprocessableEntityError" in error_str:
                print(f"\n{'=' * 70}")
                print("⚠ BATCH API ERROR: Missing 'name' field in batch creation")
                print(f"{'=' * 70}")
                print(
                    "This appears to be an Inspect AI bug where the 'name' field is"
                    " sometimes missing when creating OpenAI batches."
                )
                print("\nSuggested actions:")
                print("  1. Re-run the script - this error is often intermittent")
                print(
                    "  2. Check for active batches that may have been created: "
                    "scripts/list_active_batches.py"
                )
                print(
                    "  3. If batches were created, wait for completion or cancel them: "
                    "scripts/cancel_all_batches.py"
                )
                print(f"{'=' * 70}\n")
            elif "AlwaysReasoningModelError" in error_str or "always uses reasoning" in error_str:
                print(f"\n⚠ One or more models cannot be used in instruct mode.")
                print(f"   Use '-thinking' suffix for always-reasoning models.\n")
            else:
                print(f"\n⚠ Error in generation: {error_str[:200]}\n")
            # Try to collect any partial results from log directory
            _collect_partial_results(log_dir, eval_logs, all_models)

    # Summary of results
    total_models_attempted = len(no_batch_tasks) + len(batch_tasks)
    print(f"\n{'='*70}")
    print("GENERATION SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Successful: {len(eval_logs)}/{total_models_attempted} models")
    if failed_models:
        print(f"✗ Failed: {len(failed_models)}/{total_models_attempted} models")
        print(f"\nFailed models:")
        for model_name, error_str in failed_models:
            error_summary = error_str.split("\n")[0][:80]  # First line, truncated
            print(f"  • {model_name}: {error_summary}")
        print(f"\n💡 Tip: Re-run the script to retry failed models, or run them individually.")
    print(f"{'='*70}\n")
    
    if len(eval_logs) == 0:
        print("⚠ No models succeeded! Cannot continue with processing.")
        print("  Check errors above and retry.")
        return
    
    print(f"Processing {len(eval_logs)} successful model outputs...\n")

    # Post-process: Separate outputs by model and save to individual directories
    successful = []
    failed = []

    for idx, eval_log in enumerate(eval_logs):
        model_name = None
        try:
            # Get inspect model name from eval log
            full_model_name = eval_log.eval.model

            # First try: match against models_to_generate list to preserve original model names
            # This ensures we use the exact model name requested (e.g., "sonnet-4.5" not "sonnet-4.5-thinking")
            for short_name in models_to_generate:
                try:
                    if inspect_model_name(short_name) == full_model_name:
                        model_name = short_name
                        break
                except KeyError:
                    # Model name not in INSPECT_MODEL_NAMES, skip this candidate
                    continue

            # Second try: use index-based fallback (eval_logs order matches task submission order)
            # Skip short_model_name() because it can't distinguish between variants like
            # "sonnet-4.5" vs "sonnet-4.5-thinking" that map to the same API endpoint
            if model_name is None and idx < len(models_to_generate):
                model_name = models_to_generate[idx]
                print(
                    f"  Warning: Could not map '{full_model_name}', using index-based fallback: {model_name}"
                )

            if model_name is None:
                raise ValueError(
                    f"Could not determine short model name for '{full_model_name}'"
                )

            print(f"  Processing outputs for {model_name} (from {full_model_name})...")

            # Extract outputs (completion + CoT + signatures, if present)
            data_dict, cot_dict, signature_dict = construct_data_dicts(eval_log)

            # Save to model-specific directory
            output_path = (
                data_dir()
                / "input"
                / dataset_name
                / data_subset
                / model_name
                / "data.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(data_dict, output_path)

            cot_path = None
            if cot_dict:
                cot_path = output_path.with_name("data_cot.json")
                save_json(cot_dict, cot_path)

            signature_path = None
            if signature_dict:
                signature_path = output_path.with_name("data_signatures.json")
                save_json(signature_dict, signature_path)

            print(f"  ✓ {model_name}: Saved {len(data_dict)} samples to {output_path}")
            if cot_path:
                print(
                    f"  ✓ {model_name}: Saved {len(cot_dict)} CoT samples to {cot_path}"
                )
            if signature_path:
                print(
                    f"  ✓ {model_name}: Saved {len(signature_dict)} signature samples to {signature_path}"
                )

            # Move eval file to model directory with a meaningful name
            if hasattr(eval_log, 'location') and eval_log.location:
                eval_source = Path(eval_log.location)
                if eval_source.exists():
                    eval_dest = output_path.parent / "generation.eval"
                    try:
                        shutil.copy2(eval_source, eval_dest)
                        print(f"  ✓ {model_name}: Copied eval log to {eval_dest}")
                    except Exception as copy_err:
                        print(f"  ⚠ {model_name}: Could not copy eval file: {copy_err}")

            successful.append(model_name)

        except Exception as e:
            error_model = model_name if model_name else f"eval_log[{idx}]"
            print(f"  ✗ Error processing {error_model}: {e}")
            failed.append((error_model, str(e)))

    # Clean up the shared log directory if all files were successfully copied
    if successful and log_dir.exists():
        try:
            # Only remove eval files that were successfully processed
            remaining_files = list(log_dir.glob("*.eval"))
            if len(remaining_files) <= len(eval_logs) - len(successful):
                # All processed logs were copied, safe to clean up
                shutil.rmtree(log_dir)
                print(f"  ✓ Cleaned up shared log directory: {log_dir}")
            else:
                print(f"  ⊘ Keeping shared log directory (contains unprocessed files): {log_dir}")
        except Exception as cleanup_err:
            print(f"  ⚠ Could not clean up log directory: {cleanup_err}")

    # Apply treatments to each model's data
    print(f"\n{'=' * 70}")
    print("Applying treatments...")
    print(f"{'=' * 70}\n")

    for model_name in successful:
        base_data_path = (
            data_dir() / "input" / dataset_name / data_subset / model_name / "data.json"
        )

        try:
            apply_treatments(
                base_data_path=base_data_path,
                dataset_name=dataset_name,
                data_subset=data_subset,
                model_name=model_name,
                gen_config=gen_config,
                overwrite=overwrite,
            )
        except Exception as e:
            print(f"  ✗ Error applying treatments for {model_name}: {e}")
            failed.append((model_name, f"Treatment error: {str(e)}"))

    # Summary
    print(f"\n{'=' * 70}")
    print("SWEEP GENERATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total models requested: {total_models}")
    print(f"Skipped (already exist): {len(skipped_models)}")
    print(f"Generated: {len(successful)}")

    if successful:
        print("\nSuccessful:")
        for model in successful:
            print(f"  ✓ {model}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for model, error in failed[:10]:
            error_msg = error[:80] if len(error) > 80 else error
            print(f"  ✗ {model}: {error_msg}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    load_dotenv(override=True)

    # Preprocess sys.argv to handle -set before argparse sees it
    # argparse treats -set as a flag, so we need to escape it
    import sys
    if "--model_names" in sys.argv:
        model_names_idx = sys.argv.index("--model_names")
        # Find where model_names arguments end (next flag or end of args)
        for i in range(model_names_idx + 1, len(sys.argv)):
            if sys.argv[i] == "-set" and (i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--")):
                # Replace -set with a placeholder that doesn't start with -
                sys.argv[i] = "SET_PLACEHOLDER"
    
    parser = argparse.ArgumentParser(
        description="Sweep generate data using multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # Disable abbreviation to allow -set as a value
        epilog="""
Examples:
  # Individual model names
  python experiments/scripts/generate_data_sweep.py \\
    --model_names haiku-3-5 sonnet-3-7 gpt-4 \\
    --dataset_path=data/wikisum/debug/input/input.json \\
    --dataset_config=experiments/00_data_gen/configs/config.yaml

  # Use a model set (e.g., gen_cot models)
  python experiments/scripts/generate_data_sweep.py \\
    --model_names -set cot \\
    --dataset_path=data/input/sharegpt/english_26/input.json \\
    --dataset_config=experiments/00_data_gen/configs/config.yaml

  # Mix individual models and sets
  python experiments/scripts/generate_data_sweep.py \\
    --model_names haiku-3-5 -set dr gpt-4.1 \\
    --dataset_path=data/wikisum/debug/input/input.json \\
    --dataset_config=experiments/00_data_gen/configs/config.yaml
        """,
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names (e.g., 'haiku-3-5 sonnet-3-5 gpt-4') or model sets (e.g., '-set cot' for gen_cot set). "
        "Can mix individual models and sets.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to input.json (e.g., 'data/wikisum/debug/input/input.json')",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to generation config YAML (e.g., 'experiments/00_data_gen/configs/config.yaml')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing data files (default: skip existing files)",
    )
    parser.add_argument(
        "--batch",
        nargs="?",
        const=True,
        default=False,
        help="Enable batch mode for supported providers (OpenAI, Anthropic, Google, Together AI). "
        "Usage: --batch (default config), --batch 1000 (batch size), --batch config.yaml (config file)",
    )

    args = parser.parse_args()

    # Restore -set from placeholder
    if args.model_names:
        args.model_names = [arg.replace("SET_PLACEHOLDER", "-set") for arg in args.model_names]
    
    # Expand model set references (e.g., '-set cot' -> list of models)
    expanded_model_names = expand_model_names(args.model_names)

    # Parse batch argument
    batch_value = args.batch
    if batch_value is not False:
        # Try to convert to int if it's a number string
        if isinstance(batch_value, str) and batch_value.isdigit():
            batch_value = int(batch_value)

    run_sweep_generation(
        model_names=expanded_model_names,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        overwrite=args.overwrite,
        batch=batch_value,
    )
