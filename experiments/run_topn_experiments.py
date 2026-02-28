"""
Top-N strategy experiment runner.

Evaluates attack performance when using top-2, top-3, ..., top-10 strategies
from the ordered STRATEGIES list in src/utils/attack.py.

Experiments run in parallel batches (default: 3 at a time) against a shared
vLLM server so the model is loaded only once.

Prerequisites:
    1. Pre-generate dictionaries:
           python gen_dictionary.py

    2. Start the vLLM server in a separate terminal:
           vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000 --api-key "my-key"

    3. Set VLLM_API_KEY in .env (must match --api-key above):
           VLLM_API_KEY=my-key

Usage:
    python run_topn_experiments.py
    python run_topn_experiments.py --target-model llama-3-8b-vllm --judge-model gpt-4o-mini
    python run_topn_experiments.py --limit 50 --parallel 3 --top-min 2 --top-max 10
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Ensure project root is on sys.path when running from experiments/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import run_attack_pipeline, STRATEGIES, DICTIONARY_DIR


def run_single_experiment(args: tuple) -> tuple:
    """
    Run a single top-n experiment.
    This function runs inside a subprocess (spawned by ProcessPoolExecutor).

    Args:
        args: (top_n, target_model, judge_model, limit, dictionary_dir, output_dir)

    Returns:
        (top_n, result_file_path)
    """
    top_n, target_model, judge_model, limit, dictionary_dir, output_dir = args

    strategies = STRATEGIES[:top_n]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    model_name = (target_model or "gpt-4o-mini").replace("-", "_").replace(".", "_")
    output_file = os.path.join(output_dir, f"results_top{top_n}_{model_name}_{timestamp}.json")

    print(f"[top-{top_n}] Starting | strategies: {strategies}", flush=True)

    result_file = run_attack_pipeline(
        strategies=strategies,
        target_model=target_model,
        judge_model=judge_model,
        output_file=output_file,
        limit=limit,
        early_stop=True,
        dictionary_dir=dictionary_dir,
    )

    print(f"[top-{top_n}] ✓ Done → {result_file}", flush=True)
    return top_n, result_file


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run top-n strategy experiments (top-2 to top-10) in parallel.\n"
            "Requires a running vLLM server. See module docstring for setup."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--target-model",
        type=str,
        default="llama-3-8b-vllm",
        help="Target LLM model name (default: llama-3-8b-vllm)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge LLM model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries per experiment (default: all)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Number of experiments to run in parallel (default: 3)",
    )
    parser.add_argument(
        "--top-min",
        type=int,
        default=2,
        help="Minimum top-n to evaluate (default: 2)",
    )
    parser.add_argument(
        "--top-max",
        type=int,
        default=10,
        help="Maximum top-n to evaluate (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/topn",
        help="Directory to save experiment results (default: results/topn)",
    )
    parser.add_argument(
        "--dictionary-dir",
        type=str,
        default=DICTIONARY_DIR,
        help=f"Directory containing pre-generated dictionaries (default: {DICTIONARY_DIR})",
    )

    args = parser.parse_args()

    # Validate range
    if args.top_min < 1:
        parser.error("--top-min must be >= 1")
    if args.top_max > len(STRATEGIES):
        parser.error(f"--top-max must be <= {len(STRATEGIES)} (total number of strategies)")
    if args.top_min > args.top_max:
        parser.error("--top-min must be <= --top-max")

    top_n_range = list(range(args.top_min, args.top_max + 1))
    total_experiments = len(top_n_range)

    print("=" * 70)
    print("Top-N Strategy Experiment Runner")
    print("=" * 70)
    print(f"Experiments   : top-{args.top_min} to top-{args.top_max} ({total_experiments} total)")
    print(f"Parallel      : {args.parallel}")
    print(f"Target model  : {args.target_model}")
    print(f"Judge model   : {args.judge_model or 'gpt-4o-mini (default)'}")
    print(f"Query limit   : {args.limit or 'all'}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Dictionary dir: {args.dictionary_dir}")
    print(f"\nSTRATEGIES (ordered, first N will be used):")
    for i, s in enumerate(STRATEGIES, 1):
        print(f"  {i:2d}. {s}")
    print("=" * 70)
    print()

    experiment_args = [
        (
            n,
            args.target_model,
            args.judge_model,
            args.limit,
            args.dictionary_dir,
            args.output_dir,
        )
        for n in top_n_range
    ]

    completed_results: dict = {}
    failed: list = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        future_to_n = {
            executor.submit(run_single_experiment, ea): ea[0]
            for ea in experiment_args
        }
        for future in as_completed(future_to_n):
            top_n = future_to_n[future]
            try:
                n, result_file = future.result()
                completed_results[n] = result_file
                done = len(completed_results) + len(failed)
                print(f"[{done}/{total_experiments}] top-{n} completed → {result_file}", flush=True)
            except Exception as exc:
                failed.append(top_n)
                done = len(completed_results) + len(failed)
                print(f"[{done}/{total_experiments}] top-{top_n} FAILED: {exc}", flush=True)

    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("All experiments finished")
    print("=" * 70)
    print(f"Total time : {_format_time(total_time)}")
    print(f"Succeeded  : {len(completed_results)}/{total_experiments}")
    if failed:
        print(f"Failed     : {sorted(failed)}")
    print()
    print("Result files (sorted by top-n):")
    for n in sorted(completed_results):
        print(f"  top-{n:2d} → {completed_results[n]}")


if __name__ == "__main__":
    main()

