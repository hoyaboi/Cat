"""
Strategy comparison: visualize per-strategy attack success rates from a results JSON file.

Usage:
    python strategy_comparison.py --result results/results_llama_3_8b_164109.json
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


OUTPUT_DIR = "results/graphs"


def load_results(result_file: str):
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both formats: plain list or {"summary": ..., "tasks": [...]}
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    return data


def compute_strategy_stats(tasks: list) -> dict:
    """
    Returns {strategy: {"success": int, "total": int, "rate": float}}
    A task counts as a success for a strategy if categories[strategy]["success"] is True.
    """
    strategy_success = defaultdict(int)
    strategy_total = defaultdict(int)

    for task in tasks:
        categories = task.get("categories", {})
        for strategy, result in categories.items():
            strategy_total[strategy] += 1
            if result.get("success", False):
                strategy_success[strategy] += 1

    stats = {}
    for strategy in strategy_total:
        total = strategy_total[strategy]
        success = strategy_success[strategy]
        stats[strategy] = {
            "success": success,
            "total": total,
            "rate": success / total if total > 0 else 0.0,
        }

    return stats


def plot_strategy_comparison(stats: dict, result_file: str, output_dir: str) -> str:
    # Sort strategies by success rate (descending)
    sorted_strategies = sorted(stats.keys(), key=lambda s: stats[s]["rate"], reverse=True)
    rates = [stats[s]["rate"] * 100 for s in sorted_strategies]
    successes = [stats[s]["success"] for s in sorted_strategies]
    totals = [stats[s]["total"] for s in sorted_strategies]

    # Extract model name from result file path for the title
    base_name = os.path.splitext(os.path.basename(result_file))[0]  # e.g. results_llama_3_8b_164109

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"graph_strategies_{timestamp}.png")

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_strategies) * 0.9), 6))

    # Line + marker plot
    x = range(len(sorted_strategies))
    ax.plot(list(x), rates, marker="o", linewidth=2, markersize=8, color="#2563EB", zorder=3)

    # Annotate each point with rate% only
    for i, rate in enumerate(rates):
        ax.annotate(
            f"{rate:.1f}%",
            xy=(i, rate),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#1e3a5f",
        )

    # Grid
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, min(max(rates) * 1.35 + 5, 105))
    ax.set_xticks(list(x))
    ax.set_xticklabels(sorted_strategies, rotation=30, ha="right", fontsize=10)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Attack Success Rate", fontsize=12)
    ax.set_title(
        "Per-Strategy Attack Success Rate",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-strategy attack success rates from a results JSON."
    )
    parser.add_argument(
        "--result",
        type=str,
        required=True,
        help="Path to the results JSON file (e.g. results/results_llama_3_8b_164109.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to save the graph (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.result):
        print(f"[ERROR] Result file not found: {args.result}")
        return

    print(f"Loading results from: {args.result}")
    tasks = load_results(args.result)
    print(f"Loaded {len(tasks)} tasks")

    stats = compute_strategy_stats(tasks)

    print("\nPer-strategy success rates (sorted by rate):")
    for strategy, s in sorted(stats.items(), key=lambda x: x[1]["rate"], reverse=True):
        print(f"  {strategy:15s}: {s['success']:3d}/{s['total']:3d}  ({s['rate']*100:.1f}%)")

    output_path = plot_strategy_comparison(stats, args.result, args.output_dir)
    print(f"\nGraph saved to: {output_path}")


if __name__ == "__main__":
    main()
