"""
Top-N strategy comparison: visualize attack success rate as a function of top-n.

Reads all results_top{n}_*.json files in the given directory, extracts the
success rate from each file's summary, and plots a line graph ordered by top-n.

Left y-axis  : Attack Success Rate (ASR %)
Right y-axis : Average number of strategies used per task
               (with early_stop, a task that succeeds on the 1st strategy
               counts as 1; a task that exhausts all strategies counts as top-n)

Usage:
    python comparison_topn.py --result-dir results/topn
    python comparison_topn.py --result-dir results/topn --output-dir results/graphs
"""

import argparse
import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


OUTPUT_DIR = "results/graphs"


def find_topn_files(result_dir: str) -> list[tuple[int, str]]:
    """
    Scan result_dir for files matching results_top{n}_*.json.

    Returns:
        List of (top_n, filepath) tuples, sorted by top_n ascending.
    """
    pattern = re.compile(r"results_top(\d+)_.*\.json$")
    entries = []

    for fname in os.listdir(result_dir):
        m = pattern.match(fname)
        if m:
            top_n = int(m.group(1))
            entries.append((top_n, os.path.join(result_dir, fname)))

    # If multiple files share the same top_n, keep the most recently modified one
    by_n: dict[int, tuple[str, float]] = {}
    for top_n, fpath in entries:
        mtime = os.path.getmtime(fpath)
        if top_n not in by_n or mtime > by_n[top_n][1]:
            by_n[top_n] = (fpath, mtime)

    return sorted((n, info[0]) for n, info in by_n.items())


def load_summary(filepath: str) -> dict:
    """
    Load success stats from a result JSON file.

    Supports both:
      - New format: {"summary": {...}, "tasks": [...]}
      - Old format: plain list of tasks (computes stats manually)

    Returns:
        {"total_tasks": int, "successful_tasks": int, "success_rate": float,
         "avg_strategies_used": float, "target_llm": str, "judge_llm": str}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "summary" in data:
        s = data["summary"]
        tasks = data.get("tasks", [])
        target_llm = s.get("models", {}).get("target_llm", "unknown")
        judge_llm = s.get("models", {}).get("judge_llm", "unknown")
        total = s.get("total_tasks", 0)
        successful = s.get("successful_tasks", 0)
        success_rate = s.get("success_rate", 0.0)
    else:
        # Old format: plain list of tasks
        tasks = data if isinstance(data, list) else data.get("tasks", [])
        total = len(tasks)
        successful = sum(1 for t in tasks if t.get("success", False))
        success_rate = successful / total if total > 0 else 0.0
        target_llm = "unknown"
        judge_llm = "unknown"

    # Average number of strategies actually tried per task
    # (len of "categories" dict = strategies tried before early stop or exhaustion)
    strategy_counts = [len(t.get("categories", {})) for t in tasks]
    avg_strategies = sum(strategy_counts) / len(strategy_counts) if strategy_counts else 0.0

    return {
        "total_tasks": total,
        "successful_tasks": successful,
        "success_rate": success_rate,
        "avg_strategies_used": avg_strategies,
        "target_llm": target_llm,
        "judge_llm": judge_llm,
    }


def plot_topn_comparison(
    entries: list[tuple[int, dict]],
    result_dir: str,
    output_dir: str,
) -> str:
    """
    Draw a dual-axis line graph:
      Left  y-axis (blue)  : Attack Success Rate (ASR %)
      Right y-axis (orange): Average number of strategies used per task

    Args:
        entries: List of (top_n, summary_dict) sorted by top_n.
        result_dir: Used for the plot title.
        output_dir: Directory to save the output PNG.

    Returns:
        Path to the saved PNG file.
    """
    top_ns = [e[0] for e in entries]
    rates = [e[1]["success_rate"] * 100 for e in entries]
    successfuls = [e[1]["successful_tasks"] for e in entries]
    totals = [e[1]["total_tasks"] for e in entries]
    avg_strats = [e[1]["avg_strategies_used"] for e in entries]

    target_llm = entries[0][1]["target_llm"] if entries else "unknown"
    judge_llm = entries[0][1]["judge_llm"] if entries else "unknown"

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"graph_topn_{timestamp}.png")

    COLOR_ASR = "#2563EB"    # blue
    COLOR_AVG = "#EA580C"    # orange

    fig, ax = plt.subplots(figsize=(max(8, len(top_ns) * 0.9), 6))
    ax2 = ax.twinx()  # right y-axis

    x = list(range(len(top_ns)))

    # --- Left axis: ASR ---
    ax.plot(x, rates, marker="o", linewidth=2, markersize=8,
            color=COLOR_ASR, zorder=3, label="ASR (%)")

    for i, rate in enumerate(rates):
        ax.annotate(
            f"{rate:.1f}%",
            xy=(i, rate),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1e3a5f",
        )

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, min(max(rates) * 1.35 + 5, 105))
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=12, color=COLOR_ASR)
    ax.tick_params(axis="y", labelcolor=COLOR_ASR)

    # --- Right axis: avg strategies used ---
    ax2.plot(x, avg_strats, marker="s", linewidth=2, markersize=7,
             color=COLOR_AVG, linestyle="--", zorder=2, label="Avg strategies used")

    for i, avg in enumerate(avg_strats):
        ax2.annotate(
            f"{avg:.2f}",
            xy=(i, avg),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#7c2d12",
        )

    ax2.set_ylim(1, 4)
    ax2.set_ylabel("Avg. Strategies Used per Task", fontsize=12, color=COLOR_AVG)
    ax2.tick_params(axis="y", labelcolor=COLOR_AVG)

    # --- Shared x-axis ---
    ax.set_xticks(x)
    ax.set_xticklabels([f"top-{n}" for n in top_ns], rotation=30, ha="right", fontsize=10)
    ax.set_xlabel("Number of Strategies (top-n)", fontsize=12)

    ax.set_title(
        "ASR & Avg. Strategies Used by Top-N",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attack success rate vs top-n strategy count."
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory containing results_top{n}_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to save the graph (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.result_dir):
        print(f"[ERROR] Directory not found: {args.result_dir}")
        return

    output_dir = args.output_dir

    # Discover files
    topn_files = find_topn_files(args.result_dir)
    if not topn_files:
        print(f"[ERROR] No results_top{{n}}_*.json files found in: {args.result_dir}")
        return

    print(f"Found {len(topn_files)} result file(s) in: {args.result_dir}")

    # Load summaries
    entries: list[tuple[int, dict]] = []
    print(f"\n{'top-n':>6}  {'success':>8}  {'total':>6}  {'rate':>7}  {'avg strats':>10}  file")
    print("-" * 80)
    for top_n, fpath in topn_files:
        summary = load_summary(fpath)
        entries.append((top_n, summary))
        print(
            f"top-{top_n:>2}  "
            f"{summary['successful_tasks']:>8}  "
            f"{summary['total_tasks']:>6}  "
            f"{summary['success_rate']*100:>6.1f}%  "
            f"{summary['avg_strategies_used']:>10.2f}  "
            f"{os.path.basename(fpath)}"
        )

    # Plot
    output_path = plot_topn_comparison(entries, args.result_dir, output_dir)
    print(f"\nGraph saved to: {output_path}")


if __name__ == "__main__":
    main()

