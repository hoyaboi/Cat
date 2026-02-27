"""
Pre-generate word substitution dictionaries for all tasks × strategies.
Run this script before main.py to share dictionaries across all target models.

Usage:
    python gen_dictionary.py
    python gen_dictionary.py --limit 10
    python gen_dictionary.py --noun 80 --verb 40 --adjective 40 --adverb 20
    python gen_dictionary.py --word-model gpt-4o-mini --limit 5
"""
import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

from src.utils.attack import load_harmful_queries
from src.utils import STRATEGIES, DICTIONARY_DIR
from src.llm.generator import generate_dictionary
from src.utils.logger import log, close_log_file
from model import ModelFactory


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def run_gen_dictionary(
    word_model: Optional[str] = None,
    limit: Optional[int] = None,
    word_counts: Optional[dict] = None,
    output_dir: str = DICTIONARY_DIR,
    workers: int = 1,
) -> None:
    """
    Pre-generate all dictionaries for tasks × strategies.

    Args:
        word_model: Model name for Word LLM
        limit: Limit number of tasks to process
        word_counts: Dict of {pos: count}, e.g. {"noun": 80, "verb": 40, ...}
        output_dir: Directory to save dictionaries
        workers: Number of parallel workers for strategy-level generation (default: 1)
    """
    timestamp = datetime.now().strftime("%H%M%S")
    log_file = f"results/logs/gen_dictionary_{timestamp}.log"

    log("=" * 100, log_file=log_file)
    log("Starting Dictionary Generation", log_file=log_file)
    log("=" * 100, log_file=log_file)
    log(f"Strategies: {STRATEGIES}", log_file=log_file)
    log(f"Output directory: {output_dir}", log_file=log_file)
    log(f"Workers (strategy parallelism): {workers}", log_file=log_file)
    if limit:
        log(f"Limit: {limit} tasks", log_file=log_file)
    if word_counts:
        log(f"Word counts per POS: {word_counts}", log_file=log_file)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load queries
    try:
        queries = load_harmful_queries("data/harmful_behaviors.csv", limit=limit)
        log(f"Loaded {len(queries)} harmful queries", log_file=log_file)
    except Exception as e:
        log(f"Failed to load harmful queries: {e}", "ERROR", log_file=log_file)
        close_log_file()
        raise

    # Create Word LLM client
    log(f"Word LLM model: {word_model or 'gpt-4o-mini'}", log_file=log_file)
    try:
        word_llm = ModelFactory.create_word_llm(word_model)
        log("✓ Word LLM client created", log_file=log_file)
    except Exception as e:
        log(f"Failed to create Word LLM client: {e}", "ERROR", log_file=log_file)
        close_log_file()
        raise

    from src.llm.generator import _extract_keywords_from_query, _generate_word_list
    from src.word.dictionary import Dictionary

    total = len(queries) * len(STRATEGIES)
    done = 0
    skipped = 0
    counter_lock = threading.Lock()
    start_time = time.time()

    def _generate_strategy(
        task_num: int,
        harmful_query: str,
        strategy: str,
        harmful_words: dict,
        effective_counts: dict,
    ) -> tuple:
        """Generate dictionary for a single strategy. Returns (strategy, success, error_msg)."""
        dict_path = os.path.join(output_dir, f"task{task_num}", f"task{task_num}_{strategy.lower()}.csv")
        log(f"[Task {task_num}][{strategy}] Generating dictionary → {dict_path}", log_file=log_file)
        try:
            generate_dictionary(
                harmful_query=harmful_query,
                target_category=strategy,
                word_llm_client=word_llm,
                output_dir=output_dir,
                task_num=task_num,
                harmful_words=harmful_words,
                word_counts=effective_counts,
            )
            log(f"[Task {task_num}][{strategy}] ✓ Saved", log_file=log_file)
            return strategy, True, None
        except Exception as e:
            log(f"[Task {task_num}][{strategy}] ✗ Error: {e}", "ERROR", log_file=log_file)
            return strategy, False, str(e)

    try:
        for query_data in queries:
            task_num = query_data["task"]
            harmful_query = query_data["original_query"]

            task_start = time.time()

            log("", log_file=log_file)
            log("=" * 100, log_file=log_file)
            log(f"Task {task_num}/{len(queries)}: {harmful_query}", log_file=log_file)
            log("=" * 100, log_file=log_file)

            # Pre-generate harmful words once per task (shared across strategies, sequential)
            key_words = _extract_keywords_from_query(harmful_query)
            all_keywords = set()
            for kw in key_words.values():
                all_keywords.update(kw)
            harmful_strategy = (
                ", ".join(sorted(list(all_keywords))[:20])
                if all_keywords
                else "general technical terms"
            )

            effective_counts = word_counts or dict(Dictionary.EXPECTED_COUNTS)

            try:
                harmful_words = _generate_word_list(
                    context="",
                    word_llm_client=word_llm,
                    task_num=task_num,
                    output_dir=output_dir,
                    list_type="harmful",
                    key_words=key_words,
                    strategy=harmful_strategy,
                    word_counts=effective_counts,
                )
                log(f"[Task {task_num}] ✓ Harmful words generated", log_file=log_file)
            except Exception as e:
                log(f"[Task {task_num}] ✗ Failed to generate harmful words: {e}", "ERROR", log_file=log_file)
                with counter_lock:
                    skipped += len(STRATEGIES)
                    done += len(STRATEGIES)
                continue

            # Generate strategy dictionaries (parallel if workers > 1)
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            _generate_strategy,
                            task_num, harmful_query, strategy, harmful_words, effective_counts
                        ): strategy
                        for strategy in STRATEGIES
                    }
                    for future in as_completed(futures):
                        _, success, _ = future.result()
                        with counter_lock:
                            if not success:
                                skipped += 1
                            done += 1
            else:
                for strategy in STRATEGIES:
                    _, success, _ = _generate_strategy(
                        task_num, harmful_query, strategy, harmful_words, effective_counts
                    )
                    if not success:
                        skipped += 1
                    done += 1

            task_elapsed = time.time() - task_start
            log(f"[Task {task_num}] Completed in {_format_time(task_elapsed)} ({done}/{total} done)", log_file=log_file)

        log("", log_file=log_file)
        log("=" * 100, log_file=log_file)
        log("Dictionary Generation Complete", log_file=log_file)
        log("=" * 100, log_file=log_file)
        total_elapsed = time.time() - start_time
        log(f"Total: {done}/{total} processed, {skipped} failed", log_file=log_file)
        log(f"Time elapsed: {_format_time(total_elapsed)}", log_file=log_file)
        log(f"Dictionaries saved to: {output_dir}", log_file=log_file)

    finally:
        close_log_file()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate word substitution dictionaries for all tasks × strategies."
    )

    parser.add_argument(
        "--word-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for Word LLM (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to process (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DICTIONARY_DIR,
        help=f"Output directory for dictionaries (default: {DICTIONARY_DIR})",
    )

    parser.add_argument(
        "--noun",
        type=int,
        default=None,
        help="Number of noun mappings per dictionary (default: 80)",
    )

    parser.add_argument(
        "--verb",
        type=int,
        default=None,
        help="Number of verb mappings per dictionary (default: 40)",
    )

    parser.add_argument(
        "--adjective",
        type=int,
        default=None,
        help="Number of adjective mappings per dictionary (default: 40)",
    )

    parser.add_argument(
        "--adverb",
        type=int,
        default=None,
        help="Number of adverb mappings per dictionary (default: 20)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for strategy-level dictionary generation (default: 1)"
    )

    args = parser.parse_args()

    # Build word_counts only if any POS arg was specified
    from src.word.dictionary import Dictionary
    default_counts = dict(Dictionary.EXPECTED_COUNTS)

    word_counts = None
    if any(v is not None for v in [args.noun, args.verb, args.adjective, args.adverb]):
        word_counts = {
            "noun": args.noun if args.noun is not None else default_counts["noun"],
            "verb": args.verb if args.verb is not None else default_counts["verb"],
            "adjective": args.adjective if args.adjective is not None else default_counts["adjective"],
            "adverb": args.adverb if args.adverb is not None else default_counts["adverb"],
        }

    run_gen_dictionary(
        word_model=args.word_model,
        limit=args.limit,
        word_counts=word_counts,
        output_dir=args.output_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
