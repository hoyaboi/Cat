"""
Main entry point for CAT attack pipeline.

If the dictionary directory does not exist or is empty, dictionaries are
automatically generated before the attack pipeline begins.
Alternatively, run gen_dictionary.py manually to pre-generate them.
"""
import argparse
import os

from src.utils import run_attack_pipeline, STRATEGIES, DICTIONARY_DIR


def _dictionary_dir_is_ready(dictionary_dir: str) -> bool:
    """Return True if dictionary_dir exists and contains at least one task subdirectory."""
    if not os.path.isdir(dictionary_dir):
        return False
    # Check for any task subdirectory (e.g. task1/, task2/, ...)
    for entry in os.scandir(dictionary_dir):
        if entry.is_dir():
            return True
    return False


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description=(
            "CAT Attack Pipeline. "
            "Dictionaries are auto-generated if the dictionary directory is missing."
        )
    )

    # ── Attack arguments ──────────────────────────────────────────────────────
    parser.add_argument(
        "--target-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for Target LLM (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for Judge LLM (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)"
    )

    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        default=False,
        help="Disable early stop — run all strategies even after a successful attack (default: early stop enabled)"
    )

    parser.add_argument(
        "--dictionary-dir",
        type=str,
        default=DICTIONARY_DIR,
        help=f"Directory containing pre-generated dictionaries (default: {DICTIONARY_DIR})"
    )

    # ── Dictionary generation arguments (used only when auto-generation triggers) ─
    parser.add_argument(
        "--word-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for Word LLM used during dictionary generation (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for strategy-level dictionary generation (default: 1)"
    )

    parser.add_argument(
        "--noun",
        type=int,
        default=None,
        help="Number of noun mappings per dictionary (default: model default)"
    )

    parser.add_argument(
        "--verb",
        type=int,
        default=None,
        help="Number of verb mappings per dictionary (default: model default)"
    )

    parser.add_argument(
        "--adjective",
        type=int,
        default=None,
        help="Number of adjective mappings per dictionary (default: model default)"
    )

    parser.add_argument(
        "--adverb",
        type=int,
        default=None,
        help="Number of adverb mappings per dictionary (default: model default)"
    )

    args = parser.parse_args()

    # ── Auto-generate dictionaries if needed ──────────────────────────────────
    if not _dictionary_dir_is_ready(args.dictionary_dir):
        print(f"[INFO] Dictionary directory not found or empty: '{args.dictionary_dir}'")
        print("[INFO] Auto-generating dictionaries before attack pipeline...")

        from gen_dictionary import run_gen_dictionary
        from src.word.dictionary import Dictionary

        default_counts = dict(Dictionary.EXPECTED_COUNTS)
        word_counts = None
        if any(v is not None for v in [args.noun, args.verb, args.adjective, args.adverb]):
            word_counts = {
                "noun":      args.noun      if args.noun      is not None else default_counts["noun"],
                "verb":      args.verb      if args.verb      is not None else default_counts["verb"],
                "adjective": args.adjective if args.adjective is not None else default_counts["adjective"],
                "adverb":    args.adverb    if args.adverb    is not None else default_counts["adverb"],
            }

        run_gen_dictionary(
            word_model=args.word_model,
            limit=args.limit,
            word_counts=word_counts,
            output_dir=args.dictionary_dir,
            workers=args.workers,
        )
        print("[INFO] Dictionary generation complete. Starting attack pipeline...\n")
    else:
        print(f"[INFO] Using existing dictionaries from '{args.dictionary_dir}'")

    # ── Run attack pipeline ───────────────────────────────────────────────────
    run_attack_pipeline(
        strategies=STRATEGIES,
        target_model=args.target_model,
        judge_model=args.judge_model,
        output_file=args.output_file,
        limit=args.limit,
        early_stop=not args.no_early_stop,
        dictionary_dir=args.dictionary_dir,
    )


if __name__ == "__main__":
    main()
