"""
Main entry point for CAT attack pipeline.
Requires dictionaries to be pre-generated via gen_dictionary.py.
"""
import argparse
from src.utils import run_attack_pipeline
from src.utils.attack import DICTIONARY_DIR

# Default strategies
STRATEGIES = ["Education", "Entertainment", "Health", "Business", "Technology"]


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="CAT Attack Pipeline. Run gen_dictionary.py first to pre-generate dictionaries."
    )

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

    args = parser.parse_args()

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
