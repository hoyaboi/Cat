"""
Main entry point for CAT attack pipeline.
"""
import argparse
from src.utils import run_attack_pipeline

# Default strategies
STRATEGIES = ["Education", "Entertainment", "Health", "Business", "Technology"]


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description="CAT Attack Pipeline")
    
    parser.add_argument(
        "--word-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for Word LLM (default: gpt-4o-mini)"
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

    
    args = parser.parse_args()
    
    run_attack_pipeline(
        strategies=STRATEGIES,
        word_model=args.word_model,
        target_model=args.target_model,
        judge_model=args.judge_model,
        output_file=args.output_file,
        limit=args.limit,
        early_stop=not args.no_early_stop
    )


if __name__ == "__main__":
    main()
