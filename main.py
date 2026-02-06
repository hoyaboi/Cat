"""
Main entry point for CAT attack pipeline.
"""
from src.utils import run_attack_pipeline

# Default categories
CATEGORIES = ["Education", "Entertainment", "Health", "Business"]


# def main():
#     run_attack_pipeline(
#         categories=CATEGORIES,
#         limit=2
#     )

def main():
    run_attack_pipeline(
        categories=CATEGORIES,
        target_model="llama-3-8b",
        limit=2
    )


if __name__ == "__main__":
    main()
