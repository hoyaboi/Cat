#!/usr/bin/env python3
"""
Convert CAT results to scoring format.
Extracts task, query, and response from results files.
"""
import argparse
import json
import os
from pathlib import Path


def convert_results(input_file: str, output_file: str):
    """
    Convert CAT results file to scoring format.
    
    Args:
        input_file: Path to input results JSON file
        output_file: Path to output JSON file
    """
    # Load input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract tasks
    tasks = data.get('tasks', [])
    
    # Convert to scoring format
    scoring_results = []
    
    for task_data in tasks:
        task_num = task_data.get('task')
        original_query = task_data.get('original_query')
        categories = task_data.get('categories', {})
        
        # Select category with highest score
        selected_category = None
        selected_response = None
        highest_score = -1.0
        
        for category_name, category_data in categories.items():
            score = category_data.get('score', 0.0)
            if score is not None and score > highest_score:
                highest_score = score
                selected_category = category_name
                selected_response = category_data.get('reversed_response')
        
        # Only include if we have a response
        if selected_response:
            scoring_results.append({
                "task": task_num,
                "query": original_query,
                "response": selected_response
            })
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scoring_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Converted {len(scoring_results)} tasks from {input_file}")
    print(f"  → Saved to {output_file}")


def main():
    """Main function to convert result files."""
    parser = argparse.ArgumentParser(
        description="Convert CAT results to scoring format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_results.py --input results/results_llama_3_8b_111654.json --output results_llama_3_8b.json
  python convert_results.py --input results/results_gemini_2.5_flash_111641.json --output results_gemini_2.5_flash.json
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input results JSON file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output filename (will be saved to ~/research/sun/scoring/results/cat/results/)'
    )
    
    args = parser.parse_args()
    
    # Resolve input file path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        # If relative path, resolve from current working directory
        input_path = Path.cwd() / input_path
    
    if not input_path.exists():
        print(f"✗ Error: Input file not found: {input_path}")
        return 1
    
    # Output directory (always the same)
    output_base = Path.home() / "research" / "sun" / "scoring" / "results" / "cat" / "results"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Output file path
    output_path = output_base / args.output
    
    # Convert file
    try:
        convert_results(str(input_path), str(output_path))
        print("\n✓ Conversion complete!")
        return 0
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
