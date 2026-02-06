"""
Word LLM generator for creating word substitution dictionaries.
"""
import yaml
import csv
import hashlib
import os
from pathlib import Path
from typing import Optional

from model import ModelFactory, LLMClient
from src.word.dictionary import Dictionary


def generate_dictionary(
    harmful_query: str,
    target_category: str,
    word_llm_client: Optional[LLMClient] = None,
    output_dir: str = "results/dictionaries"
) -> Dictionary:
    """
    Generate word substitution dictionary using Word LLM.
    
    Args:
        harmful_query: Harmful query to generate dictionary for
        target_category: Target category ("Education", "Entertainment", "Health", "Business")
        word_llm_client: Word LLM client (None = use default)
        output_dir: Directory to save dictionary CSV file
    
    Returns:
        Dictionary object
    
    Raises:
        ValueError: If dictionary generation or validation fails
    """
    # Load system prompt
    prompt_path = Path("prompts/word_llm_system.yaml")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_config = yaml.safe_load(f)
        system_prompt = prompt_config['prompt']
    
    # Create user prompt
    user_prompt = f"""# Harmful Query
{harmful_query}

# Target Category
{target_category}"""
    
    # Get or create Word LLM client
    if word_llm_client is None:
        word_llm_client = ModelFactory.create_word_llm()
    
    # Call Word LLM
    response = word_llm_client.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
    )
    
    # Parse CSV response
    dictionary = _parse_csv_response(response)
    
    # Validate dictionary
    is_valid, errors = dictionary.validate()
    if not is_valid:
        # Log warnings but continue
        print(f"Warning: Dictionary validation failed:")
        for error in errors:
            print(f"  - {error}")
    
    # Save dictionary to CSV file
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename: {category}_{query_hash}.csv
    query_hash = hashlib.md5(harmful_query.encode()).hexdigest()[:8]
    filename = f"{target_category.lower()}_{query_hash}.csv"
    file_path = os.path.join(output_dir, filename)
    
    dictionary.save_to_csv(file_path)
    
    return dictionary


def _parse_csv_response(response: str) -> Dictionary:
    """
    Parse CSV response from Word LLM into Dictionary object.
    
    Args:
        response: CSV string response from LLM
    
    Returns:
        Dictionary object
    
    Raises:
        ValueError: If CSV parsing fails
    """
    mappings = {}
    
    # Extract CSV content (handle markdown code blocks)
    csv_content = response.strip()
    
    if csv_content.startswith("```"):
        lines = csv_content.split('\n')
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        csv_content = '\n'.join(lines)
    
    # Parse CSV
    lines = csv_content.strip().split('\n')
    
    # Skip header if present
    if lines and lines[0].lower().startswith('category'):
        lines = lines[1:]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse CSV line
        try:
            # Handle CSV parsing (may contain commas in values)
            reader = csv.reader([line])
            row = next(reader)
            
            if len(row) < 3:
                continue
            
            category = row[0].lower().strip()
            original = row[1].strip()
            alternative = row[2].strip()
            
            if not category or not original or not alternative:
                continue
            
            if category not in mappings:
                mappings[category] = {}
            
            mappings[category][original] = alternative
            
        except Exception as e:
            # Skip malformed lines
            continue
    
    if not mappings:
        raise ValueError("Failed to parse dictionary from LLM response. No valid entries found.")
    
    return Dictionary(mappings)
