"""
Target LLM attacker for sending converted queries and receiving responses.
"""
import yaml
from pathlib import Path
from typing import Optional

from model import ModelFactory, LLMClient
from src.word import Dictionary, convert_query


def attack_target_llm(
    harmful_query: str,
    dictionary: Dictionary,
    target_llm_client: Optional[LLMClient] = None,
) -> str:
    """
    Attack Target LLM with converted query.
    
    Args:
        harmful_query: Original harmful query
        dictionary: Word substitution dictionary
        target_llm_client: Target LLM client (None = use default)
    
    Returns:
        Target LLM response (in Alternative words)
    """
    # Load system prompt
    prompt_path = Path("prompts/target_llm_system.yaml")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_config = yaml.safe_load(f)
        system_prompt_template = prompt_config['prompt']
    
    # Convert dictionary to CSV string
    dictionary_csv = dictionary.to_csv_string()
    
    # Insert dictionary into system prompt
    system_prompt = system_prompt_template.replace('{dictionary}', dictionary_csv)
    
    # Convert harmful query to alternative words
    converted_query = convert_query(harmful_query, dictionary)
    
    # Get or create Target LLM client
    if target_llm_client is None:
        target_llm_client = ModelFactory.create_target_llm()
    
    # Call Target LLM
    response = target_llm_client.call(
        system_prompt=system_prompt,
        user_prompt=converted_query,
        temperature=0.7,
    )
    
    return response
