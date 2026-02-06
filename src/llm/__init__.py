"""
LLM modules for word generation and target attacks.
"""
from src.llm.generator import generate_dictionary
from src.llm.target import attack_target_llm

__all__ = [
    'generate_dictionary',
    'attack_target_llm',
]
