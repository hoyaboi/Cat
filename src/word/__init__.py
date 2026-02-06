"""
Word substitution dictionary and query conversion modules.
"""
from src.word.dictionary import Dictionary
from src.word.converter import convert_query, reverse_convert

__all__ = [
    'Dictionary',
    'convert_query',
    'reverse_convert',
]
