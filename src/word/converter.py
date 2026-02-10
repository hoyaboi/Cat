"""
Word conversion utilities for transforming queries and responses.
"""
import re
from typing import Optional, List, Tuple
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tag import pos_tag

from .dictionary import Dictionary

# Download required NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except:
        nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def _get_wordnet_pos(treebank_tag: str) -> Optional[str]:
    """
    Convert treebank POS tag to WordNet POS tag.
    
    Args:
        treebank_tag: Treebank POS tag
    
    Returns:
        WordNet POS tag or None
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def _lemmatize_word(word: str, pos_tag: Optional[str] = None) -> str:
    """
    Lemmatize a word to its base form.
    
    Args:
        word: Word to lemmatize
        pos_tag: POS tag (optional, for better accuracy)
    
    Returns:
        Lemmatized word
    """
    word_lower = word.lower()
    
    if pos_tag:
        wn_pos = _get_wordnet_pos(pos_tag)
        if wn_pos:
            return lemmatizer.lemmatize(word_lower, wn_pos)
    
    return lemmatizer.lemmatize(word_lower)


def _inflect_word(base_word: str, original_word: str, pos_tag: Optional[str] = None) -> str:
    """
    Apply the same inflection pattern from original word to base word.
    
    Args:
        base_word: Base form of the word (from dictionary)
        original_word: Original word with inflection
        pos_tag: POS tag of original word
    
    Returns:
        Inflected word matching original_word's pattern
    """
    original_lower = original_word.lower()
    base_lower = base_word.lower()
    
    # If words are same length or original is shorter, return base as-is
    if len(original_lower) <= len(base_lower):
        return base_word
    
    # Simple heuristic: preserve suffix pattern
    # For verbs: -ed, -ing, -s endings
    if pos_tag and pos_tag.startswith('V'):
        if original_lower.endswith('ed'):
            return base_lower + 'ed' if not base_lower.endswith('e') else base_lower[:-1] + 'ed'
        elif original_lower.endswith('ing'):
            if base_lower.endswith('e'):
                return base_lower[:-1] + 'ing'
            else:
                return base_lower + 'ing'
        elif original_lower.endswith('s') and not original_lower.endswith('ss'):
            return base_lower + 's' if not base_lower.endswith('s') else base_lower
    
    # For nouns: plural forms (-s, -es, -ies)
    if pos_tag and pos_tag.startswith('N'):
        if original_lower.endswith('ies'):
            return base_lower[:-1] + 'ies' if base_lower.endswith('y') else base_lower + 'ies'
        elif original_lower.endswith('es'):
            return base_lower + 'es'
        elif original_lower.endswith('s') and not base_lower.endswith('s'):
            return base_lower + 's'
    
    # Default: return base word
    return base_word


def _tokenize_with_positions(text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    """
    Tokenize text and return tokens with their positions and POS tags.
    
    Args:
        text: Input text
    
    Returns:
        List of tuples: (word, start_pos, end_pos, pos_tag)
    """
    tokens = []
    words = nltk.word_tokenize(text)
    pos_tags = pos_tag(words)
    
    current_pos = 0
    for word, tag in pos_tags:
        # Find word position in original text (case-insensitive)
        word_lower = word.lower()
        start = text.lower().find(word_lower, current_pos)
        if start == -1:
            # If not found, use approximate position
            start = current_pos
        end = start + len(word)
        current_pos = end
        
        tokens.append((word, start, end, tag))
    
    return tokens


def convert_query(harmful_query: str, dictionary: Dictionary) -> str:
    """
    Convert harmful query words to alternative words.
    
    Args:
        harmful_query: Original harmful query (e.g., "How to make a bomb")
        dictionary: Word substitution dictionary
    
    Returns:
        Converted query with alternative words (e.g., "When for draw the paper?")
    """
    if not harmful_query.strip():
        return harmful_query
    
    # Tokenize with POS tags
    tokens = _tokenize_with_positions(harmful_query)
    
    result_parts = []
    last_end = 0
    
    for word, start, end, pos_tag in tokens:
        # Preserve text before token
        if start > last_end:
            result_parts.append(harmful_query[last_end:start])
        
        # Try exact match first
        alternative = dictionary.get_alternative(word)
        
        # If not found, try lemmatization
        if not alternative:
            lemma = _lemmatize_word(word, pos_tag)
            base_alternative = dictionary.get_alternative(lemma)
            
            if base_alternative:
                # Apply same inflection pattern
                alternative = _inflect_word(base_alternative, word, pos_tag)
                # Preserve original case
                if word[0].isupper():
                    alternative = alternative.capitalize()
        
        # Use alternative if found, otherwise keep original
        result_parts.append(alternative if alternative else word)
        last_end = end
    
    # Add remaining text
    if last_end < len(harmful_query):
        result_parts.append(harmful_query[last_end:])
    
    return ''.join(result_parts)


def reverse_convert(response: str, dictionary: Dictionary) -> str:
    """
    Reverse convert response from alternative words back to original words.
    
    Args:
        response: Response with alternative words
        dictionary: Word substitution dictionary
    
    Returns:
        Response with original words restored
    """
    if not response.strip():
        return response
    
    # Tokenize with POS tags
    tokens = _tokenize_with_positions(response)
    
    result_parts = []
    last_end = 0
    
    for word, start, end, pos_tag in tokens:
        # Preserve text before token
        if start > last_end:
            result_parts.append(response[last_end:start])
        
        # Try exact match first (reverse lookup)
        original = dictionary.get_original(word)
        
        # If not found, try lemmatization
        if not original:
            lemma = _lemmatize_word(word, pos_tag)
            base_original = dictionary.get_original(lemma)
            
            if base_original:
                # Apply same inflection pattern
                original = _inflect_word(base_original, word, pos_tag)
                # Preserve original case
                if word[0].isupper():
                    original = original.capitalize()
        
        # Use original if found, otherwise keep alternative
        result_parts.append(original if original else word)
        last_end = end
    
    # Add remaining text
    if last_end < len(response):
        result_parts.append(response[last_end:])
    
    return ''.join(result_parts)
