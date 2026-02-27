"""
Word LLM generator for creating word substitution dictionaries.
"""
import yaml
import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Set
import nltk
from nltk.tag import pos_tag

from model import ModelFactory, LLMClient
from src.word import Dictionary

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except:
        nltk.download('averaged_perceptron_tagger', quiet=True)


WORD_PROMPT_PATH_HARMFUL = Path("prompts/word_llm_harmful_system.yaml")
WORD_PROMPT_PATH_BENIGN = Path("prompts/word_llm_benign_system.yaml")
MAX_ATTEMPTS = 5
VALID_CATEGORIES = {'noun', 'verb', 'adjective', 'adverb'}


def _extract_keywords_from_query(query: str) -> Dict[str, Set[str]]:
    """
    Extract key words from harmful query by POS category.
    
    Args:
        query: Harmful query string
    
    Returns:
        Dictionary mapping category to set of keywords
    """
    keywords = {
        'noun': set(),
        'verb': set(),
        'adjective': set(),
        'adverb': set()
    }
    
    # Tokenize and tag
    words = nltk.word_tokenize(query.lower())
    pos_tags = pos_tag(words)
    
    # Common stop words to exclude
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
                  'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                  'should', 'could', 'may', 'might', 'can', 'must', 'this', 'that', 
                  'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 
                  'how', 'what', 'when', 'where', 'why', 'who', 'which', 'give', 
                  'detailed', 'instructions', 'on', 'make', 'other'}
    
    # Map POS tags to categories
    for word, tag in pos_tags:
        # Remove punctuation
        word_clean = re.sub(r'[^\w]', '', word.lower())
        if not word_clean or word_clean in stop_words or len(word_clean) < 2:
            continue
        
        # Map POS tags to our categories
        if tag.startswith('NN'):  # Noun
            keywords['noun'].add(word_clean)
        elif tag.startswith('VB'):  # Verb
            keywords['verb'].add(word_clean)
        elif tag.startswith('JJ'):  # Adjective
            keywords['adjective'].add(word_clean)
        elif tag.startswith('RB'):  # Adverb
            keywords['adverb'].add(word_clean)
    
    return keywords


def generate_dictionary(
    harmful_query: str,
    target_category: str,
    word_llm_client: Optional[LLMClient] = None,
    output_dir: str = "results/dictionaries",
    task_num: Optional[int] = None,
    harmful_words: Optional[Dict[str, List[str]]] = None,
    word_counts: Optional[Dict[str, int]] = None,
) -> Dictionary:
    """
    Generate word substitution dictionary using Word LLM.
    
    Args:
        harmful_query: The harmful query string (used only for keyword extraction, not passed to LLM)
        target_category: Target benign category/strategy (e.g., "Education", "Health")
        word_llm_client: LLM client for word generation
        output_dir: Directory to save dictionary files
        task_num: Task number for file naming
        harmful_words: Optional pre-generated harmful words dict. If provided, 
                      will reuse instead of generating new ones.
        word_counts: Optional dict of {pos: count} overriding Dictionary.EXPECTED_COUNTS.
                     e.g. {"noun": 80, "verb": 40, "adjective": 40, "adverb": 20}
    
    Returns:
        Dictionary instance with word mappings
    """
    if word_llm_client is None:
        word_llm_client = ModelFactory.create_word_llm()
    
    effective_counts = word_counts or dict(Dictionary.EXPECTED_COUNTS)

    # Generate harmful words only if not provided
    if harmful_words is None:
        # Extract key words from harmful query (these become the strategy)
        key_words = _extract_keywords_from_query(harmful_query)
        
        # Create strategy string from all extracted keywords
        all_keywords = set()
        for category_keywords in key_words.values():
            all_keywords.update(category_keywords)
        harmful_strategy = ", ".join(sorted(list(all_keywords))[:20]) if all_keywords else "general technical terms"
        
        harmful_words = _generate_word_list(
            context="",  # Empty context - strategy is in system prompt
            word_llm_client=word_llm_client,
            task_num=task_num,
            output_dir=output_dir,
            list_type="harmful",
            key_words=key_words,
            strategy=harmful_strategy,
            word_counts=effective_counts,
        )
    
    # Always generate benign words for the target category (strategy)
    benign_words = _generate_word_list(
        context="",  # Empty context - strategy is in system prompt
        word_llm_client=word_llm_client,
        task_num=task_num,
        output_dir=output_dir,
        list_type="benign",
        strategy=target_category,
        word_counts=effective_counts,
    )
    
    all_mappings = _match_word_lists(harmful_words, benign_words, effective_counts)
    
    dictionary = Dictionary(all_mappings)
    _validate_and_save_dictionary(dictionary, target_category, output_dir, task_num)
    
    return dictionary


def _generate_word_list(
    context: str,
    word_llm_client: LLMClient,
    task_num: Optional[int],
    output_dir: str,
    list_type: str,
    key_words: Optional[Dict[str, Set[str]]] = None,
    strategy: str = "",
    word_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, List[str]]:
    """Generate a list of words (harmful or benign) organized by category."""
    all_words: Dict[str, List[str]] = {}
    effective_counts = word_counts or dict(Dictionary.EXPECTED_COUNTS)

    for category, expected_count in effective_counts.items():
        category_keywords = key_words.get(category, set()) if key_words else set()
        category_words = _generate_category_words_with_retry(
            context=context,
            category=category,
            expected_count=expected_count,
            word_llm_client=word_llm_client,
            task_num=task_num,
            output_dir=output_dir,
            list_type=list_type,
            key_words=category_keywords,
            strategy=strategy
        )
        all_words[category] = category_words
    
    return all_words


def _generate_category_words_with_retry(
    context: str,
    category: str,
    expected_count: int,
    word_llm_client: LLMClient,
    task_num: Optional[int],
    output_dir: str,
    list_type: str,
    key_words: Optional[Set[str]] = None,
    strategy: str = ""
) -> List[str]:
    """Generate words for a category by generating 2x expected_count words at a time until we have enough."""
    category_words = []
    seen_words = set()
    attempt = 0
    
    # First, add key words at the beginning to ensure they're included in the mapping
    if key_words and list_type == "harmful":
        for key_word in sorted(key_words):  # Sort for consistency
            key_word_lower = key_word.lower()
            if key_word_lower not in seen_words:
                seen_words.add(key_word_lower)
                category_words.append(key_word)
                if len(category_words) >= expected_count:
                    return category_words
    
    while len(category_words) < expected_count and attempt < MAX_ATTEMPTS:
        attempt += 1
        
        # Calculate how many words to generate: 2x the expected count
        words_to_generate = expected_count * 2
        
        response = _call_llm_for_words(
            context=context,
            category=category,
            existing_words=set(category_words),
            word_llm_client=word_llm_client,
            task_num=task_num,
            output_dir=output_dir,
            list_type=list_type,
            key_words=key_words,
            word_count=words_to_generate,
            strategy=strategy
        )
        
        parsed_words = _parse_word_list(response, category)
        
        for word in parsed_words:
            if len(category_words) >= expected_count:
                break
            word_lower = word.lower()
            if word_lower not in seen_words:
                seen_words.add(word_lower)
                category_words.append(word)
        
        # If we have enough words, return early
        if len(category_words) >= expected_count:
            return category_words
    
    return category_words


def _call_llm_for_words(
    context: str,
    category: str,
    existing_words: set,
    word_llm_client: LLMClient,
    task_num: Optional[int],
    output_dir: str,
    list_type: str,
    key_words: Optional[Set[str]] = None,
    word_count: int = 100,
    strategy: str = ""
) -> str:
    """Call LLM to generate words for a category."""
    system_prompt = _load_and_prepare_prompt(category, list_type, word_count, strategy)
    user_prompt = _create_user_prompt(context, category, existing_words, list_type, key_words)
    
    response = word_llm_client.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=800,
    )
    
    return response


def _load_and_prepare_prompt(category: str, list_type: str, word_count: int = 100, strategy: str = "") -> str:
    """Load prompt template and replace placeholders."""
    # Select prompt file based on list_type
    if list_type == "harmful":
        prompt_path = WORD_PROMPT_PATH_HARMFUL
    else:
        prompt_path = WORD_PROMPT_PATH_BENIGN
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_config = yaml.safe_load(f)
        template = prompt_config['prompt']
    
    # Replace placeholders
    template = template.replace("[CATEGORY]", category)
    template = template.replace("[WORD_COUNT]", str(word_count))
    template = template.replace("[STRATEGY]", strategy)
    
    return template


def _create_user_prompt(
    context: str,
    category: str,
    existing_words: set,
    list_type: str,
    key_words: Optional[Set[str]] = None
) -> str:
    """Create user prompt based on list type."""
    existing_list = ", ".join(list(existing_words)[:20]) if existing_words else "None"
    
    if list_type == "harmful":
        # For harmful words: DO NOT include original query, only key words
        prompt_parts = []
        
        # Add key words if available (these are the strategy keywords)
        if key_words and len(key_words) > 0:
            key_words_list = ", ".join(sorted(list(key_words))[:10])
            prompt_parts.append(f"Key words to include: {key_words_list}")
        else:
            # If no key words, provide minimal instruction
            prompt_parts.append("Generate words related to the strategy provided in the system prompt.")
        
        # Add existing words if any
        if existing_words:
            prompt_parts.append(f"Already generated: {existing_list}")
        
        return "\n".join(prompt_parts) if prompt_parts else "Generate words now."
    else:
        # For benign words: strategy is already in system prompt, minimal user prompt
        prompt_parts = []
        
        # Add existing words if any
        if existing_words:
            prompt_parts.append(f"Already generated: {existing_list}")
        
        return "\n".join(prompt_parts) if prompt_parts else "Generate words now."


def _parse_word_list(response: str, category: str) -> List[str]:
    """Parse word list from LLM response."""
    words = []
    seen_words = set()
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    
    for line in lines:
        word = _extract_word_from_line(line)
        if word:
            word_lower = word.lower()
            if word_lower not in seen_words:
                seen_words.add(word_lower)
                words.append(word)
    
    return words


def _extract_word_from_line(line: str) -> Optional[str]:
    """Extract a single word from a line."""
    if not line or ' ' in line:
        return None
    
    line_lower = line.lower().strip()
    if line_lower.startswith(('category', 'word', 'output', 'format', 'example', 'requirements')):
        return None
    
    if ':' in line:
        return None
    
    word = line.strip().rstrip('.,;:!?')
    
    if not word or not word.replace('-', '').replace('_', '').isalnum():
        return None
    
    invalid_chars = ['(', ')', '[', ']', '{', '}', '=', '+', '*', '/', '\\']
    if any(char in word for char in invalid_chars):
        return None
    
    return word


def _match_word_lists(
    harmful_words: Dict[str, List[str]],
    benign_words: Dict[str, List[str]],
    word_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, str]]:
    """Match harmful and benign word lists 1:1 by category."""
    all_mappings: Dict[str, Dict[str, str]] = {}
    effective_counts = word_counts or dict(Dictionary.EXPECTED_COUNTS)

    for category in effective_counts.keys():
        harmful_list = harmful_words.get(category, [])
        benign_list = benign_words.get(category, [])
        
        category_mappings = {}
        min_count = min(len(harmful_list), len(benign_list))
        
        for i in range(min_count):
            category_mappings[harmful_list[i]] = benign_list[i]
        
        all_mappings[category] = category_mappings
    
    return all_mappings


def _validate_and_save_dictionary(
    dictionary: Dictionary,
    target_category: str,
    output_dir: str,
    task_num: Optional[int]
) -> None:
    """Validate dictionary and save to CSV file."""
    dictionary.validate()

    if task_num:
        save_dir = os.path.join(output_dir, f"task{task_num}")
        filename = f"task{task_num}_{target_category.lower()}.csv"
    else:
        save_dir = output_dir
        filename = f"{target_category.lower()}.csv"

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    
    dictionary.save_to_csv(file_path)


def _save_debug_response(
    response: str,
    context: str,
    category: str,
    count_needed: int,
    existing_count: int,
    task_num: Optional[int],
    output_dir: str,
    list_type: str
) -> None:
    """Save debug response to file."""
    debug_dir = os.path.join(output_dir, "..", "debug_responses")
    os.makedirs(debug_dir, exist_ok=True)
    
    if task_num is not None:
        debug_file = os.path.join(debug_dir, f"task{task_num}_{list_type}_{category}_raw.txt")
    else:
        debug_file = os.path.join(debug_dir, f"{list_type}_{category}_raw.txt")
    
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {list_type.upper()} {category.upper()} GENERATION ===\n")
        f.write(f"Context: {context}\n")
        f.write(f"Needed: {count_needed}\n")
        f.write(f"Existing words: {existing_count}\n\n")
        f.write(response)
