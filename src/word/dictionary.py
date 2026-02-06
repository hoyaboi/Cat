"""
Dictionary class for managing word substitution mappings.
"""
import csv
import os
from typing import Dict, Optional, Tuple, List
from pathlib import Path


class Dictionary:
    """Manages word substitution dictionary (Original → Alternative)."""
    
    # Expected word counts by category (Total: 170 per list - harmful and benign)
    EXPECTED_COUNTS = {
        "noun": 50,
        "verb": 50,
        "adjective": 30,
        "adverb": 30,
        "unit": 10,
    }
    
    def __init__(self, mappings: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize dictionary.
        
        Args:
            mappings: Dictionary in format {category: {original: alternative}}
        """
        self.mappings = mappings or {}
        # Create reverse mapping for fast lookup
        self._reverse_mappings = {}
        for category, words in self.mappings.items():
            self._reverse_mappings[category] = {
                alt: orig for orig, alt in words.items()
            }
    
    @classmethod
    def load_from_csv(cls, file_path: str) -> 'Dictionary':
        """
        Load dictionary from CSV file.
        
        Args:
            file_path: Path to CSV file (format: Category,Original,Alternative)
        
        Returns:
            Dictionary instance
        """
        mappings = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = row['Category'].lower().strip()
                original = row['Original'].strip()
                alternative = row['Alternative'].strip()
                
                if category not in mappings:
                    mappings[category] = {}
                
                mappings[category][original] = alternative
        
        return cls(mappings)
    
    def save_to_csv(self, file_path: str) -> None:
        """
        Save dictionary to CSV file.
        
        Args:
            file_path: Path to save CSV file
        """
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Original', 'Alternative'])
            
            for category in sorted(self.mappings.keys()):
                for original, alternative in sorted(self.mappings[category].items()):
                    writer.writerow([category, original, alternative])
    
    def get_alternative(self, original: str, category: Optional[str] = None) -> Optional[str]:
        """
        Get alternative word for original word.
        
        Args:
            original: Original word to look up
            category: Optional category to search in (None = search all)
        
        Returns:
            Alternative word if found, None otherwise
        """
        if category:
            return self.mappings.get(category, {}).get(original)
        
        # Search all categories
        for cat_words in self.mappings.values():
            if original in cat_words:
                return cat_words[original]
        
        return None
    
    def get_original(self, alternative: str, category: Optional[str] = None) -> Optional[str]:
        """
        Get original word for alternative word (reverse lookup).
        
        Args:
            alternative: Alternative word to look up
            category: Optional category to search in (None = search all)
        
        Returns:
            Original word if found, None otherwise
        """
        if category:
            return self._reverse_mappings.get(category, {}).get(alternative)
        
        # Search all categories
        for cat_words in self._reverse_mappings.values():
            if alternative in cat_words:
                return cat_words[alternative]
        
        return None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate dictionary structure and counts.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check category counts
        for category, expected_count in self.EXPECTED_COUNTS.items():
            actual_count = len(self.mappings.get(category, {}))
            if actual_count != expected_count:
                errors.append(
                    f"Category '{category}': expected {expected_count} words, "
                    f"got {actual_count}"
                )
        
        # Check for duplicates within each category only
        for category, words in self.mappings.items():
            category_originals = []
            category_alternatives = []
            
            for original, alternative in words.items():
                if original in category_originals:
                    errors.append(f"Duplicate original word '{original}' in category '{category}'")
                category_originals.append(original)
                
                if alternative in category_alternatives:
                    errors.append(f"Duplicate alternative word '{alternative}' in category '{category}'")
                category_alternatives.append(alternative)
        
        return len(errors) == 0, errors
    
    def to_csv_string(self) -> str:
        """
        Convert dictionary to CSV string format for prompt insertion.
        
        Returns:
            CSV string representation
        """
        lines = ["Category,Original,Alternative"]
        
        for category in sorted(self.mappings.keys()):
            for original, alternative in sorted(self.mappings[category].items()):
                lines.append(f"{category},{original},{alternative}")
        
        return "\n".join(lines)
    
    def get_all_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Get all mappings.
        
        Returns:
            Dictionary in format {category: {original: alternative}}
        """
        return self.mappings.copy()
