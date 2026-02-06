"""
Factory for creating LLM clients for different roles (Word, Target, Judge).
"""
from typing import Optional
from .model_config import LLMClient, ModelConfig, AVAILABLE_MODELS


class ModelFactory:
    """Factory class for creating LLM clients."""
    
    @staticmethod
    def create_client(model_name: str) -> LLMClient:
        """
        Create an LLM client instance based on model name.
        
        Args:
            model_name: Model name (e.g., "gpt-4", "llama-3-8b")
            
        Returns:
            LLMClient instance
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        
        config = AVAILABLE_MODELS[model_name]
        return config.to_client()
    
    @staticmethod
    def create_word_llm(model_name: Optional[str] = None) -> LLMClient:
        """
        Create Word LLM client.
        
        Args:
            model_name: Model name (defaults to "gpt-4o-mini")
            
        Returns:
            Word LLM client instance
        """
        if model_name is None:
            # Default to GPT-4o-mini for Word LLM
            model_name = "gpt-4o-mini"
        
        return ModelFactory.create_client(model_name)
    
    @staticmethod
    def create_target_llm(model_name: Optional[str] = None) -> LLMClient:
        """
        Create Target LLM client.
        
        Args:
            model_name: Model name (defaults to "gpt-4o-mini")
            
        Returns:
            Target LLM client instance
        """
        if model_name is None:
            # Default to GPT-4o-mini for Target LLM
            model_name = "gpt-4o-mini"
        
        return ModelFactory.create_client(model_name)
    
    @staticmethod
    def create_judge_llm(model_name: Optional[str] = None) -> LLMClient:
        """
        Create Judge LLM client.
        
        Args:
            model_name: Model name (defaults to "gpt-4o-mini")
            
        Returns:
            Judge LLM client instance
        """
        if model_name is None:
            # Default to GPT-4o-mini for Judge LLM
            model_name = "gpt-4o-mini"
        
        return ModelFactory.create_client(model_name)
