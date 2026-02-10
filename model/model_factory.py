"""
Factory for creating LLM clients for different roles (Word, Target, Judge).
"""
from typing import Optional, Dict
from .model_config import LLMClient, ModelConfig, AVAILABLE_MODELS


class ModelFactory:
    """Factory class for creating LLM clients."""
    
    # Cache for model clients: model_name -> LLMClient instance
    _client_cache: Dict[str, LLMClient] = {}
    
    @staticmethod
    def create_client(model_name: str) -> LLMClient:
        """
        Create an LLM client instance based on model name.
        Caches clients so the same model is only loaded once.
        
        Args:
            model_name: Model name (e.g., "gpt-4", "llama-3-8b")
            
        Returns:
            LLMClient instance (cached if already created)
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        
        # Return cached client if it exists
        if model_name in ModelFactory._client_cache:
            return ModelFactory._client_cache[model_name]
        
        # Create new client and cache it
        config = AVAILABLE_MODELS[model_name]
        client = config.to_client()
        ModelFactory._client_cache[model_name] = client
        
        return client
    
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
