"""
Model configuration and definitions for various LLM providers.
"""
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv
from .clients import LLMClient, OpenAIClient, HuggingFaceClient, GeminiClient

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    model_type: str
    deployment_name: str
    api_key_env: str
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    api_key_optional: bool = False  # If True, api_key_env is not required (e.g. local vLLM server)
    
    def to_client(self) -> LLMClient:
        """
        Create an LLM client instance from this configuration.
        
        Returns:
            LLMClient instance
        """
        api_key = os.getenv(self.api_key_env)
        
        if not api_key and self.model_type != "huggingface" and not self.api_key_optional:
            raise ValueError(f"API key not found: {self.api_key_env}")
        
        # Use a placeholder for local servers that don't require authentication
        if not api_key and self.api_key_optional:
            api_key = "not-required"
        
        if self.model_type == "openai":
            endpoint = self.endpoint if self.endpoint else "https://api.openai.com/v1"
            return OpenAIClient(
                model_name=self.deployment_name,
                api_key=api_key,
                endpoint=endpoint,
            )
        
        elif self.model_type == "huggingface":
            return HuggingFaceClient(
                model_id=self.deployment_name,
                hf_access_token=api_key,
                use_cuda=True,
                trust_remote_code=True,
                max_new_tokens=256,
                device_map="auto",
                torch_dtype="float16" if self._check_cuda() else None,
            )
        
        elif self.model_type == "gemini":
            return GeminiClient(
                model_name=self.deployment_name,
                api_key=api_key,
            )
        
        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                f"Please implement custom model support in ModelConfig.to_client() method."
            )
    
    @staticmethod
    def _check_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


# Available model presets
AVAILABLE_MODELS = {
    # OpenAI Models
    "gpt-4": ModelConfig(
        name="GPT-4",
        model_type="openai",
        deployment_name="gpt-4",
        api_key_env="OPENAI_API_KEY"
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        model_type="openai",
        deployment_name="gpt-4o",
        api_key_env="OPENAI_API_KEY"
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o-mini",
        model_type="openai",
        deployment_name="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY"
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5-Turbo",
        model_type="openai",
        deployment_name="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY"
    ),
    
    # Llama Models (HuggingFace)
    "llama-2-7b": ModelConfig(
        name="LLaMA-2-7b",
        model_type="huggingface",
        deployment_name="meta-llama/Llama-2-7b-chat-hf",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    "llama-2-70b": ModelConfig(
        name="LLaMA-2-70b",
        model_type="huggingface",
        deployment_name="meta-llama/Llama-2-70b-chat-hf",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    "llama-3-8b": ModelConfig(
        name="LLaMA-3-8B",
        model_type="huggingface",
        deployment_name="meta-llama/Meta-Llama-3-8B-Instruct",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    "llama-3-70b": ModelConfig(
        name="LLaMA-3-70b",
        model_type="huggingface",
        deployment_name="meta-llama/Meta-Llama-3-70B-Instruct",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    
    # Qwen Models (HuggingFace)
    "qwen2.5-7b": ModelConfig(
        name="Qwen2.5-7B",
        model_type="huggingface",
        deployment_name="Qwen/Qwen2.5-7B-Instruct",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    "qwen2.5-14b": ModelConfig(
        name="Qwen2.5-14B",
        model_type="huggingface",
        deployment_name="Qwen/Qwen2.5-14B-Instruct",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    "qwen2.5-32b": ModelConfig(
        name="Qwen2.5-32B",
        model_type="huggingface",
        deployment_name="Qwen/Qwen2.5-32B-Instruct",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    "qwen2.5-72b": ModelConfig(
        name="Qwen2.5-72B",
        model_type="huggingface",
        deployment_name="Qwen/Qwen2.5-72B-Instruct",
        api_key_env="HUGGINGFACE_TOKEN"
    ),
    
    # Gemini Models (Google)
    "gemini-2.5-flash": ModelConfig(
        name="Gemini 2.5 Flash",
        model_type="gemini",
        deployment_name="gemini-2.5-flash",
        api_key_env="GOOGLE_API_KEY"
    ),

    # vLLM local servers (OpenAI-compatible API, no API key required)
    # llama:  vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000 --dtype float16
    # qwen:   vllm serve Qwen/Qwen2.5-7B-Instruct            --port 8001 --dtype float16
    "llama-3-8b-vllm": ModelConfig(
        name="LLaMA-3-8B (vLLM)",
        model_type="openai",
        deployment_name="meta-llama/Meta-Llama-3-8B-Instruct",
        api_key_env="VLLM_API_KEY",
        endpoint="http://localhost:8000/v1",
        api_key_optional=True
    ),
    "qwen2.5-7b-vllm": ModelConfig(
        name="Qwen2.5-7B (vLLM)",
        model_type="openai",
        deployment_name="Qwen/Qwen2.5-7B-Instruct",
        api_key_env="VLLM_API_KEY",
        endpoint="http://localhost:8001/v1",
        api_key_optional=True
    ),
}
