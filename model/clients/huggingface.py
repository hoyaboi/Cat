"""
HuggingFace model client for local inference.
"""
from typing import Optional
import os
from .base import LLMClient


class HuggingFaceClient(LLMClient):
    """HuggingFace model client for local inference."""
    
    def __init__(
        self,
        model_id: str,
        hf_access_token: Optional[str] = None,
        use_cuda: bool = True,
        trust_remote_code: bool = True,
        max_new_tokens: int = 256,
        device_map: str = "auto",
        torch_dtype: Optional[str] = None,
    ):
        """
        Initialize HuggingFace client.
        
        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-chat-hf")
            hf_access_token: HuggingFace access token (defaults to HUGGINGFACE_TOKEN env var)
            use_cuda: Whether to use CUDA if available
            trust_remote_code: Whether to trust remote code
            max_new_tokens: Maximum new tokens to generate
            device_map: Device mapping strategy
            torch_dtype: Torch data type (e.g., "float16")
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch packages are required. "
                "Install with: pip install transformers torch"
            )
        
        self.model_id = model_id
        self.hf_access_token = hf_access_token or os.getenv("HUGGINGFACE_TOKEN")
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.max_new_tokens = max_new_tokens
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=self.hf_access_token,
            trust_remote_code=trust_remote_code,
        )
        
        # Set pad_token if not set (required for some models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = None
        if torch_dtype == "float16" and self.use_cuda:
            dtype = torch.float16
        elif torch_dtype == "bfloat16" and self.use_cuda:
            dtype = torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=self.hf_access_token,
            trust_remote_code=trust_remote_code,
            device_map=device_map if self.use_cuda else None,
            torch_dtype=dtype,
        )
        
        if not self.use_cuda:
            self.model = self.model.to("cpu")
    
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call HuggingFace model."""
        import torch
        
        # Check if tokenizer has chat template (for Instruct models)
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use chat template for proper formatting
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Tokenize the formatted prompt
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        else:
            # Fallback to simple concatenation for models without chat template
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            formatted_prompt = full_prompt
        
        if self.use_cuda:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Store input length to extract only generated tokens
        input_length = inputs['input_ids'].shape[1]
        
        # Generate
        max_new_tokens = max_tokens or self.max_new_tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        # Extract only the generated tokens (skip input tokens)
        generated_tokens = outputs[0][input_length:]
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
