"""
LLM Client Module
Interfaces with LLM APIs (OpenAI, local, etc.)
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI
import config

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI LLM Client"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        logger.info(f"Initialized OpenAI client with model: {model}")
    
    def generate(self, prompt: str, temperature: float = 0.3, 
                 max_tokens: int = 500, **kwargs) -> str:
        """
        Generate response using OpenAI API
        
        Args:
            prompt: Formatted prompt
            temperature: Sampling temperature
            max_tokens: Max response tokens
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful customer support assistant. Be concise and cite sources."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=config.LLM_TIMEOUT
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise


class LocalLLMClient(LLMClient):
    """Local LLM Client (using transformers)"""
    
    def __init__(self, model: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize local LLM client
        
        Args:
            model: HuggingFace model ID
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers package required")
        
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto"
        )
        
        logger.info(f"Initialized local LLM client with model: {model}")
    
    def generate(self, prompt: str, temperature: float = 0.3, 
                 max_tokens: int = 500, **kwargs) -> str:
        """
        Generate response using local LLM
        
        Args:
            prompt: Formatted prompt
            temperature: Sampling temperature
            max_tokens: Max response tokens
            
        Returns:
            Generated response text
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            outputs = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=0.95
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error with local LLM: {str(e)}")
            raise


def create_llm_client(provider: str = "openai", **kwargs) -> LLMClient:
    """
    Factory function to create LLM client
    
    Args:
        provider: "openai" or "local"
        **kwargs: Additional arguments
        
    Returns:
        LLMClient instance
    """
    if provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "local":
        return LocalLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
