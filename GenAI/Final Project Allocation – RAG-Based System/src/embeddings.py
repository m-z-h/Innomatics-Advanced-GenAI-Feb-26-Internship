"""
Embeddings Module
Generates embeddings for text using various providers
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
import os

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts"""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI Embedding Provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding provider
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: Embedding model to use
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.dimension = 1536 if "3-small" in model else 3072
        
        logger.info(f"Initialized OpenAI embedding provider with model: {model}")
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts"""
        # OpenAI API can handle multiple texts in one call
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        
        # Sort by index to maintain order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [np.array(e.embedding) for e in embeddings]


class LocalEmbedding(EmbeddingProvider):
    """Local Embedding Provider using Sentence Transformers"""
    
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize local embedding provider
        
        Args:
            model: Hugging Face model ID
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package required. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Initialized local embedding provider with model: {model}")
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return list(embeddings)


def create_embedding_provider(provider: str = "openai", **kwargs) -> EmbeddingProvider:
    """
    Factory function to create embedding provider
    
    Args:
        provider: "openai" or "local"
        **kwargs: Additional arguments for provider
        
    Returns:
        EmbeddingProvider instance
    """
    if provider == "openai":
        return OpenAIEmbedding(**kwargs)
    elif provider == "local":
        return LocalEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
