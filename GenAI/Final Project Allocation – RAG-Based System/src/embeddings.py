"""
Embeddings Module
Generates embeddings for text using various providers
"""

import logging
import hashlib
import re
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
import os
import config

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
        
        # Fail fast and let the local fallback take over when the remote API is
        # unavailable or over quota.
        self.client = OpenAI(api_key=self.api_key, max_retries=0)
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


class HashEmbedding(EmbeddingProvider):
    """Offline-safe embedding provider using hashed token features."""

    def __init__(self, dimension: int = config.EMBEDDING_DIMENSION):
        self.dimension = dimension
        logger.info(f"Initialized hash embedding provider with dimension: {dimension}")

    def _embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())

        if not tokens:
            return vector

        for token in tokens:
            token_bytes = token.encode("utf-8")
            index = int(hashlib.md5(token_bytes).hexdigest(), 16) % self.dimension
            sign = 1.0 if int(hashlib.sha1(token_bytes).hexdigest(), 16) % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def embed(self, text: str) -> np.ndarray:
        return self._embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self._embed_text(text) for text in texts]


class FallbackEmbeddingProvider(EmbeddingProvider):
    """Uses a primary provider and falls back to a local hash provider on failure."""

    def __init__(self, primary: EmbeddingProvider, fallback: EmbeddingProvider):
        self.primary = primary
        self.fallback = fallback
        self.dimension = getattr(primary, "dimension", getattr(fallback, "dimension", config.EMBEDDING_DIMENSION))

    def embed(self, text: str) -> np.ndarray:
        try:
            return self.primary.embed(text)
        except Exception as exc:
            logger.warning(f"Primary embedding provider failed, using fallback: {exc}")
            return self.fallback.embed(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        try:
            return self.primary.embed_batch(texts)
        except Exception as exc:
            logger.warning(f"Primary embedding provider failed for batch, using fallback: {exc}")
            return self.fallback.embed_batch(texts)


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
        try:
            primary = OpenAIEmbedding(**kwargs)
            return FallbackEmbeddingProvider(primary=primary, fallback=HashEmbedding(dimension=primary.dimension))
        except Exception as exc:
            logger.warning(f"OpenAI embedding provider unavailable, using hash fallback: {exc}")
            return HashEmbedding()
    elif provider == "local":
        try:
            return LocalEmbedding(**kwargs)
        except Exception as exc:
            logger.warning(f"Local embedding provider unavailable, using hash fallback: {exc}")
            return HashEmbedding()
    elif provider == "hash":
        return HashEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
