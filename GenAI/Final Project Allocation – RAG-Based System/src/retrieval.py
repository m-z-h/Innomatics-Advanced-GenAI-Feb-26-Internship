"""
Retrieval Module
Retrieves relevant chunks from vector store for user queries
"""

import logging
import time
from typing import Optional
import numpy as np
from src.embeddings import EmbeddingProvider
from src.vector_store import VectorStore, RetrievalResult, SearchResult

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant chunks for queries"""
    
    def __init__(self, vector_store: VectorStore, embedding_provider: EmbeddingProvider):
        """
        Initialize retriever
        
        Args:
            vector_store: VectorStore instance
            embedding_provider: EmbeddingProvider instance
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        
        logger.info("Initialized Retriever")
    
    def retrieve(self, query: str, top_k: int = 5, 
                 threshold: float = 0.6) -> RetrievalResult:
        """
        Retrieve relevant chunks for query
        
        Args:
            query: User query string
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            RetrievalResult
        """
        start_time = time.time()
        
        try:
            # Embed query
            query_embedding = self.embedding_provider.embed(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k=top_k)
            
            # Filter by threshold
            filtered_results = [r for r in results if r.similarity_score >= threshold]
            
            # Calculate confidence as average similarity
            if filtered_results:
                confidence = np.mean([r.similarity_score for r in filtered_results])
            else:
                confidence = 0.0
            
            # Calculate retrieval time
            retrieval_time_ms = (time.time() - start_time) * 1000
            
            status = "success" if filtered_results else "no_results"
            
            result = RetrievalResult(
                query=query,
                results=filtered_results,
                total_results=len(filtered_results),
                retrieval_time_ms=retrieval_time_ms,
                confidence=confidence,
                status=status
            )
            
            logger.info(f"Retrieved {len(filtered_results)} chunks for query (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            
            return RetrievalResult(
                query=query,
                results=[],
                total_results=0,
                retrieval_time_ms=(time.time() - start_time) * 1000,
                confidence=0.0,
                status="error"
            )
    
    def retrieve_with_reranking(self, query: str, top_k: int = 5, 
                                threshold: float = 0.6) -> RetrievalResult:
        """
        Retrieve with optional reranking (future: add cross-encoder)
        
        Args:
            query: User query
            top_k: Final number of results
            threshold: Minimum threshold
            
        Returns:
            RetrievalResult
        """
        # For now, same as regular retrieve
        # Future: Add cross-encoder reranking here
        return self.retrieve(query, top_k=top_k, threshold=threshold)
    
    def retrieve_by_metadata(self, query: str, filters: dict, 
                            top_k: int = 5) -> RetrievalResult:
        """
        Retrieve with metadata filtering
        
        Args:
            query: User query
            filters: Metadata filters (e.g., {"source_file": "faq.pdf"})
            top_k: Number of results
            
        Returns:
            RetrievalResult
        """
        # Retrieve all results first
        result = self.retrieve(query, top_k=top_k * 2, threshold=0.0)
        
        # Apply filters
        filtered_results = []
        for search_result in result.results:
            match = True
            for key, value in filters.items():
                if search_result.chunk.metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered_results.append(search_result)
            
            if len(filtered_results) >= top_k:
                break
        
        # Recalculate confidence
        if filtered_results:
            confidence = np.mean([r.similarity_score for r in filtered_results])
        else:
            confidence = 0.0
        
        return RetrievalResult(
            query=query,
            results=filtered_results,
            total_results=len(filtered_results),
            retrieval_time_ms=result.retrieval_time_ms,
            confidence=confidence,
            status="success" if filtered_results else "no_results"
        )
