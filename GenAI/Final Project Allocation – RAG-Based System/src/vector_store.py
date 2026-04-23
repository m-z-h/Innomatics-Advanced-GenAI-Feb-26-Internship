"""
Vector Store Module
Handles storage and retrieval of embeddings in ChromaDB
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from src.chunking import Chunk
from src.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result"""
    chunk: Chunk
    similarity_score: float
    rank: int
    
    def __repr__(self):
        return f"SearchResult(rank={self.rank}, score={self.similarity_score:.3f}, text_len={len(self.chunk.text)})"


@dataclass
class RetrievalResult:
    """Results from a retrieval operation"""
    query: str
    results: List[SearchResult]
    total_results: int
    retrieval_time_ms: float
    confidence: float
    status: str  # "success", "no_results", "error"
    
    def __repr__(self):
        return f"RetrievalResult(total={self.total_results}, confidence={self.confidence:.3f}, status={self.status})"


class VectorStore:
    """Base class for vector stores"""
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        raise NotImplementedError
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        raise NotImplementedError


class ChromaDBStore(VectorStore):
    """ChromaDB Vector Store Implementation"""
    
    def __init__(self, embedding_provider: EmbeddingProvider, persist_dir: str, 
                 collection_name: str = "rag_knowledge_base"):
        """
        Initialize ChromaDB store
        
        Args:
            embedding_provider: EmbeddingProvider instance
            persist_dir: Directory to persist ChromaDB
            collection_name: Name of collection
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb package required. Install with: pip install chromadb")
        
        self.embedding_provider = embedding_provider
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Store chunks for reference
        self.chunks_store: Dict[str, Chunk] = {}
        
        logger.info(f"Initialized ChromaDB store at {persist_dir}")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to vector store
        
        Args:
            chunks: List of Chunk objects to add
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to ChromaDB")
        
        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_provider.embed_batch(texts)
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        
        # Prepare metadata
        metadatas = []
        for chunk in chunks:
            metadata = {
                "source_file": chunk.source_file,
                "page_number": str(chunk.page_number),
                "chunk_size": str(len(chunk.text)),
                **{f"meta_{k}": str(v) for k, v in chunk.metadata.items()}
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        
        # Store chunks locally for reference
        for chunk in chunks:
            self.chunks_store[chunk.chunk_id] = chunk
        
        logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of SearchResults
        """
        # Convert numpy array to list for ChromaDB
        query_embedding_list = query_embedding.tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=k
        )
        
        search_results = []
        
        # Process results
        if results["ids"] and results["ids"][0]:
            for rank, (chunk_id, distance, metadata, text) in enumerate(
                zip(results["ids"][0], results["distances"][0], 
                    results["metadatas"][0], results["documents"][0])
            ):
                # Convert distance to similarity (cosine distance to similarity)
                # Cosine distance ranges from 0 to 2, convert to [0, 1]
                similarity = 1 - (distance / 2)
                
                # Retrieve chunk from store
                chunk = self.chunks_store.get(chunk_id)
                
                if not chunk:
                    # Reconstruct chunk if not in store
                    chunk = Chunk(
                        text=text,
                        chunk_id=chunk_id,
                        source_file=metadata.get("source_file", "unknown"),
                        page_number=int(metadata.get("page_number", 0)),
                        metadata=metadata
                    )
                
                result = SearchResult(
                    chunk=chunk,
                    similarity_score=similarity,
                    rank=rank + 1
                )
                search_results.append(result)
        
        return search_results
    
    def delete_chunk(self, chunk_id: str) -> None:
        """Delete a chunk from the store"""
        self.collection.delete(ids=[chunk_id])
        if chunk_id in self.chunks_store:
            del self.chunks_store[chunk_id]
        logger.info(f"Deleted chunk {chunk_id}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_dir": self.persist_dir
        }


class InMemoryStore(VectorStore):
    """Simple in-memory vector store for testing"""
    
    def __init__(self, embedding_provider: EmbeddingProvider):
        """Initialize in-memory store"""
        self.embedding_provider = embedding_provider
        self.chunks: List[Chunk] = []
        self.embeddings: List[np.ndarray] = []
        
        logger.info("Initialized in-memory vector store")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to store"""
        self.chunks.extend(chunks)
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_provider.embed_batch(texts)
        self.embeddings.extend(embeddings)
        
        logger.info(f"Added {len(chunks)} chunks to in-memory store (total: {len(self.chunks)})")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Search for similar chunks using cosine similarity"""
        if not self.chunks:
            return []
        
        # Compute cosine similarities
        similarities = []
        for emb in self.embeddings:
            # Cosine similarity
            dot_product = np.dot(query_embedding, emb)
            norm_q = np.linalg.norm(query_embedding)
            norm_e = np.linalg.norm(emb)
            
            if norm_q == 0 or norm_e == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_q * norm_e)
            
            similarities.append(similarity)
        
        # Get top-k
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(SearchResult(
                chunk=self.chunks[idx],
                similarity_score=float(similarities[idx]),
                rank=rank + 1
            ))
        
        return results
