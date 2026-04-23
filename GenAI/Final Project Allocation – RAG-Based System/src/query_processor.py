"""
Query Processing Module
Processes user queries (cleaning, intent detection, etc.)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional
from src.retrieval import Retriever, RetrievalResult
import config

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Result of intent detection"""
    intent: str
    confidence: float
    category: str


@dataclass
class ProcessedQuery:
    """Result of query processing"""
    original_query: str
    cleaned_query: str
    intent: str
    intent_confidence: float
    retrieved_chunks: any  # List[Chunk]
    retrieval_scores: list
    avg_retrieval_confidence: float
    llm_context: str
    requires_escalation_check: bool


class QueryProcessor:
    """Processes user queries"""
    
    def __init__(self, retriever: Retriever):
        """
        Initialize query processor
        
        Args:
            retriever: Retriever instance
        """
        self.retriever = retriever
        self.intent_categories = config.INTENT_CATEGORIES
        
        logger.info("Initialized QueryProcessor")
    
    def process_query(self, user_query: str) -> ProcessedQuery:
        """
        Process user query end-to-end
        
        Args:
            user_query: Raw user query
            
        Returns:
            ProcessedQuery
        """
        # Clean query
        cleaned_query = self._clean_query(user_query)
        logger.info(f"Cleaned query: {user_query} -> {cleaned_query}")
        
        # Detect intent
        intent_result = self.detect_intent(cleaned_query)
        logger.info(f"Detected intent: {intent_result.intent} (confidence: {intent_result.confidence})")
        
        # Retrieve chunks
        retrieval_result = self.retriever.retrieve(
            cleaned_query,
            top_k=config.RETRIEVAL_TOP_K,
            threshold=config.RETRIEVAL_SCORE_THRESHOLD
        )
        
        # Prepare LLM context
        llm_context = self._prepare_llm_context(cleaned_query, retrieval_result)
        
        # Determine if escalation check needed
        requires_escalation_check = (
            retrieval_result.confidence < config.ESCALATION_THRESHOLD or
            intent_result.confidence < 0.5 or
            len(retrieval_result.results) < 2
        )
        
        return ProcessedQuery(
            original_query=user_query,
            cleaned_query=cleaned_query,
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            retrieved_chunks=retrieval_result.results,
            retrieval_scores=[r.similarity_score for r in retrieval_result.results],
            avg_retrieval_confidence=retrieval_result.confidence,
            llm_context=llm_context,
            requires_escalation_check=requires_escalation_check
        )
    
    def detect_intent(self, query: str) -> IntentResult:
        """
        Detect intent from query
        
        Args:
            query: User query
            
        Returns:
            IntentResult
        """
        query_lower = query.lower()
        
        # Simple keyword-based intent detection
        best_intent = "general"
        best_confidence = 0.0
        
        for intent, keywords in self.intent_categories.items():
            if not keywords:  # Skip general category for matching
                continue
            
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            
            if matches > 0:
                # Confidence based on number of matches
                confidence = min(0.5 + (matches * 0.25), 0.95)
                
                if confidence > best_confidence:
                    best_intent = intent
                    best_confidence = confidence
        
        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            category=best_intent
        )
    
    def _clean_query(self, query: str) -> str:
        """
        Clean user query
        
        Args:
            query: Raw query
            
        Returns:
            Cleaned query
        """
        # Remove extra whitespace
        cleaned = " ".join(query.split())
        
        # Convert to lowercase for processing
        cleaned = cleaned.lower()
        
        # Remove trailing punctuation
        cleaned = cleaned.rstrip("?!.,")
        
        return cleaned
    
    def _prepare_llm_context(self, query: str, retrieval_result: RetrievalResult) -> str:
        """
        Prepare context for LLM
        
        Args:
            query: User query
            retrieval_result: Results from retrieval
            
        Returns:
            Formatted prompt context
        """
        # Build context from retrieved chunks
        context_parts = []
        
        if retrieval_result.results:
            context_parts.append("Context from knowledge base:")
            context_parts.append("=" * 50)
            
            for i, search_result in enumerate(retrieval_result.results, 1):
                chunk = search_result.chunk
                citation = f"[Source: {chunk.source_file}, Page: {chunk.page_number}]"
                context_parts.append(f"\n[{i}] {chunk.text}\n{citation}")
        else:
            context_parts.append("No relevant context found in knowledge base.")
        
        # Build prompt
        prompt_parts = [
            "You are a helpful customer support assistant.",
            "Answer the following question based ONLY on the provided context.",
            "If you don't have enough information, say 'I don't have information about this.'",
            "Be concise (1-2 sentences) and cite your sources.",
            "",
            "\n".join(context_parts),
            "",
            f"Question: {query}",
            "Answer:"
        ]
        
        return "\n".join(prompt_parts)
