"""
LangGraph Workflow Engine
Orchestrates RAG system with state-based graph execution
"""

import logging
from typing import TypedDict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

from src.query_processor import ProcessedQuery
from src.retrieval import RetrievalResult
from src.hitl import EscalationData, EscalationManager
from src.llm_client import LLMClient
import config

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State object for workflow"""
    query_id: str
    user_id: str
    query: str
    processed_query: Optional[ProcessedQuery]
    retrieval_result: Optional[RetrievalResult]
    llm_response: Optional[str]
    confidence_score: float
    should_escalate: bool
    escalation_reason: Optional[str]
    escalation_id: Optional[str]
    final_response: str
    status: str  # "input", "retrieval", "decision", "generation", "escalation", "output"
    metadata: dict


@dataclass
class WorkflowResult:
    """Result from workflow execution"""
    query_id: str
    response: str
    is_escalated: bool
    escalation_id: Optional[str]
    confidence: float
    sources: list
    execution_time_ms: float
    status: str


class RAGWorkflowEngine:
    """LangGraph-based RAG workflow orchestrator"""
    
    def __init__(self, query_processor, retriever, llm_client: LLMClient, 
                 escalation_manager: EscalationManager):
        """
        Initialize workflow engine
        
        Args:
            query_processor: QueryProcessor instance
            retriever: Retriever instance
            llm_client: LLMClient instance
            escalation_manager: EscalationManager instance
        """
        self.query_processor = query_processor
        self.retriever = retriever
        self.llm_client = llm_client
        self.escalation_manager = escalation_manager
        
        logger.info("Initialized RAGWorkflowEngine")
    
    def execute(self, query: str, user_id: str = "default") -> WorkflowResult:
        """
        Execute workflow for user query
        
        Args:
            query: User query
            user_id: User ID
            
        Returns:
            WorkflowResult
        """
        import time
        start_time = time.time()
        
        # Initialize state
        state = self._init_state(query, user_id)
        
        try:
            # Node 1: Input Processing
            state = self._input_node(state)
            
            # Node 2: Retrieval
            state = self._retrieval_node(state)
            
            # Node 3: Decision
            state = self._decision_node(state)
            
            # Node 4: Conditional branching
            if state["should_escalate"]:
                state = self._escalation_node(state)
            else:
                state = self._generation_node(state)
            
            # Node 5: Output
            state = self._output_node(state)
            
            # Build result
            execution_time_ms = (time.time() - start_time) * 1000
            
            result = WorkflowResult(
                query_id=state["query_id"],
                response=state["final_response"],
                is_escalated=state["should_escalate"],
                escalation_id=state["escalation_id"],
                confidence=state["confidence_score"],
                sources=[
                    {
                        "file": r.chunk.source_file,
                        "page": r.chunk.page_number,
                        "score": r.similarity_score
                    }
                    for r in state["retrieval_result"].results
                ] if state["retrieval_result"] else [],
                execution_time_ms=execution_time_ms,
                status=state["status"]
            )
            
            logger.info(f"Workflow completed: {result.status}, escalated={result.is_escalated}")
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return WorkflowResult(
                query_id=state["query_id"],
                response=f"An error occurred: {str(e)}",
                is_escalated=True,
                escalation_id=None,
                confidence=0.0,
                sources=[],
                execution_time_ms=execution_time_ms,
                status="error"
            )
    
    def _init_state(self, query: str, user_id: str) -> WorkflowState:
        """Initialize workflow state"""
        return WorkflowState(
            query_id=f"qry_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            query=query,
            processed_query=None,
            retrieval_result=None,
            llm_response=None,
            confidence_score=0.0,
            should_escalate=False,
            escalation_reason=None,
            escalation_id=None,
            final_response="",
            status="input",
            metadata={}
        )
    
    def _input_node(self, state: WorkflowState) -> WorkflowState:
        """Input node: Process and validate query"""
        logger.info(f"INPUT_NODE: Processing query: {state['query']}")
        
        processed_query = self.query_processor.process_query(state["query"])
        
        state["processed_query"] = processed_query
        state["status"] = "retrieval"
        state["confidence_score"] = processed_query.intent_confidence
        
        return state
    
    def _retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """Retrieval node: Get relevant chunks"""
        logger.info("RETRIEVAL_NODE: Retrieving chunks")
        
        processed_query = state["processed_query"]
        
        # Retrieve chunks
        retrieval_result = self.retriever.retrieve(
            processed_query.cleaned_query,
            top_k=config.RETRIEVAL_TOP_K,
            threshold=config.RETRIEVAL_SCORE_THRESHOLD
        )
        
        state["retrieval_result"] = retrieval_result
        state["confidence_score"] = retrieval_result.confidence
        state["status"] = "decision"
        
        return state
    
    def _decision_node(self, state: WorkflowState) -> WorkflowState:
        """Decision node: Decide whether to generate or escalate"""
        logger.info("DECISION_NODE: Making routing decision")
        
        confidence = state["confidence_score"]
        num_results = len(state["retrieval_result"].results)
        
        # Decision logic
        if confidence < config.ESCALATION_THRESHOLD:
            state["should_escalate"] = True
            state["escalation_reason"] = "low_confidence"
            logger.info(f"Escalating due to low confidence: {confidence:.3f}")
        
        elif num_results == 0:
            state["should_escalate"] = True
            state["escalation_reason"] = "no_relevant_chunks"
            logger.info("Escalating due to no relevant chunks")
        
        elif state["processed_query"].requires_escalation_check:
            state["should_escalate"] = True
            state["escalation_reason"] = "requires_review"
            logger.info("Escalating for complex query")
        
        else:
            state["should_escalate"] = False
            state["escalation_reason"] = None
        
        return state
    
    def _generation_node(self, state: WorkflowState) -> WorkflowState:
        """Generation node: Generate response using LLM"""
        logger.info("GENERATION_NODE: Generating response")
        
        try:
            llm_response = self.llm_client.generate(
                state["processed_query"].llm_context,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS
            )
            
            state["llm_response"] = llm_response
            state["status"] = "output"
            
            logger.info("Successfully generated LLM response")
            
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            
            # Fallback: escalate if LLM fails
            state["should_escalate"] = True
            state["escalation_reason"] = "llm_error"
            state["status"] = "escalation"
        
        return state
    
    def _escalation_node(self, state: WorkflowState) -> WorkflowState:
        """Escalation node: Prepare escalation"""
        logger.info(f"ESCALATION_NODE: Escalating ({state['escalation_reason']})")
        
        # Create escalation data
        escalation_data = EscalationData(
            original_query=state["query"],
            retrieved_chunks=state["retrieval_result"].results if state["retrieval_result"] else [],
            ai_response=state["llm_response"] or "Unable to generate response",
            confidence_score=state["confidence_score"],
            escalation_reason=state["escalation_reason"],
            user_id=state["user_id"]
        )
        
        # Create escalation
        escalation_id = self.escalation_manager.create_escalation(escalation_data)
        
        state["escalation_id"] = escalation_id
        state["status"] = "escalation"
        
        return state
    
    def _output_node(self, state: WorkflowState) -> WorkflowState:
        """Output node: Format final response"""
        logger.info("OUTPUT_NODE: Formatting response")
        
        if state["should_escalate"]:
            response = f"Your query is being escalated to our support team. Reference ID: {state['escalation_id']}"
        else:
            response = state["llm_response"]
            
            # Add citations if enabled
            if config.INCLUDE_SOURCE_CITATIONS and state["retrieval_result"].results:
                citations = "\n\nSources:\n"
                for i, result in enumerate(state["retrieval_result"].results, 1):
                    citations += f"- {result.chunk.source_file} (Page {result.chunk.page_number})\n"
                response += citations
            
            # Add confidence if enabled
            if config.INCLUDE_CONFIDENCE_SCORE:
                response += f"\n[Confidence: {state['confidence_score']:.1%}]"
        
        state["final_response"] = response
        state["status"] = "output"
        
        return state
