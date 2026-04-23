"""
FastAPI REST API for RAG System
"""

import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

import config
from src.graph_engine import RAGWorkflowEngine
from src.hitl import EscalationManager

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Customer Support Assistant",
    description="Retrieval-Augmented Generation system with HITL",
    version="1.0.0"
)

# Global state (will be initialized)
workflow_engine: Optional[RAGWorkflowEngine] = None
escalation_manager: Optional[EscalationManager] = None


# Request/Response models
class QueryRequest(BaseModel):
    """User query request"""
    query: str
    user_id: Optional[str] = "default"
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response"""
    query_id: str
    response: str
    is_escalated: bool
    escalation_id: Optional[str] = None
    confidence: float
    sources: List[dict]
    execution_time_ms: float


class EscalationReview(BaseModel):
    """Escalation for human review"""
    escalation_id: str
    original_query: str
    ai_response: str
    confidence_score: float
    escalation_reason: str
    created_at: str


class EscalationResolution(BaseModel):
    """Resolution from support agent"""
    human_response: str
    agent_name: str
    feedback_rating: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global workflow_engine, escalation_manager
    
    logger.info("Initializing RAG system...")
    
    try:
        from src.embeddings import create_embedding_provider
        from src.vector_store import ChromaDBStore
        from src.retrieval import Retriever
        from src.query_processor import QueryProcessor
        from src.llm_client import create_llm_client
        
        # Create components
        embedding_provider = create_embedding_provider(
            provider=config.EMBEDDING_PROVIDER
        )
        
        vector_store = ChromaDBStore(
            embedding_provider=embedding_provider,
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_COLLECTION_NAME
        )
        
        retriever = Retriever(vector_store, embedding_provider)
        query_processor = QueryProcessor(retriever)
        
        llm_client = create_llm_client(provider="openai")
        escalation_manager = EscalationManager()
        
        # Create workflow engine
        workflow_engine = RAGWorkflowEngine(
            query_processor=query_processor,
            retriever=retriever,
            llm_client=llm_client,
            escalation_manager=escalation_manager
        )
        
        logger.info("RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process user query"""
    
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        logger.info(f"Processing query from {request.user_id}: {request.query}")
        
        # Execute workflow
        result = workflow_engine.execute(
            query=request.query,
            user_id=request.user_id
        )
        
        return QueryResponse(
            query_id=result.query_id,
            response=result.response,
            is_escalated=result.is_escalated,
            escalation_id=result.escalation_id,
            confidence=result.confidence,
            sources=result.sources,
            execution_time_ms=result.execution_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/escalations", response_model=List[EscalationReview])
async def get_escalations(status: str = "pending", limit: int = 10):
    """Get escalations for human review"""
    
    if not escalation_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        escalations = escalation_manager.get_escalation_queue(status=status, limit=limit)
        
        return [
            EscalationReview(
                escalation_id=esc.escalation_id,
                original_query=esc.original_query,
                ai_response=esc.ai_response,
                confidence_score=esc.confidence_score,
                escalation_reason=esc.escalation_reason,
                created_at=esc.created_at.isoformat()
            )
            for esc in escalations
        ]
        
    except Exception as e:
        logger.error(f"Error fetching escalations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/escalation/{escalation_id}", response_model=EscalationReview)
async def get_escalation(escalation_id: str):
    """Get specific escalation"""
    
    if not escalation_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        esc = escalation_manager.get_escalation(escalation_id)
        
        if not esc:
            raise HTTPException(status_code=404, detail="Escalation not found")
        
        return EscalationReview(
            escalation_id=esc.escalation_id,
            original_query=esc.original_query,
            ai_response=esc.ai_response,
            confidence_score=esc.confidence_score,
            escalation_reason=esc.escalation_reason,
            created_at=esc.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching escalation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/escalation/{escalation_id}/resolve")
async def resolve_escalation(escalation_id: str, resolution: EscalationResolution):
    """Resolve escalation with human response"""
    
    if not escalation_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = escalation_manager.submit_human_response(
            escalation_id=escalation_id,
            human_response=resolution.human_response,
            agent_name=resolution.agent_name,
            feedback_rating=resolution.feedback_rating
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Escalation not found")
        
        return {"status": "resolved", "escalation_id": escalation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving escalation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics"""
    
    if not escalation_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        stats = escalation_manager.get_feedback_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower()
    )
