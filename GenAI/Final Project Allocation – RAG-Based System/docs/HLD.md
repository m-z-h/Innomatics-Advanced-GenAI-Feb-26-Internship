# High-Level Design (HLD) - RAG-Based Customer Support Assistant

## 1. System Overview

### Problem Definition
Customer support teams face challenges in quickly retrieving relevant information from large knowledge bases to answer user queries accurately. Manual search is time-consuming, and responses often lack context. A system that automatically retrieves relevant information and generates contextual responses can reduce response time, improve consistency, and enhance customer satisfaction.

### Scope of the System
- **In Scope**: PDF-based knowledge base processing, semantic retrieval, intelligent routing, human escalation
- **Out of Scope**: Multi-language support, real-time conversation history persistence, complex workflow automation beyond routing
- **User Base**: Support agents, end customers through web/CLI interface
- **Expected Load**: 100-1000 queries/day initially, scalable to 10K+ queries/day

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INTERACTION LAYER                             │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │  Web Interface   │      │  CLI Interface   │      │  API Interface   │  │
│  └────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘  │
└───────────┼──────────────────────────┼──────────────────────────┼────────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │    QUERY PROCESSING & ROUTING      │
                    │  ┌──────────────────────────────┐  │
                    │  │  Intent Detection Module     │  │
                    │  └──────────────────────────────┘  │
                    └──────────────────┬─────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────────┐    ┌──────────────────────┐    ┌────────────────────┐
│  RETRIEVAL LAYER  │    │  LangGraph Workflow  │    │  HITL Escalation   │
│  ┌─────────────┐  │    │  ┌────────────────┐  │    │  ┌──────────────┐  │
│  │Query        │  │    │  │Processing Node │  │    │  │Escalation    │  │
│  │Embedding    │  │    │  ├────────────────┤  │    │  │Manager       │  │
│  │Generation   │  │    │  │Output Node     │  │    │  ├──────────────┤  │
│  ├─────────────┤  │    │  ├────────────────┤  │    │  │Human Review  │  │
│  │ChromaDB     │  │    │  │State Object    │  │    │  │Integration   │  │
│  │Vector Query │  │    │  └────────────────┘  │    │  └──────────────┘  │
│  ├─────────────┤  │    └──────────────────────┘    └────────────────────┘
│  │Retriever    │  │
│  │(Top-K)      │  │
│  └─────────────┘  │
└───────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│      LLM PROCESSING LAYER                │
│  ┌────────────────────────────────────┐  │
│  │ Prompt Engineering                 │  │
│  │ - Context injection                │  │
│  │ - Question format                  │  │
│  │ - Response constraints             │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ LLM (OpenAI / Llama / Local)       │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ Response Validation                │  │
│  │ - Confidence scoring               │  │
│  │ - Hallucination detection          │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│     RESPONSE GENERATION & ROUTING        │
│  ┌────────────────────────────────────┐  │
│  │ Confidence Check                   │  │
│  │ - If high → Direct response        │  │
│  │ - If low → Escalate                │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
        │
        ├─────────────────┬─────────────────┐
        ▼                 ▼                 ▼
    ┌────────┐     ┌──────────┐     ┌────────────┐
    │Response│     │Escalation│     │HITL Queue  │
    │to User │     │Decision  │     │for Review  │
    └────────┘     └──────────┘     └────────────┘
```

---

## 3. Component Description

### 3.1 Document Loader
**Purpose**: Extract text from PDF knowledge base files
- **Input**: PDF files
- **Processing**: 
  - PDF parsing (PyPDF2 / pdfplumber)
  - Text extraction with metadata preservation
  - Handling multi-page documents
- **Output**: Raw text with page/source references

### 3.2 Chunking Strategy
**Purpose**: Break documents into manageable pieces for embedding
- **Strategy**: Semantic chunking with overlap
- **Parameters**:
  - Chunk size: 1024 tokens (~4KB text)
  - Overlap: 128 tokens (12.5%)
  - Strategy: Split on sentence/paragraph boundaries
- **Output**: Chunks with metadata (source, page number, position)

### 3.3 Embedding Model
**Purpose**: Convert text to high-dimensional vectors
- **Model**: OpenAI text-embedding-3-small (default)
- **Vector Dimension**: 1536
- **Purpose**: Enables semantic similarity search
- **Alternative**: Sentence-Transformers for local deployment

### 3.4 Vector Store (ChromaDB)
**Purpose**: Store and index embeddings for fast retrieval
- **Why ChromaDB**:
  - Lightweight, no separate database needed
  - Built for embeddings
  - Supports persistent & in-memory modes
  - Simple filtering by metadata
- **Storage**: Persistent on disk for knowledge base
- **Indexing**: Similarity search using cosine distance

### 3.5 Retriever
**Purpose**: Find relevant knowledge base chunks for user query
- **Retrieval Type**: Top-K semantic search (K=5 default)
- **Process**:
  1. Embed user query
  2. Search ChromaDB for similar chunks
  3. Apply confidence threshold
  4. Return ranked results with scores
- **Fallback**: If no results above threshold, signal for escalation

### 3.6 LLM (Large Language Model)
**Purpose**: Generate contextual responses
- **Primary**: OpenAI GPT-4 / GPT-3.5-turbo
- **Alternative**: Local Llama-2 for privacy
- **Usage**: Synthesize answer from retrieved chunks
- **Temperature**: 0.3 (low for factual consistency)

### 3.7 Graph Workflow Engine (LangGraph)
**Purpose**: Orchestrate multi-step processing with conditional routing
- **Why LangGraph**:
  - Explicit state management
  - Deterministic flow control
  - Natural conditional branching
  - Integrates with LangChain
- **Graph Structure**: 2-3 nodes with conditional edges

### 3.8 Routing Layer
**Purpose**: Route queries based on intent and confidence
- **Routing Rules**:
  - High confidence (>0.85) → Direct response
  - Medium confidence (0.65-0.85) → Response with disclaimer
  - Low confidence (<0.65) → Escalate to human
  - Query complexity → Escalate if multi-step needed
  - Out-of-scope queries → Escalate

### 3.9 HITL (Human-in-the-Loop) Module
**Purpose**: Enable human review for escalated queries
- **Escalation Queue**: Database of pending human review
- **Agent Interface**: Show query + retrieved chunks + AI response
- **Human Action**: Approve, modify, or provide new response
- **Feedback Loop**: Learn from human interventions

---

## 4. Data Flow

### Query Lifecycle

```
1. USER SUBMITS QUERY
   ├─ Query: "How do I reset my password?"
   
2. QUERY PROCESSING
   ├─ Clean query text
   ├─ Detect intent (password_reset, billing, technical_support, etc.)
   ├─ Store query context (user_id, timestamp, session)
   
3. EMBEDDING & RETRIEVAL
   ├─ Generate query embedding (1536-dim vector)
   ├─ Search ChromaDB for top-5 similar chunks
   ├─ Retrieve chunks: [
   │   {chunk: "Password reset: Go to settings...", score: 0.92},
   │   {chunk: "Two-factor auth in account settings...", score: 0.87},
   │   {chunk: "Account recovery process...", score: 0.81},
   │   {...}, {...}
   │  ]
   ├─ Filter chunks by confidence threshold (>0.75)
   
4. LANGGRAPH WORKFLOW PROCESSING
   ├─ Node 1 (Input): Format retrieval results + query
   ├─ State: {
   │   query: str,
   │   intent: str,
   │   retrieved_chunks: List[Chunk],
   │   confidence_scores: List[float],
   │   state: "processing" | "escalate" | "respond"
   │  }
   ├─ Node 2 (Process): Check confidence + generate response
   ├─ Conditional Edge: 
   │   if confidence > 0.8 → Go to Output Node
   │   if confidence < 0.6 → Go to Escalation Node
   │   else → Go to Output Node (with disclaimer)
   
5. RESPONSE GENERATION or ESCALATION
   ├─ If Output Node:
   │   ├─ Format prompt: System + retrieved context + query
   │   ├─ Call LLM with top-3 chunks as context
   │   ├─ Validate response for hallucination
   │   ├─ Generate confidence score for response
   │   └─ Return response to user
   ├─ If Escalation Node:
   │   ├─ Create escalation record
   │   ├─ Add to human review queue
   │   ├─ Send "Human agent will review" message
   │   └─ Notify support team
   
6. HUMAN REVIEW (If Escalated)
   ├─ Agent sees: Query + retrieved chunks + AI response
   ├─ Agent takes action: Approve / Modify / Override
   ├─ Response sent to user
   ├─ Feedback recorded for model training
   
7. RESPONSE DELIVERY
   ├─ Format response (HTML, JSON, plain text)
   ├─ Include source citations
   ├─ Show confidence level (optional)
   └─ Deliver to user
```

---

## 5. Technology Choices

### Why ChromaDB?
| Criterion | ChromaDB | Alternatives |
|-----------|----------|---------------|
| **Setup** | Trivial (pip install) | Pinecone (cloud), Weaviate (docker) |
| **Cost** | Free | Pinecone (pay per query) |
| **Learning Curve** | Minimal | Medium-High |
| **Scalability** | Good (persistent) | Better (cloud) |
| **Use Case** | Perfect for RAG prototypes | Production at scale |

**Decision**: ChromaDB for development/prototype. Migrate to Pinecone/Weaviate for production scale (>1M vectors).

### Why LangGraph?
| Feature | LangGraph | Alternatives |
|---------|-----------|---------------|
| **Explicit State** | ✓ (Built-in) | LangChain (implicit) |
| **Conditional Routes** | ✓ (Native) | Apache Airflow (overkill) |
| **Debugging** | ✓ (State tracking) | Temporal (complex) |
| **Integration** | ✓ (LangChain ecosystem) | - |

**Decision**: LangGraph for workflow control. Clear state + conditional routing perfect for HITL.

### LLM Choice
- **Development**: OpenAI GPT-3.5-turbo (fast, reliable, low cost)
- **Production**: GPT-4 (higher accuracy) or Llama-2 (privacy)
- **Fallback**: Local embeddings + retrieval still work offline

### Additional Tools
- **PDF Processing**: pdfplumber (reliable, metadata-aware)
- **Embeddings**: OpenAI API or Sentence-Transformers (local)
- **Vector DB**: ChromaDB + optional Pinecone for scale
- **API Framework**: FastAPI (async, production-ready)
- **State Management**: SQLite for escalation queue initially

---

## 6. Scalability Considerations

### Handling Large Documents
- **Chunking Strategy**: Adaptive chunk size based on document structure
- **Parallel Processing**: Process PDFs in batches using multiprocessing
- **Incremental Indexing**: Add new documents to ChromaDB without re-processing

### Increasing Query Load
- **Current**: Single-threaded query handling (~10 QPS)
- **Scale 1**: Async processing with FastAPI (~100 QPS)
- **Scale 2**: Distributed retrieval with load balancing (~1000 QPS)
- **Scale 3**: Dedicated embedding service + Pinecone vector DB (~10K QPS)

### Latency Optimization
| Component | Latency | Optimization |
|-----------|---------|--------------|
| Embedding generation | 200ms | Batch embeddings, cache common queries |
| Vector search | 50ms | Use smaller embedding models, increase replicas |
| LLM call | 1-3s | Stream responses, use faster models (3.5-turbo) |
| HITL escalation | Variable | Pre-format escalation data, queue asynchronously |

**Current System Latency**: ~3.5 seconds (embedding 200ms + retrieval 50ms + LLM 2.5s + overhead 750ms)

### Cost Optimization
- Use smaller embedding models (text-embedding-3-small vs large)
- Cache frequent queries
- Batch process PDFs during off-peak hours
- Consider local models for sensitive data (privacy + cost)

---

## 7. Deployment Architecture

### Development Environment
- Local ChromaDB instance
- OpenAI API key for LLM
- Python 3.9+, FastAPI server on localhost:8000

### Production Environment
- **Vector DB**: Pinecone (managed, scalable)
- **LLM**: OpenAI API or self-hosted Llama on container
- **API Server**: FastAPI on Kubernetes/Docker
- **Escalation Queue**: PostgreSQL + Redis for caching
- **Monitoring**: Prometheus metrics + ELK stack for logs

---

## 8. Summary

This RAG system combines semantic retrieval with intelligent routing to provide contextual customer support. LangGraph enables explicit workflow control with conditional escalation to humans when confidence is low. ChromaDB provides lightweight but scalable vector storage, while OpenAI's LLM synthesizes answers from retrieved context. The HITL module ensures quality by routing complex queries to human agents, creating a feedback loop for continuous improvement.

