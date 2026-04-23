# Low-Level Design (LLD) - RAG-Based Customer Support Assistant

## 1. Module-Level Design

### 1.1 Document Processing Module

**File**: `src/document_processor.py`

```
DocumentProcessor
├─ load_pdf(pdf_path: str) → PDFDocument
│  ├─ Uses: pdfplumber.open()
│  ├─ Extracts: text, metadata (title, author, creation_date)
│  └─ Returns: PDFDocument object with pages
│
├─ extract_text_with_metadata(pdf: PDFDocument) → List[TextBlock]
│  ├─ Iterates through pages
│  ├─ Preserves: page numbers, section headers
│  └─ Returns: List[{text, page, section, char_position}]
│
└─ batch_process_pdfs(pdf_dir: str) → List[PDFDocument]
   ├─ Parallel processing using multiprocessing.Pool
   ├─ Handles errors gracefully
   └─ Returns: List of processed documents
```

**Responsibilities**:
- Load and parse PDF files
- Extract text with page references
- Handle corrupted/encrypted PDFs
- Preserve document structure

### 1.2 Chunking Module

**File**: `src/chunking.py`

```
ChunkingStrategy (Abstract)
├─ BaseChunker
│  ├─ chunk_size: int = 1024 (tokens)
│  ├─ overlap: int = 128 (tokens)
│  └─ separator: str = "\n\n"
│
└─ SentenceChunker (Concrete)
   ├─ split_by_sentences(text: str) → List[str]
   ├─ split_by_paragraphs(text: str) → List[str]
   └─ smart_split(text: str) → List[Chunk]
      ├─ Algorithm:
      │  1. Split by double newline (paragraphs)
      │  2. If paragraph > max_size, split by sentences
      │  3. If sentence > max_size, split by tokens
      │  4. Add overlap between chunks
      └─ Returns: List[Chunk with metadata]
```

**Data Structure**: `Chunk`
```python
@dataclass
class Chunk:
    text: str              # Actual content
    chunk_id: str          # Unique identifier
    source_file: str       # PDF filename
    page_number: int       # Page in PDF
    start_char: int        # Position in document
    metadata: Dict[str, Any]  # Custom metadata
    created_at: datetime
```

### 1.3 Embedding Module

**File**: `src/embeddings.py`

```
EmbeddingProvider (Abstract)
├─ OpenAIEmbedding (Concrete)
│  ├─ model: str = "text-embedding-3-small"
│  ├─ dimension: int = 1536
│  └─ embed(text: str) → np.ndarray
│     ├─ Calls OpenAI API
│     ├─ Caches results
│     └─ Handles rate limiting
│
└─ LocalEmbedding (Concrete)
   ├─ model: str = "sentence-transformers/all-MiniLM-L6-v2"
   ├─ dimension: int = 384
   └─ embed(text: str) → np.ndarray
      ├─ Uses HuggingFace
      ├─ GPU accelerated if available
      └─ No API costs
```

**Usage Pattern**:
```python
embedder = EmbeddingProvider.factory(provider="openai")  # or "local"
embedding = embedder.embed("How do I reset my password?")
# Returns: np.ndarray of shape (1536,)
```

### 1.4 Vector Storage Module

**File**: `src/vector_store.py`

```
VectorStore (Abstract)
├─ ChromaDB (Concrete)
│  ├─ __init__(persist_dir: str, embedding_fn)
│  │  └─ Creates persistent collection for knowledge base
│  │
│  ├─ add_chunks(chunks: List[Chunk]) → None
│  │  ├─ Embeds all chunks
│  │  ├─ Stores in ChromaDB with metadata
│  │  └─ Updates internal index
│  │
│  ├─ search(query_embedding: np.ndarray, k: int = 5) → List[SearchResult]
│  │  ├─ Similarity search in ChromaDB
│  │  ├─ Returns top-K results with scores
│  │  └─ Scores normalized to [0, 1]
│  │
│  └─ delete_chunk(chunk_id: str) → None
│     └─ Removes chunk from index
│
└─ PineconeDB (Future: Production)
   ├─ Similar interface to ChromaDB
   ├─ Scales to millions of vectors
   └─ Managed cloud service
```

**Data Structure**: `SearchResult`
```python
@dataclass
class SearchResult:
    chunk: Chunk
    similarity_score: float  # [0, 1]
    rank: int
    matched_content: str  # Highlighted match
```

### 1.5 Retrieval Module

**File**: `src/retrieval.py`

```
Retriever
├─ __init__(vector_store: VectorStore, embedder: EmbeddingProvider)
├─ retrieve(query: str, top_k: int = 5, threshold: float = 0.6) 
│  │  → RetrievalResult
│  ├─ Step 1: Embed query
│  ├─ Step 2: Search vector store
│  ├─ Step 3: Filter by threshold
│  ├─ Step 4: Rank and format results
│  └─ Step 5: Return RetrievalResult
│
├─ retrieve_with_reranking(query: str, top_k: int = 5)
│  │  → RetrievalResult
│  ├─ Retrieve top-10 candidates
│  ├─ Rerank using cross-encoder (optional)
│  └─ Return top-K
│
└─ retrieve_with_filters(query: str, filters: Dict) → RetrievalResult
   ├─ Apply metadata filters (e.g., source_file, date)
   └─ Then search within filtered set
```

**Data Structure**: `RetrievalResult`
```python
@dataclass
class RetrievalResult:
    query: str
    query_embedding: np.ndarray
    results: List[SearchResult]
    total_results: int
    retrieval_time_ms: float
    confidence: float  # avg(similarity_scores)
    status: str  # "success" | "no_results" | "error"
```

### 1.6 Query Processing Module

**File**: `src/query_processor.py`

```
QueryProcessor
├─ __init__(retriever: Retriever, llm_client: LLMClient)
├─ process_query(user_query: str) → ProcessedQuery
│  ├─ Clean query text (remove special chars, normalize)
│  ├─ Detect intent (classification)
│  ├─ Retrieve relevant chunks
│  ├─ Prepare context for LLM
│  └─ Return ProcessedQuery
│
├─ detect_intent(query: str) → IntentResult
│  ├─ Categories: password_reset, billing, technical_support, etc.
│  ├─ Method: Keywords + optional ML classifier
│  └─ Returns: {intent, confidence, category}
│
└─ prepare_llm_context(retrieval_result: RetrievalResult) 
   │  → str  # Formatted prompt
   ├─ Template: "Context: {chunks}\n\nQuestion: {query}\n\nAnswer:"
   ├─ Include source citations
   └─ Limit total context length (~2000 tokens)
```

**Data Structure**: `ProcessedQuery`
```python
@dataclass
class ProcessedQuery:
    original_query: str
    cleaned_query: str
    intent: str
    intent_confidence: float
    retrieved_chunks: List[Chunk]
    retrieval_scores: List[float]
    avg_retrieval_confidence: float
    llm_context: str
    requires_escalation_check: bool
```

### 1.7 Graph Execution Module

**File**: `src/graph_engine.py`

```
GraphEngine (LangGraph-based)
├─ State (TypedDict)
│  ├─ query: str
│  ├─ processed_query: ProcessedQuery
│  ├─ retrieval_result: RetrievalResult
│  ├─ llm_response: str
│  ├─ confidence_score: float
│  ├─ should_escalate: bool
│  ├─ escalation_reason: str
│  └─ final_response: str
│
├─ build_graph() → StateGraph
│  ├─ Node: "input_node" (format query)
│  ├─ Node: "retrieval_node" (retrieve chunks)
│  ├─ Node: "decision_node" (check confidence)
│  ├─ Node: "generation_node" (generate response)
│  ├─ Node: "escalation_node" (prepare escalation)
│  ├─ Node: "output_node" (format response)
│  │
│  └─ Edges:
│     ├─ input_node → retrieval_node
│     ├─ retrieval_node → decision_node
│     ├─ decision_node → generation_node (if high_confidence)
│     ├─ decision_node → escalation_node (if low_confidence)
│     ├─ generation_node → output_node
│     └─ escalation_node → output_node
│
├─ process_workflow(query: str) → WorkflowResult
│  ├─ Execute graph
│  ├─ Track state transitions
│  └─ Return final result
│
└─ debug_trace(query: str) → ExecutionTrace
   ├─ Logs all state transitions
   └─ Useful for debugging routing issues
```

### 1.8 HITL Module

**File**: `src/hitl.py`

```
EscalationManager
├─ __init__(db_connection, notification_service)
├─ create_escalation(escalation_data: EscalationData) → str
│  ├─ Generate escalation_id
│  ├─ Store in database
│  ├─ Send notification to support team
│  └─ Return escalation_id
│
├─ get_escalation_queue(limit: int = 10) → List[EscalationData]
│  └─ Retrieve pending escalations for human review
│
├─ submit_human_response(escalation_id: str, 
│  │                      human_response: str,
│  │                      agent_name: str) → bool
│  ├─ Update escalation with human response
│  ├─ Mark as resolved
│  ├─ Log for feedback learning
│  └─ Send response to user
│
└─ get_feedback_stats() → Dict
   ├─ Count escalations by reason
   ├─ Agent performance metrics
   └─ Useful for monitoring
```

**Data Structure**: `EscalationData`
```python
@dataclass
class EscalationData:
    escalation_id: str
    original_query: str
    retrieval_results: List[SearchResult]
    ai_response: str  # What AI was going to say
    confidence_score: float
    escalation_reason: str  # "low_confidence" | "complex_query" | "out_of_scope"
    user_id: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    human_response: Optional[str] = None
    feedback_rating: Optional[int] = None  # 1-5 stars
```

---

## 2. Data Structures

### Core Data Models

```python
# Document representation
@dataclass
class Document:
    doc_id: str
    filename: str
    content: str
    created_at: datetime
    metadata: Dict[str, Any]

# Chunk representation
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    page_number: int
    start_char: int
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]

# Query-Response schema
@dataclass
class UserQuery:
    query_id: str
    user_id: str
    text: str
    session_id: str
    created_at: datetime

@dataclass
class SystemResponse:
    response_id: str
    query_id: str
    text: str
    source_chunks: List[str]
    confidence: float
    is_escalated: bool
    created_at: datetime

# State object for graph (LangGraph)
class WorkflowState(TypedDict):
    query: str
    user_id: str
    processed_query: ProcessedQuery
    retrieval_result: RetrievalResult
    llm_response: str
    confidence_score: float
    should_escalate: bool
    escalation_reason: str
    final_response: str
    metadata: Dict[str, Any]
```

---

## 3. LangGraph Workflow Design

### 3.1 Graph Structure

```
Input Query: "How do I reset my password?"
    │
    ▼
┌─────────────────────────────────┐
│ INPUT_NODE                      │
│ - Clean query                   │
│ - Detect intent                 │
│ - Add to state                  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ RETRIEVAL_NODE                  │
│ - Embed query                   │
│ - Search ChromaDB               │
│ - Filter by threshold           │
│ - Store results in state        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ DECISION_NODE                   │
│ Check conditions:               │
│ - High confidence?              │
│ - Enough results?               │
│ - Query complexity?             │
└─────────────────────────────────┘
    │
    ├─ YES (confidence > 0.8) ──────────────┐
    │                                       │
    │                                       ▼
    │                            ┌──────────────────┐
    │                            │GENERATION_NODE   │
    │                            │- Format prompt   │
    │                            │- Call LLM        │
    │                            │- Score response  │
    │                            └──────────────────┘
    │                                       │
    │                                       ▼
    │                            ┌──────────────────┐
    │                            │OUTPUT_NODE       │
    │                            │- Format response │
    │                            │- Return to user  │
    │                            └──────────────────┘
    │
    └─ NO (confidence < 0.8) ───────────────┐
                                             │
                                             ▼
                                  ┌──────────────────┐
                                  │ESCALATION_NODE   │
                                  │- Create record   │
                                  │- Notify team     │
                                  │- Queue for human │
                                  └──────────────────┘
                                             │
                                             ▼
                                  ┌──────────────────┐
                                  │OUTPUT_NODE       │
                                  │- Send escalation │
                                  │  message         │
                                  └──────────────────┘
```

### 3.2 Node Implementations

**Node: INPUT_NODE**
```python
def input_node(state: WorkflowState) -> WorkflowState:
    """Format and validate input query"""
    raw_query = state["query"]
    
    # Clean query
    cleaned = raw_query.strip().lower()
    
    # Detect intent
    intent_result = query_processor.detect_intent(cleaned)
    
    # Create ProcessedQuery
    state["processed_query"] = ProcessedQuery(
        original_query=raw_query,
        cleaned_query=cleaned,
        intent=intent_result["intent"],
        intent_confidence=intent_result["confidence"],
        ...
    )
    
    return state
```

**Node: RETRIEVAL_NODE**
```python
def retrieval_node(state: WorkflowState) -> WorkflowState:
    """Retrieve relevant chunks from knowledge base"""
    query = state["processed_query"].cleaned_query
    
    # Retrieve
    result = retriever.retrieve(
        query=query,
        top_k=5,
        threshold=0.6
    )
    
    state["retrieval_result"] = result
    
    # Store confidence for decision making
    state["confidence_score"] = result.confidence
    
    return state
```

**Node: DECISION_NODE**
```python
def decision_node(state: WorkflowState) -> WorkflowState:
    """Decide whether to generate response or escalate"""
    confidence = state["confidence_score"]
    num_results = len(state["retrieval_result"].results)
    
    # Decision rules
    if confidence < 0.6 or num_results == 0:
        state["should_escalate"] = True
        state["escalation_reason"] = "low_confidence"
    else:
        state["should_escalate"] = False
    
    return state
```

**Node: GENERATION_NODE**
```python
def generation_node(state: WorkflowState) -> WorkflowState:
    """Generate LLM response with retrieved context"""
    
    # Format context
    context = query_processor.prepare_llm_context(
        state["retrieval_result"]
    )
    
    # Call LLM
    response = llm_client.generate(
        prompt=context,
        temperature=0.3,
        max_tokens=500
    )
    
    state["llm_response"] = response
    
    # Score response confidence
    state["confidence_score"] = score_response(response, state["retrieval_result"])
    
    return state
```

**Node: ESCALATION_NODE**
```python
def escalation_node(state: WorkflowState) -> WorkflowState:
    """Prepare escalation to human"""
    
    escalation_data = EscalationData(
        escalation_id=uuid.uuid4().hex,
        original_query=state["query"],
        retrieval_results=state["retrieval_result"].results,
        ai_response=state["llm_response"],
        confidence_score=state["confidence_score"],
        escalation_reason=state["escalation_reason"],
        user_id=state["user_id"],
        created_at=datetime.now()
    )
    
    # Create escalation
    escalation_manager.create_escalation(escalation_data)
    
    state["final_response"] = f"Your query has been escalated to our support team. Reference ID: {escalation_data.escalation_id}"
    
    return state
```

**Node: OUTPUT_NODE**
```python
def output_node(state: WorkflowState) -> WorkflowState:
    """Format final response"""
    
    if state["should_escalate"]:
        response = state["final_response"]
    else:
        response = format_response(
            state["llm_response"],
            state["retrieval_result"],
            state["confidence_score"]
        )
    
    state["final_response"] = response
    return state
```

### 3.3 Conditional Edges

```python
def should_escalate(state: WorkflowState) -> str:
    """Routing logic for edges"""
    if state["should_escalate"]:
        return "escalation_node"
    else:
        return "generation_node"

# Add conditional edge
graph.add_conditional_edges(
    "decision_node",
    should_escalate,
    {
        "escalation_node": "escalation_node",
        "generation_node": "generation_node"
    }
)
```

---

## 4. Conditional Routing Logic

### Answer Generation Criteria

```
Generate Response If:
  ✓ confidence_score >= 0.75
  ✓ At least 3 relevant chunks found
  ✓ Query matches known intents
  ✓ No errors in retrieval
  ✓ LLM response passes validation
```

### Escalation Criteria

| Criteria | Condition | Action |
|----------|-----------|--------|
| **Low Confidence** | score < 0.6 | Escalate immediately |
| **No Results** | num_results == 0 | Escalate immediately |
| **Complex Query** | length > 200 chars OR multi-intent | Escalate |
| **Out of Scope** | intent not in known_intents | Escalate with suggestion |
| **Sensitive Query** | keywords: payment, legal, etc. | Escalate for human judgment |

### Example Routing Table

```
Input: "How do I reset my password?"
│
├─ Intent: password_reset (confidence: 0.95) ✓
├─ Retrieved chunks: 5 (avg score: 0.88) ✓
├─ LLM response: "Go to Settings > Account > Password Reset..." ✓
├─ Response validation: Pass ✓
│
└─ ROUTE: DIRECT_RESPONSE (Confidence: 0.88)


Input: "I'm having trouble with the new payment system and my account is locked"
│
├─ Intent: billing (0.7) + technical_support (0.6) ✓ (Multi-intent)
├─ Retrieved chunks: 2 (avg score: 0.62) ⚠
├─ Query complexity: High (multi-part) ⚠
│
└─ ROUTE: ESCALATE (Reason: Complex multi-intent query)
```

---

## 5. HITL Design

### 5.1 Escalation Workflow

```
Step 1: ESCALATION CREATED
├─ Query + Context + AI Response stored
├─ Notification sent to support team
└─ User informed of escalation ID

Step 2: HUMAN REVIEW
├─ Support agent views escalation
├─ Reviews: Query, Retrieved chunks, AI response
├─ Decision: Approve, Modify, or Override

Step 3: RESPONSE TO USER
├─ Human response or modified response sent
├─ Escalation marked as resolved
├─ Feedback recorded

Step 4: FEEDBACK LOOP
├─ Analyze human overrides
├─ Identify patterns in escalations
├─ Update chunking / retrieval strategy if needed
└─ Consider fine-tuning prompts
```

### 5.2 Agent Interface Schema

```python
@dataclass
class EscalationView:
    """What support agent sees"""
    escalation_id: str
    user_query: str
    retrieved_chunks: List[Dict]  # {content, source, score}
    ai_response: str
    confidence_score: float
    reason: str
    submission_time: datetime
    
    actions: List[str]  # ["approve", "modify", "override"]
```

### 5.3 Feedback Integration

```python
class FeedbackAnalyzer:
    def analyze_human_overrides(self, time_period: str) -> Dict:
        """Identify patterns in human corrections"""
        return {
            "total_escalations": int,
            "avg_resolution_time": float,
            "most_common_reasons": List[str],
            "agent_performance": Dict[str, float],
            "recommended_actions": List[str]
        }
```

---

## 6. API / Interface Design

### 6.1 REST API Endpoints

```
POST /api/v1/query
├─ Request: {"query": str, "user_id": str, "session_id": str}
├─ Response: {
│   "query_id": str,
│   "response": str,
│   "sources": List[{chunk_id, page, snippet}],
│   "confidence": float,
│   "is_escalated": bool,
│   "escalation_id": Optional[str]
│  }
└─ Status: 200 (success) | 400 (invalid) | 500 (error)


GET /api/v1/escalation/{escalation_id}
├─ Returns: EscalationData (for support agent review)
└─ Status: 200 | 404

POST /api/v1/escalation/{escalation_id}/resolve
├─ Request: {"human_response": str, "agent_name": str}
├─ Action: Mark escalation resolved, send response to user
└─ Status: 200 | 400 | 404


GET /api/v1/health
├─ Returns: {"status": "ok", "version": str}
└─ Status: 200

POST /api/v1/documents/upload
├─ Request: Form data with PDF file
├─ Action: Process and index PDF
├─ Response: {"doc_id": str, "chunks_created": int}
└─ Status: 200 | 400 | 413 (file too large)
```

### 6.2 CLI Interface

```bash
# Query
$ python cli.py query "How do I reset my password?" --user-id user123
Output:
Response: "Go to Settings > Account..."
Confidence: 0.88
Sources: [page 5, page 12]

# Upload document
$ python cli.py upload documents/faq.pdf
Output:
Uploaded: faq.pdf
Chunks created: 45
Embeddings: Created in 3.2s

# Check escalations
$ python cli.py escalations --pending
Output:
[ID: esc_001, Query: "...", Status: pending, Created: 2 hours ago]
```

### 6.3 Interaction Flow Diagram

```
USER                    API SERVER              BACKEND SERVICES
│                           │                            │
├─ POST /query ────────────>│                            │
│                           ├─ Clean query             │
│                           ├─ Detect intent           │
│                           ├──────────────────────────>│ Retriever
│                           │<─ Top-5 chunks ──────────│
│                           ├─ Check confidence        │
│                           │
│                           ├─ IF high confidence:     │
│                           ├──────────────────────────>│ LLM
│                           │<─ Response ──────────────│
│                           │
│                           ├─ IF low confidence:      │
│                           ├──────────────────────────>│ HITL
│                           │   Create escalation      │
│                           │
│<─ 200 OK + Response ──────│
│   (or escalation ID)      │
│                           │
```

---

## 7. Error Handling

### 7.1 Missing Data Errors

```python
class DocumentProcessingError(Exception):
    """Raised when PDF processing fails"""
    pass

try:
    pdf = loader.load_pdf("missing.pdf")
except DocumentProcessingError as e:
    logger.error(f"Failed to load PDF: {e}")
    # Fallback: Return escalation
    escalate_with_message("Document loading failed")
```

### 7.2 No Relevant Chunks Found

```python
def handle_no_results(query: str, user_id: str) -> str:
    """Handle when retrieval returns no results"""
    
    if len(retrieval_result.results) == 0:
        escalate_query(
            query=query,
            user_id=user_id,
            reason="no_relevant_documents",
            message="Query outside knowledge base scope"
        )
        return "I couldn't find relevant information. A specialist will help you soon."
```

### 7.3 LLM Failure Handling

```python
def handle_llm_error(query: str, chunks: List[Chunk]) -> str:
    """Graceful fallback when LLM fails"""
    
    try:
        response = llm_client.generate(prompt)
    except LLMServiceError as e:
        logger.error(f"LLM failed: {e}")
        
        # Fallback 1: Return first chunk as response
        if chunks:
            return f"Based on available information: {chunks[0].text}"
        
        # Fallback 2: Escalate
        else:
            return escalate_query_error(query)
```

### 7.4 Vector Store Errors

```python
def handle_vector_store_error(query: str) -> str:
    """Handle ChromaDB connection failures"""
    
    try:
        results = vector_store.search(query_embedding)
    except ConnectionError as e:
        logger.error(f"Vector store unavailable: {e}")
        
        # Fallback: Use keyword search
        results = keyword_search(query)
        
        if not results:
            return escalate_query_error(query, "Vector store unavailable")
```

### 7.5 Error Handling Matrix

| Error Type | Severity | Action |
|------------|----------|--------|
| **PDF parsing fails** | High | Log + Escalate |
| **Embedding API down** | High | Use local embeddings + Escalate |
| **ChromaDB unavailable** | High | Retry + Escalate if persistent |
| **LLM service error** | Medium | Fallback response + Escalate |
| **Rate limit exceeded** | Medium | Queue and retry later |
| **Invalid user query** | Low | Return helpful error message |
| **Network timeout** | High | Retry with exponential backoff |

---

## 8. Database Schema (SQLite for Escalations)

```sql
-- Escalations table
CREATE TABLE escalations (
    escalation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    original_query TEXT NOT NULL,
    ai_response TEXT,
    confidence_score FLOAT,
    escalation_reason TEXT,
    status TEXT CHECK (status IN ('pending', 'resolved')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    human_response TEXT,
    agent_name TEXT,
    feedback_rating INTEGER CHECK (feedback_rating >= 1 AND feedback_rating <= 5)
);

-- Queries table (for analytics)
CREATE TABLE queries (
    query_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    was_escalated BOOLEAN,
    confidence_score FLOAT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table (metadata)
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    source_file TEXT,
    page_number INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 9. Summary

LLD provides implementation-level detail for each module, explicit data structures, LangGraph node/edge design, and error handling strategies. Key design decisions:
- **Semantic chunking** with overlap for context preservation
- **Confidence-based routing** to escalation
- **Graceful degradation** with fallback mechanisms
- **Human-in-the-loop** for quality assurance

This design supports incremental development: start with basic retrieval + LLM, add LangGraph, add HITL, then scale.

