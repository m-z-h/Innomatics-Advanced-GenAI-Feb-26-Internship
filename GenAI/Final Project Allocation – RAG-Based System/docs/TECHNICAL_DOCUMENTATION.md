# Technical Documentation - RAG-Based Customer Support Assistant

## 1. Introduction

### What is RAG?

Retrieval-Augmented Generation (RAG) is a machine learning technique that combines two powerful capabilities:

1. **Retrieval**: Finding relevant information from a knowledge base using semantic similarity
2. **Generation**: Using a large language model (LLM) to synthesize answers based on retrieved context

**Traditional LLM Problem**: Without external context, LLMs generate responses based solely on training data, which:
- Cannot access company-specific information
- May hallucinate or provide outdated information
- Cannot cite sources
- Cannot adapt to new knowledge without retraining

**RAG Solution**: By retrieving relevant documents before generation, RAG systems:
- Ground responses in verified knowledge
- Provide source citations
- Enable real-time knowledge base updates
- Reduce hallucinations significantly
- Support human-in-the-loop oversight

### Why RAG is Needed

**Use Case**: Customer support for a SaaS product with 1000+ documentation pages
- Support agents need quick access to relevant information
- Customers want contextually accurate answers
- Company wants to reduce response time and escalation rate
- System must cite sources for compliance/transparency

**Without RAG**: Support agent must search documentation manually → slow, inconsistent
**With RAG**: System automatically retrieves + synthesizes answer → fast, consistent, traceable

### Use Case Overview

**System**: RAG-Based Customer Support Assistant
**Users**: 
- End customers (ask questions via web/chat)
- Support agents (review escalations, provide feedback)
- System administrators (manage knowledge base)

**Workflow**:
```
Customer: "How do I reset my password?"
    ↓
System retrieves: [FAQ_page_5.txt, Account_Settings.txt, Security_Guide.txt]
    ↓
LLM synthesizes: "Go to Settings > Account > Password Reset..."
    ↓
Customer receives response with sources
```

---

## 2. System Architecture Explanation

### 2.1 Component Interactions

#### Phase 1: Offline - Knowledge Base Indexing
```
PDF Files
  ↓
[Document Loader] 
  ↓ Extracts text + metadata
Raw text with page numbers
  ↓
[Chunking Module]
  ↓ Splits into 1024-token chunks with overlap
List of Chunks
  ↓
[Embedding Module]
  ↓ Converts each chunk to 1536-dim vector
Embedding vectors
  ↓
[ChromaDB]
  ↓ Stores vectors + metadata
INDEXED KNOWLEDGE BASE (persistent)
```

This happens once during setup/knowledge base update.

#### Phase 2: Online - Query Processing
```
User Query: "How do I reset my password?"
  ↓
[Query Processing]
  ├─ Clean: lowercase, remove special chars
  ├─ Detect Intent: password_reset (confidence: 0.95)
  └─ State: WorkflowState
  ↓
[LangGraph Execution]
  ├─ INPUT_NODE: Format query
  ├─ RETRIEVAL_NODE: Embed query → Search ChromaDB
  │  Returns: Top-5 chunks with similarity scores
  ├─ DECISION_NODE: Analyze confidence
  │  If confidence > 0.8 → GENERATION_NODE
  │  If confidence < 0.6 → ESCALATION_NODE
  ├─ GENERATION_NODE or ESCALATION_NODE
  └─ OUTPUT_NODE: Format response
  ↓
Response or Escalation ID
  ↓
User receives response
```

### 2.2 Data Flow Example

**Input**: "What's your refund policy?"

```
1. RETRIEVAL LAYER
   Query: "What's your refund policy?"
   ├─ Embedding generated (1536 dims)
   ├─ ChromaDB search returns:
   │  [
   │    {chunk: "Refunds: 30-day money back...", score: 0.92},
   │    {chunk: "Returns & Refunds Policy...", score: 0.88},
   │    {chunk: "Billing questions FAQ...", score: 0.75}
   │  ]
   └─ Retrieval confidence: (0.92 + 0.88 + 0.75) / 3 = 0.85 ✓
   
2. DECISION LAYER
   Confidence = 0.85 > threshold (0.8)
   → Proceed to GENERATION_NODE
   
3. GENERATION LAYER
   Prompt Construction:
   """
   Context from knowledge base:
   [Chunk 1]: "Refunds: 30-day money back guarantee..."
   [Chunk 2]: "Returns & Refunds Policy: All purchases..."
   
   Question: What's your refund policy?
   
   Answer: (1-2 sentences, cite sources)
   """
   
   LLM generates:
   "We offer a 30-day money-back guarantee. If you're not satisfied, 
    you can request a full refund. See our Refunds Policy document."
   
4. OUTPUT LAYER
   Response Format:
   {
     "answer": "We offer a 30-day money-back guarantee...",
     "sources": [
       {"title": "Refunds Policy", "page": 5, "confidence": 0.92}
     ],
     "confidence": 0.87,
     "escalated": false
   }
```

### 2.3 Error Handling Flow

```
Scenario 1: No relevant chunks found
User: "What is quantum computing?"
   ↓
Retrieval returns 0 chunks with score > 0.6
   ↓
DECISION_NODE detects: confidence = 0
   ↓
ESCALATION_NODE triggered
   ↓
Escalation created + user notified
   ↓
Support agent reviews + provides response

Scenario 2: LLM service error
   ↓
LLM API times out
   ↓
Exception caught in GENERATION_NODE
   ↓
ESCALATION_NODE as fallback
   ↓
Escalation created with note: "LLM unavailable"
```

---

## 3. Design Decisions

### 3.1 Chunk Size: 1024 Tokens (4KB text)

**Rationale**:
- **Too Small (<512 tokens)**: Fragments context, loses coherence
- **Too Large (>2048 tokens)**: Increases noise, slower retrieval, high embedding cost
- **1024 tokens**: Balances context preservation with retrieval precision

**Example**:
```
Small chunk: "How do I enable 2FA?"
                → Missing context, ambiguous

1024-token chunk: "Account Security
                   Two-factor authentication (2FA) adds an extra layer...
                   Enable 2FA: Go to Settings > Security > 2FA...
                   Note: You'll receive codes via SMS or authenticator app."
                → Complete, contextual, retrievable

Large chunk: [entire FAQ document + other content]
            → Noisy, retrieves irrelevant stuff too
```

### 3.2 Embedding Strategy: Sentence-Transformer vs OpenAI

**Choice**: OpenAI text-embedding-3-small (1536 dims)

**Trade-offs**:

| Factor | OpenAI | Local (Sentence-Transformer) |
|--------|--------|------------------------------|
| **Accuracy** | 9.5/10 | 8/10 |
| **Speed** | 200ms (API) | 50ms (local) |
| **Cost** | $0.02/M tokens | Free |
| **Privacy** | Data sent to OpenAI | Stays local |
| **Scalability** | Unlimited (cloud) | Limited by GPU |

**Decision**: OpenAI for accuracy + reliability (for production) OR Local for privacy/cost

### 3.3 Retrieval Approach: Semantic + Optional Reranking

**Stage 1: Dense Retrieval** (Fast)
```
Query embedding → ChromaDB ANN search → Top-10 candidates
Time: 50ms, Recall: 90%+
```

**Stage 2: Reranking** (Optional, slow)
```
Top-10 candidates → Cross-encoder → Top-5 reranked
Time: +200ms, Precision: +5%

When to use: Complex queries, low confidence scenarios
```

**Decision**: Dense retrieval for latency (<100ms SLA). Reranking only if confidence < 0.7.

### 3.4 Prompt Design Logic

**Principle**: Few-shot examples + explicit constraints

**Prompt Template**:
```
You are a helpful customer support assistant. Answer based ONLY on provided context.

Context:
{chunks with citations}

Instructions:
- Be concise (1-2 sentences)
- Cite sources explicitly
- If unsure, say "I'm not certain"
- Never make up information

Question: {user_query}

Answer:
```

**Why this works**:
- Explicit constraints reduce hallucination
- Citations improve transparency
- Few-shot examples (not shown) guide tone
- Temperature=0.3 (low) for factual consistency

---

## 4. Workflow Explanation

### 4.1 LangGraph Usage: Why Not Just Sequential Code?

**Sequential approach** (traditional):
```python
# Problem: Spaghetti logic, hard to debug
query = user_input
chunks = retrieve(query)
if confidence(chunks) > 0.8:
    response = generate(chunks)
    return response
else:
    escalate()
    return "Escalated"
```

**Issues**:
- State implicit (scattered variables)
- Hard to trace execution path
- Difficult to add conditional branches
- Testing requires mocking entire flow

**LangGraph approach** (explicit state machine):
```python
# Benefit: Clear state transitions, debuggable
state = {"query": user_input, ...}
state = input_node(state)        # State now has intent
state = retrieval_node(state)    # State now has chunks
state = decision_node(state)     # State now has routing decision

if state["should_escalate"]:
    state = escalation_node(state)
else:
    state = generation_node(state)

state = output_node(state)
return state["final_response"]
```

**Benefits**:
- Explicit state at each step
- Deterministic routing
- Easy to visualize and debug
- Clean separation of concerns

### 4.2 Node Responsibilities

| Node | Input | Processing | Output |
|------|-------|-----------|--------|
| **INPUT_NODE** | Raw query string | Clean, detect intent | ProcessedQuery object |
| **RETRIEVAL_NODE** | Cleaned query | Embed + search ChromaDB | List of chunks + scores |
| **DECISION_NODE** | Chunks + scores | Evaluate confidence | Routing decision |
| **GENERATION_NODE** | Query + chunks | Format prompt + call LLM | Generated response |
| **ESCALATION_NODE** | Query + context | Create escalation record | Escalation ID |
| **OUTPUT_NODE** | Response/Escalation | Format for client | Final JSON/text |

### 4.3 State Transitions

```
WorkflowState TypedDict:
{
    query: str,                      # Original user query
    user_id: str,                    # Identify user
    processed_query: ProcessedQuery, # After cleaning + intent
    retrieval_result: RetrievalResult,  # After ChromaDB search
    llm_response: str,               # After LLM generation
    confidence_score: float,         # Confidence [0, 1]
    should_escalate: bool,           # Routing decision
    escalation_reason: str,          # Why escalated
    final_response: str              # What user sees
}
```

**Transition Example**:
```
INPUT_NODE:
  {} 
  → {query, user_id, processed_query}

RETRIEVAL_NODE:
  {query, user_id, processed_query}
  → {query, user_id, processed_query, retrieval_result}

DECISION_NODE:
  {query, ..., retrieval_result}
  → {query, ..., retrieval_result, should_escalate, confidence_score}

[Branching]
If should_escalate=True:
  ESCALATION_NODE
  → {query, ..., escalation_reason, final_response}
Else:
  GENERATION_NODE
  → {query, ..., llm_response, final_response}

OUTPUT_NODE:
  {query, ..., final_response}
  → {query, ..., final_response} [formatted]
```

---

## 5. Conditional Logic

### 5.1 Intent Detection

**Simple Rule-Based Method**:
```python
intents = {
    "password_reset": ["reset password", "forgot password", "can't login"],
    "billing": ["charge", "invoice", "payment", "refund"],
    "technical_support": ["error", "bug", "not working", "crash"],
    "account": ["profile", "email", "settings", "delete account"]
}

def detect_intent(query: str) -> str:
    query_lower = query.lower()
    for intent, keywords in intents.items():
        if any(kw in query_lower for kw in keywords):
            return intent
    return "general"  # Unknown intent
```

**Advanced Option** (ML classifier):
```
Train classifier on historical support queries
→ More accurate for nuanced questions
→ Requires labeled training data
```

### 5.2 Routing Decisions

**Decision Tree**:
```
Is retrieval_confidence >= 0.75?
├─ YES:
│  └─ Does retrieved content have >= 3 chunks?
│     ├─ YES: Generate response
│     └─ NO: Escalate (insufficient context)
└─ NO:
   └─ Is query complexity high (length > 200 chars)?
      ├─ YES: Escalate (complex query)
      └─ NO: Attempt generation with disclaimer
```

**Routing Matrix**:
```
                    High Confidence    Low Confidence
                    (>0.75)           (<0.75)
────────────────────────────────────────────────
Few results (<3)    Escalate          Escalate
Many results (>=3)  Generate          Escalate or
                    response          Generate+
                                      Disclaimer
```

### 5.3 Example Routing Scenarios

**Scenario 1: Clear, in-scope question**
```
Query: "How do I enable 2FA?"
├─ Intent: account (0.95)
├─ Retrieval: 5 chunks (scores: 0.92, 0.87, 0.81, 0.78, 0.73)
├─ Confidence: 0.82 ✓
├─ Complexity: Low ✓
└─ Route: GENERATE RESPONSE
```

**Scenario 2: Ambiguous question**
```
Query: "Something's broken"
├─ Intent: technical_support (0.45) [ambiguous]
├─ Retrieval: 2 chunks (scores: 0.62, 0.55)
├─ Confidence: 0.58 ✗
├─ Complexity: Low
└─ Route: ESCALATE (insufficient context)
```

**Scenario 3: Out-of-scope query**
```
Query: "What's the meaning of life?"
├─ Intent: general (not recognized)
├─ Retrieval: 0 chunks above threshold
├─ Confidence: 0 ✗
├─ Complexity: Out-of-scope
└─ Route: ESCALATE (out of knowledge base)
```

---

## 6. HITL Implementation

### 6.1 Role of Human Intervention

**Why HITL is Critical**:

1. **Quality Assurance**: Low-confidence queries reviewed before reaching customer
2. **Learning**: Human corrections provide feedback signal
3. **Complex Cases**: Multi-part questions needing coordination
4. **Sensitive Topics**: Legal, billing, policy decisions

**When Human Needed**:
- Confidence < 0.6
- Query involves money/legal terms
- Multi-intent queries
- Out-of-scope questions

### 6.2 Escalation Lifecycle

```
Step 1: QUERY ARRIVES
User: "I was charged twice on my account"

Step 2: SYSTEM PROCESSES
├─ Intent: billing (high confidence)
├─ Retrieval: 4 chunks on refunds/charges
├─ BUT: Query involves potential error (sensitive)
└─ Decision: Escalate for human judgment

Step 3: ESCALATION CREATED
├─ Store: Query + retrieved chunks + AI response
├─ Notify: Support team gets alert
└─ Show user: "A specialist will help within 2 hours"

Step 4: HUMAN REVIEW
Support agent sees:
  Query: "I was charged twice on my account"
  Retrieved info: [Refund policy, Billing FAQ, ...]
  AI suggested response: "According to our policy, ..."
  Agent action: Approve / Modify / Override

Step 5: RESOLUTION
├─ If approve: Send AI response as-is
├─ If modify: Send edited response
└─ If override: Send completely custom response

Step 6: FEEDBACK
├─ Log: What human did vs what AI suggested
├─ Pattern: Identify systematic AI weaknesses
└─ Improve: Update prompts / retrieval strategy
```

### 6.3 Benefits and Limitations

**Benefits**:
| Benefit | Impact |
|---------|--------|
| **Quality Control** | Only high-quality responses reach customers |
| **Trust** | Complex cases get expert attention |
| **Learning** | Feedback loop improves system over time |
| **Compliance** | Human oversight for sensitive topics |
| **Accountability** | Clear audit trail |

**Limitations**:
| Limitation | Mitigation |
|------------|-----------|
| **Latency** | Humans slower than AI (~minutes vs seconds) | Use only for complex queries |
| **Cost** | Support team time required | Automate easy 80% |
| **Bottleneck** | Escalations can queue up | Monitor SLA, hire as needed |
| **Inconsistency** | Different agents, different answers | Train, document, provide templates |

---

## 7. Challenges & Trade-offs

### 7.1 Retrieval Accuracy vs Speed

**Problem**: More accurate retrieval (reranking, semantic expansion) takes longer

**Trade-off**:
```
Fast (<100ms):
  Dense retrieval only
  ✓ Low latency
  ✗ ~5-10% accuracy loss

Balanced (100-500ms):
  Dense retrieval + optional reranking
  ✓ Good accuracy (95%)
  ✓ Acceptable latency
  ✗ More complex code

Accurate (500ms+):
  Dense + reranking + semantic expansion
  ✓ Highest accuracy (~98%)
  ✗ Slow for high-volume
```

**Decision**: Balanced approach. Use fast path (dense) for 80% of queries, reranking for low-confidence (<0.7).

### 7.2 Chunk Size vs Context Quality

**Problem**: Larger chunks preserve context but increase retrieval noise

**Trade-off**:
```
512 tokens:
  ✓ Precise matching
  ✗ Fragments context ("How do I..." vs full answer)

1024 tokens:
  ✓ Good context preservation
  ✓ Reasonable size for embedding
  ✓ Balanced trade-off

2048 tokens:
  ✓ Full context (e.g., entire FAQ)
  ✗ Embedding cost 2x
  ✗ Retrieval noise (returns irrelevant content)
```

**Decision**: 1024 tokens with 128-token overlap. Preserves answer + some context without over-chunking.

### 7.3 Cost vs Performance

**LLM Choices**:

| Model | Cost | Speed | Quality |
|-------|------|-------|---------|
| **GPT-3.5-turbo** | $0.50/M | Fast | Good (95%) |
| **GPT-4** | $3/M | Slow | Excellent (99%) |
| **Llama-2 (local)** | Free | Medium | Good (90%) |

**Decision**: GPT-3.5-turbo for balance. Switch to Llama-2 if privacy critical or cost becomes concern.

### 7.4 Scalability vs Operational Complexity

**At 100 QPS** (1M queries/day):
```
Single machine + ChromaDB: ✓ Works
Load: ~80% CPU
Cost: ~$500/month (server + API)

At 1000 QPS** (10M queries/day):
```
Need distributed architecture:
├─ Load balancer (distribute queries)
├─ Multiple API servers (process queries)
├─ Pinecone or Weaviate (distributed vector DB)
├─ Redis (cache + rate limiting)
└─ Monitoring + alerting

Operational complexity: 5x higher
Cost: $2000+/month
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Module**: Chunking

```python
def test_chunk_overlap():
    text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4."
    chunks = chunker.chunk(text, size=2, overlap=1)
    assert len(chunks) == 3
    assert "Sentence 2" in chunks[0] and chunks[1]  # Overlap
```

**Module**: Intent Detection

```python
def test_password_reset_intent():
    result = detect_intent("How do I reset my password?")
    assert result == "password_reset"
    
def test_unknown_intent():
    result = detect_intent("What's the meaning of life?")
    assert result == "general"
```

### 8.2 Integration Tests

**Test**: End-to-end query → response

```python
def test_simple_query():
    query = "How do I reset my password?"
    response = system.process_query(query, user_id="test_user")
    
    assert response.status == "success"
    assert len(response.sources) > 0
    assert "password" in response.text.lower()
    assert response.confidence > 0.7
```

**Test**: Escalation logic

```python
def test_escalation_on_low_confidence():
    query = "Something obscure about quantum mechanics"
    response = system.process_query(query)
    
    assert response.escalated == True
    assert response.escalation_id is not None
```

### 8.3 Sample Queries

**In-Scope Queries** (should generate responses):
```
1. "How do I reset my password?" → password_reset intent
2. "What's your refund policy?" → billing intent
3. "How do I enable 2FA?" → account intent
4. "I'm getting an error 503" → technical_support intent
```

**Edge Cases** (should escalate):
```
1. "Something seems wrong" → Ambiguous, insufficient context
2. "I want to sue you" → Legal, sensitive
3. "Can you explain quantum computing?" → Out of scope
4. "I need to speak to a manager" → Requires human
```

**Complex Queries** (multi-intent, may need escalation):
```
1. "I was charged twice AND can't reset password" → 2 intents
2. "Your product is awful and I want a refund NOW" → Emotional + billing
```

### 8.4 Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Accuracy** | >85% | Human evaluation of 100 responses |
| **Latency** | <3s | P95 response time |
| **Escalation Rate** | <15% | Count escalations / total queries |
| **HITL Resolution Rate** | >80% | Count resolved / total escalations |
| **Customer Satisfaction** | >4/5 | Post-conversation rating |

---

## 9. Future Enhancements

### 9.1 Multi-Document Support

**Current**: Single knowledge base

**Future**:
```
├─ FAQ database
├─ API documentation
├─ Product guides
├─ Troubleshooting wiki
└─ Real-time ticket history
```

**Implementation**:
- Metadata tagging per source
- Source-specific filtering in retrieval
- Priority weighting (FAQ > guides)

### 9.2 Feedback Loop Integration

**Current**: Escalations logged, not fully analyzed

**Future**:
```
1. Collect human corrections
2. Identify patterns (e.g., 30% escalations are about pricing)
3. Add new FAQ chunks on pricing
4. Re-test system
5. Measure improvement
```

### 9.3 Memory / Context Windows

**Current**: Each query independent

**Future**:
```
Query 1: "How do I reset my password?"
→ Response + Context stored

Query 2: "Will I receive an email?" [Follow-up]
→ System remembers Query 1 context
→ Understands "email" refers to password reset email
→ Better response
```

**Implementation**: Session memory with context windows

### 9.4 Continuous Learning

**Current**: Manual feedback from escalations

**Future**:
```
1. Auto-evaluate system responses
2. A/B test prompt variations
3. Track metric changes
4. Automatically promote better version
5. No human intervention needed (mostly)
```

### 9.5 Deployment Architectures

**Current Development**:
```
Laptop / Local Server
├─ Python FastAPI
├─ Local ChromaDB
└─ OpenAI API calls
```

**Production v1** (100 QPS):
```
Docker container on AWS
├─ API: FastAPI on ECS
├─ Vector DB: Pinecone
├─ Cache: Redis
└─ Monitoring: CloudWatch
```

**Production v2** (1000+ QPS):
```
Kubernetes cluster
├─ API servers (scaled)
├─ Vector DB (distributed)
├─ Queue (SQS for async)
├─ Feedback loop (ML pipeline)
└─ Full observability (Prometheus + Grafana)
```

---

## 10. Conclusion

This RAG system demonstrates:

✓ **Retrieval** grounded in real documents (ChromaDB)
✓ **Generation** with context (LLM + prompting)
✓ **Orchestration** with clear state flow (LangGraph)
✓ **Intelligent routing** based on confidence
✓ **Human oversight** for quality (HITL)
✓ **Production-ready** error handling and scalability

The system is designed for incremental development: start with core RAG, add LangGraph, add HITL, then scale to production. Each layer can be tested independently.

