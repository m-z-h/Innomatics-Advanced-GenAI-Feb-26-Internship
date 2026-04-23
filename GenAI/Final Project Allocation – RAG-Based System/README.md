# RAG-Based Customer Support Assistant

A production-ready Retrieval-Augmented Generation (RAG) system with Human-in-the-Loop (HITL) escalation, built with LangGraph for intelligent workflow orchestration.

## 📋 Project Overview

This system combines semantic retrieval with an LLM to provide accurate, context-grounded customer support responses. Features include:

- **RAG Pipeline**: PDF ingestion → semantic chunking → embedding → vector search
- **LangGraph Workflow**: Explicit state machine with conditional routing
- **Intelligent Routing**: Confidence-based escalation to humans
- **HITL Integration**: Support agent interface for complex queries
- **Production APIs**: FastAPI REST endpoints + CLI interface

## 🏗️ Architecture

### Components

```
User Query
    ↓
[Intent Detection] → [Semantic Retrieval] → [Confidence Analysis]
    ↓                    ↓                    ↓
[LangGraph Workflow Engine]
    ├─ INPUT_NODE: Clean & process query
    ├─ RETRIEVAL_NODE: Embed & search knowledge base
    ├─ DECISION_NODE: Check confidence scores
    ├─ GENERATION_NODE: LLM response synthesis
    ├─ ESCALATION_NODE: Create HITL request
    └─ OUTPUT_NODE: Format response
    ↓
[Response or Escalation ID]
```

### Technology Stack

- **Embeddings**: OpenAI text-embedding-3-small (1536-dim vectors)
- **Vector DB**: ChromaDB (persistent, scalable)
- **LLM**: OpenAI GPT-3.5-turbo
- **Orchestration**: LangGraph (state-based workflows)
- **API**: FastAPI + Uvicorn
- **Database**: SQLite for escalations (extensible to PostgreSQL)

## 📦 Installation

### Prerequisites

- Python 3.9+
- OpenAI API key
- pip or conda

### Setup

```bash
# Clone/navigate to project
cd rag-support-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Add your OpenAI API key to .env
# OPENAI_API_KEY=sk-...
```

## 🚀 Quick Start

### 1. Upload a Knowledge Base PDF

```bash
python cli.py upload data/sample_faq.pdf
```

This will:
- Extract text from PDF
- Chunk using semantic boundaries (1024 tokens, 128-token overlap)
- Generate embeddings
- Index in ChromaDB

### 2. Query the System

```bash
python cli.py query "How do I reset my password?"
```

### 3. Start the REST API

```bash
python api.py
```

The API will be available at `http://localhost:8000`

Access documentation at `http://localhost:8000/docs`

## 📚 API Endpoints

### Query Processing

```bash
POST /api/v1/query
{
  "query": "How do I reset my password?",
  "user_id": "customer_123"
}

Response:
{
  "query_id": "qry_abc123",
  "response": "Go to Settings > Account > Password Reset...",
  "is_escalated": false,
  "confidence": 0.87,
  "sources": [
    {"file": "faq.pdf", "page": 5, "score": 0.92}
  ],
  "execution_time_ms": 3245
}
```

### Escalation Management

```bash
# Get pending escalations
GET /api/v1/escalations?status=pending

# Get specific escalation
GET /api/v1/escalation/{escalation_id}

# Resolve escalation
POST /api/v1/escalation/{escalation_id}/resolve
{
  "human_response": "Here's how to reset your password...",
  "agent_name": "John Smith",
  "feedback_rating": 5
}

# Get statistics
GET /api/v1/stats
```

## 🎯 Workflow Logic

### Decision Tree

```
Query arrives
    ↓
Retrieve relevant chunks
    ├─ Confidence >= 0.8? → Generate response
    ├─ Confidence 0.6-0.8? → Generate with disclaimer
    └─ Confidence < 0.6? → Escalate
        ├─ No chunks found? → Escalate
        ├─ Complex query? → Escalate
        └─ Out of scope? → Escalate
    ↓
Response or Escalation ID
```

### Escalation Reasons

- `low_confidence`: Retrieval confidence < 60%
- `no_relevant_chunks`: No chunks above similarity threshold
- `requires_review`: Complex multi-intent query
- `llm_error`: LLM service unavailable
- `out_of_scope`: Query outside knowledge base domain

## 🔧 Configuration

Edit `config.py` for system parameters:

```python
# Chunking
CHUNK_SIZE = 1024  # tokens
CHUNK_OVERLAP = 128  # tokens

# Retrieval
RETRIEVAL_TOP_K = 5
RETRIEVAL_SCORE_THRESHOLD = 0.6

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.80
ESCALATION_THRESHOLD = 0.60

# LLM
LLM_TEMPERATURE = 0.3  # Low for factual consistency
LLM_MAX_TOKENS = 500
```

## 📊 System Metrics

The system tracks:

- **Total queries**: Number processed
- **Escalation rate**: % of queries escalated
- **Average confidence**: Mean retrieval confidence
- **Resolution time**: Time to escalation resolution
- **Feedback ratings**: Agent performance via customer ratings
- **Reason breakdown**: Common escalation reasons

Get stats via:

```bash
python cli.py stats
```

Or API:

```bash
GET /api/v1/stats
```

## 🧪 Testing

Run sample queries:

```bash
# Password reset (in-scope)
python cli.py query "How do I reset my password?"

# Billing (in-scope if FAQ covers it)
python cli.py query "What's your refund policy?"

# Out of scope
python cli.py query "What is quantum computing?"
```

## 📈 Scaling Considerations

### Current Architecture
- **Max throughput**: ~10 QPS (local)
- **Latency**: ~3-5 seconds per query
- **Cost**: Low (OpenAI embeddings + API)

### Scale to Production

**1000 QPS requires**:
- Distributed FastAPI servers (Kubernetes)
- Pinecone or Weaviate for vector DB
- Redis for caching
- Async job queue (Celery/Bull)
- Monitoring (Prometheus + Grafana)

See `docs/HLD.md` for detailed scalability analysis.

## 📝 Documentation

- **[HLD.md](docs/HLD.md)**: High-level architecture, design decisions, system overview
- **[LLD.md](docs/LLD.md)**: Low-level implementation details, data structures, module design
- **[TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)**: Complete technical guide, workflow explanation, testing strategy

## 🎓 Learning Path

1. **Read**: `docs/HLD.md` - Understand system architecture
2. **Review**: `docs/LLD.md` - Study implementation details
3. **Explore**: `src/` - Examine actual code
4. **Run**: `python cli.py upload data/sample.pdf` - Test locally
5. **Deploy**: `python api.py` - Start API server

## 🔐 Environment Variables

Create `.env` file:

```
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_PROVIDER=openai
DEBUG_MODE=False
LOG_LEVEL=INFO
API_PORT=8000
```

## 🚨 Error Handling

The system gracefully handles:

- **Missing PDFs**: Logs error and skips
- **Embedding API failures**: Falls back to local embeddings or escalates
- **Vector store unavailable**: Uses keyword search fallback
- **LLM service error**: Escalates immediately
- **Invalid queries**: Returns helpful error message

## 🔄 HITL Workflow

1. **Escalation Triggered**: Query confidence < 60% or complex query detected
2. **Create Record**: Store query + retrieved context + AI response
3. **Notify Team**: Support agent receives notification
4. **Agent Reviews**: See query, retrieved chunks, AI response
5. **Agent Responds**: Approve, modify, or override response
6. **User Notified**: Response sent with source attribution
7. **Feedback Loop**: Track overrides to improve system

## 📦 Project Structure

```
rag-support-system/
├── src/
│   ├── document_processor.py      # PDF loading
│   ├── chunking.py                 # Document chunking
│   ├── embeddings.py               # Embedding providers
│   ├── vector_store.py             # ChromaDB wrapper
│   ├── retrieval.py                # Retrieval logic
│   ├── query_processor.py          # Intent detection
│   ├── llm_client.py               # LLM integration
│   ├── hitl.py                     # Escalation management
│   └── graph_engine.py             # LangGraph workflow
├── docs/
│   ├── HLD.md                      # High-level design
│   ├── LLD.md                      # Low-level design
│   └── TECHNICAL_DOCUMENTATION.md  # Full technical guide
├── data/
│   └── (knowledge base PDFs)
├── tests/
│   └── (test files)
├── config.py                       # System configuration
├── cli.py                          # CLI interface
├── api.py                          # FastAPI server
└── requirements.txt                # Dependencies
```

## 🎯 Future Enhancements

- [ ] Multi-document source selection
- [ ] User feedback loop for continuous learning
- [ ] Session memory for follow-up questions
- [ ] A/B testing framework
- [ ] Kubernetes deployment templates
- [ ] Real-time analytics dashboard
- [ ] Advanced reranking with cross-encoders

## 📝 License

This is an educational project for the GenAI Internship.

## 👥 Support

For issues or questions, refer to documentation in `docs/` folder or check code comments.

---

**Built with attention to system design, production readiness, and learning.**
