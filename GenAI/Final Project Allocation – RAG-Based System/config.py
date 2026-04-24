"""
Configuration module for RAG system
Loads environment variables and sets system defaults
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Directories
# `config.py` lives at the project root, so its parent is the repo base.
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# ChromaDB Configuration
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "rag_knowledge_base"

# Embedding Configuration
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # "openai" or "local"

# Chunking Configuration
CHUNK_SIZE = 1024  # tokens
CHUNK_OVERLAP = 128  # tokens
CHUNKING_STRATEGY = "semantic"  # "semantic" or "fixed"

# Retrieval Configuration
RETRIEVAL_TOP_K = 5
RETRIEVAL_SCORE_THRESHOLD = 0.6
RETRIEVAL_WITH_RERANKING = False

# Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.80
LOW_CONFIDENCE_THRESHOLD = 0.60
ESCALATION_THRESHOLD = 0.60

# LLM Configuration
LLM_TEMPERATURE = 0.3  # Low for factual consistency
LLM_MAX_TOKENS = 500
LLM_TIMEOUT = 30  # seconds

# HITL Configuration
ENABLE_HITL = True
HITL_DB_URL = os.getenv("HITL_DB_URL", "sqlite:///./rag_escalations.db")

# FastAPI Configuration
API_PORT = int(os.getenv("API_PORT", 8000))
API_HOST = os.getenv("API_HOST", "127.0.0.1")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Query Processing
INTENT_CATEGORIES = {
    "password_reset": ["reset password", "forgot password", "can't login", "account locked"],
    "billing": ["charge", "invoice", "payment", "refund", "bill", "subscription"],
    "technical_support": ["error", "bug", "not working", "crash", "broken"],
    "account": ["profile", "email", "settings", "delete account", "name change"],
    "general": []  # Catch-all
}

# Response Settings
INCLUDE_SOURCE_CITATIONS = True
INCLUDE_CONFIDENCE_SCORE = True

