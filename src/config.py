"""
config.py — Centralized configuration management
Loads settings from .env file with safe defaults for all parameters.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Single source of truth for all system configuration."""

    # --- LLM ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3-8b-8192")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # --- Retrieval ---
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "4"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # --- Confidence / HITL ---
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    HITL_TIMEOUT_SECONDS: int = int(os.getenv("HITL_TIMEOUT_SECONDS", "60"))

    # --- Paths ---
    PDF_PATH: str = os.getenv("PDF_PATH", "data/knowledge_base.pdf")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    ESCALATION_LOG: str = os.getenv("ESCALATION_LOG", "logs/escalation_log.json")

    # --- Embedding ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "rag_support_kb")

    # --- Sensitive keywords for HITL escalation ---
    SENSITIVE_KEYWORDS: list = [
        "sue", "lawyer", "legal action", "court", "fraud", "scam",
        "refund dispute", "escalate", "manager", "supervisor",
        "complaint", "threatening", "harassment", "useless", "pathetic"
    ]

    @classmethod
    def validate(cls) -> list[str]:
        """Return list of validation warnings (not errors — system has fallbacks)."""
        warnings = []
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY not set. LLM calls will fail unless using Ollama fallback.")
        if not os.path.exists(cls.PDF_PATH):
            warnings.append(f"PDF not found at: {cls.PDF_PATH}. Run ingest after placing your PDF.")
        return warnings


config = Config()
