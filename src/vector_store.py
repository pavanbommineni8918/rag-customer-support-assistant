"""
vector_store.py — Embedding generation and ChromaDB management
Handles: embedding model setup → vector store creation → retrieval
"""

import os
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from src.config import config


# ─────────────────────────────────────────────
#  EMBEDDING MODEL
# ─────────────────────────────────────────────

def get_embedding_model():
    """
    Load and return the sentence-transformer embedding model.
    Uses all-MiniLM-L6-v2: 384-dim, fast, accurate, fully offline.
    Falls back to a simple TF-IDF-style approach if model unavailable.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{config.EMBEDDING_MODEL}",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"[VectorStore] Loaded embedding model: {config.EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        print(f"[VectorStore] WARNING: HuggingFace embeddings unavailable ({e}). "
              "Using fallback embeddings.")
        return _get_fallback_embeddings()


def _get_fallback_embeddings():
    """Simple fallback when sentence-transformers is not available."""
    from langchain_community.embeddings import FakeEmbeddings
    print("[VectorStore] Using FakeEmbeddings (for testing only — no real semantic search)")
    return FakeEmbeddings(size=384)


# ─────────────────────────────────────────────
#  VECTOR STORE — BUILD (ingestion time)
# ─────────────────────────────────────────────

def build_vector_store(chunks: List[Document], embeddings=None):
    """
    Create a ChromaDB vector store from document chunks.
    Persists to disk at CHROMA_PERSIST_DIR.
    """
    import chromadb
    from langchain_community.vectorstores import Chroma

    if embeddings is None:
        embeddings = get_embedding_model()

    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)

    # Build vector store (embeds all chunks, stores to disk)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION,
    )

    count = vector_store._collection.count()
    print(f"[VectorStore] Stored {count} vectors in ChromaDB at '{config.CHROMA_PERSIST_DIR}'")
    return vector_store


# ─────────────────────────────────────────────
#  VECTOR STORE — LOAD (query time)
# ─────────────────────────────────────────────

def load_vector_store(embeddings=None):
    """
    Load an existing ChromaDB vector store from disk.
    Raises RuntimeError if no index exists (run ingest first).
    """
    from langchain_community.vectorstores import Chroma

    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        raise RuntimeError(
            f"No ChromaDB index found at '{config.CHROMA_PERSIST_DIR}'. "
            "Run: python ingest.py --pdf your_document.pdf"
        )

    if embeddings is None:
        embeddings = get_embedding_model()

    vector_store = Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=config.CHROMA_COLLECTION,
    )

    count = vector_store._collection.count()
    if count == 0:
        raise RuntimeError("ChromaDB collection is empty. Run ingest again.")

    print(f"[VectorStore] Loaded ChromaDB with {count} vectors")
    return vector_store


# ─────────────────────────────────────────────
#  RETRIEVER
# ─────────────────────────────────────────────

def retrieve_chunks(
    query: str,
    vector_store,
    k: int = None,
    threshold: float = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k most relevant chunks for a given query.

    Args:
        query: Natural language question from the user
        vector_store: Loaded ChromaDB vector store
        k: Number of chunks to retrieve (default from config)
        threshold: Max distance threshold — higher distance = less similar (default from config)

    Returns:
        List of (Document, similarity_score) sorted by relevance (best first)
        Score: 0.0 = identical, 1.0 = completely different
    """
    k = k or config.TOP_K_CHUNKS
    threshold = threshold or config.SIMILARITY_THRESHOLD

    try:
        results = vector_store.similarity_search_with_score(query, k=k)
    except Exception as e:
        print(f"[VectorStore] Retrieval error: {e}")
        return []

    # Filter out low-relevance chunks (high distance score = low similarity)
    filtered = [
        (doc, score) for doc, score in results
        if score <= threshold
    ]
    if not filtered:
     best_score = results[0][1] if results else None
    formatted_score = f"{best_score:.3f}" if best_score is not None else "N/A"

    print(
        f"[VectorStore] No chunks passed threshold {threshold}. "
        f"Best score was: {formatted_score}"
    )

    return filtered


# ─────────────────────────────────────────────
#  INDEX STATUS CHECK
# ─────────────────────────────────────────────

def index_exists() -> bool:
    """Check if a ChromaDB index has been built."""
    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        return False
    try:
        from langchain_community.vectorstores import Chroma
        embeddings = _get_fallback_embeddings()  # Cheap check — no model load
        vs = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=config.CHROMA_COLLECTION,
        )
        return vs._collection.count() > 0
    except Exception:
        return False
