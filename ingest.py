#!/usr/bin/env python3
"""
ingest.py — One-time knowledge base ingestion script
Usage:
    python ingest.py                         # Uses PDF_PATH from .env
    python ingest.py --pdf path/to/doc.pdf   # Specify a custom PDF path
    python ingest.py --reset                 # Clear existing index and re-ingest
"""

import sys
import os
import argparse
import time

# Make src/ importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG knowledge base")
    parser.add_argument("--pdf", type=str, default=None,
                        help=f"Path to PDF file (default: {config.PDF_PATH})")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing ChromaDB index before ingesting")
    return parser.parse_args()


def reset_index():
    """Delete existing ChromaDB persistence directory."""
    import shutil
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
        print(f"[Ingest] Cleared existing index at '{config.CHROMA_PERSIST_DIR}'")
    else:
        print("[Ingest] No existing index to clear.")


def run_ingestion(pdf_path: str = None):
    """Full ingestion pipeline: PDF → chunks → embeddings → ChromaDB."""
    from src.document_processor import ingest_pdf
    from src.vector_store import get_embedding_model, build_vector_store

    pdf = pdf_path or config.PDF_PATH

    print("\n" + "═" * 55)
    print("  RAG KNOWLEDGE BASE INGESTION")
    print("═" * 55)
    print(f"  PDF Path:    {pdf}")
    print(f"  Chunk Size:  {config.CHUNK_SIZE} chars")
    print(f"  Overlap:     {config.CHUNK_OVERLAP} chars")
    print(f"  Embedding:   {config.EMBEDDING_MODEL}")
    print(f"  DB Path:     {config.CHROMA_PERSIST_DIR}")
    print("═" * 55 + "\n")

    # Step 1: Load and chunk PDF
    print("Step 1/3: Loading and chunking PDF...")
    t0 = time.time()
    chunks = ingest_pdf(pdf)
    print(f"          → {len(chunks)} chunks created in {time.time()-t0:.1f}s\n")

    # Step 2: Load embedding model
    print("Step 2/3: Loading embedding model...")
    t1 = time.time()
    embeddings = get_embedding_model()
    print(f"          → Model ready in {time.time()-t1:.1f}s\n")

    # Step 3: Build and persist vector store
    print("Step 3/3: Embedding chunks and storing in ChromaDB...")
    t2 = time.time()
    vector_store = build_vector_store(chunks, embeddings)
    print(f"          → Vector store built in {time.time()-t2:.1f}s\n")

    total = time.time() - t0
    print("═" * 55)
    print(f"  ✅ INGESTION COMPLETE")
    print(f"  Chunks indexed: {len(chunks)}")
    print(f"  Total time:     {total:.1f}s")
    print("═" * 55)
    print("\n  You can now run: python main.py\n")


if __name__ == "__main__":
    args = parse_args()

    if args.reset:
        reset_index()

    run_ingestion(pdf_path=args.pdf)
