"""
document_processor.py — PDF loading and text cleaning
Handles: loading → noise removal → structured Document objects
"""

import re
import os
from typing import List
from langchain_core.documents import Document
from src.config import config


def _simple_splitter(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Built-in fallback chunker: respects paragraphs, then sentences."""
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                # Start new chunk with overlap from end of previous
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = (overlap_text + "\n\n" + para).strip()
            else:
                # Para itself is large — split by character
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
                current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c.strip()) > 20]


# ─────────────────────────────────────────────
#  PDF LOADER
# ─────────────────────────────────────────────

def load_pdf(file_path: str = None) -> List[Document]:
    """Load a PDF file page by page into LangChain Document objects."""
    path = file_path or config.PDF_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PDF not found at '{path}'. "
            "Place your knowledge base PDF at the configured path."
        )

    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        documents = []

        for page_num, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = _clean_text(raw_text)

            if len(cleaned.strip()) < 20:
                continue

            documents.append(Document(
                page_content=cleaned,
                metadata={
                    "source": os.path.basename(path),
                    "page": page_num + 1,
                    "total_pages": len(reader.pages),
                }
            ))

        print(f"[DocumentProcessor] Loaded {len(documents)} pages from '{path}'")
        return documents

    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {e}") from e


# ─────────────────────────────────────────────
#  CHUNKING
# ─────────────────────────────────────────────

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    # Try LangChain splitter first
    splitter = None
    for mod in ["langchain.text_splitter", "langchain_text_splitters",
                "langchain_community.document_loaders"]:
        try:
            import importlib
            m = importlib.import_module(mod)
            cls = getattr(m, "RecursiveCharacterTextSplitter", None)
            if cls:
                splitter = cls(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                break
        except Exception:
            continue

    all_chunks = []
    idx = 0

    for doc in documents:
        if splitter:
            raw_chunks = splitter.split_text(doc.page_content)
        else:
            raw_chunks = _simple_splitter(doc.page_content, chunk_size, overlap)

        for text in raw_chunks:
            if not text.strip():
                continue
            all_chunks.append(Document(
                page_content=text,
                metadata={
                    **doc.metadata,
                    "chunk_id": idx,
                    "char_count": len(text),
                }
            ))
            idx += 1

    # Add total chunk count to all metadata
    for chunk in all_chunks:
        chunk.metadata["chunk_total"] = len(all_chunks)

    print(f"[DocumentProcessor] Created {len(all_chunks)} chunks "
          f"(avg size: {sum(len(c.page_content) for c in all_chunks)//max(len(all_chunks),1)} chars)")
    return all_chunks


# ─────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Remove noise: excess whitespace, page artifacts, repeated dashes."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'-{3,}', '---', text)
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
#  FULL INGESTION PIPELINE
# ─────────────────────────────────────────────

def ingest_pdf(file_path: str = None) -> List[Document]:
    """End-to-end: load PDF → clean → chunk."""
    documents = load_pdf(file_path)
    chunks = chunk_documents(documents)
    return chunks
