"""
llm_handler.py — LLM integration, prompt engineering, and response parsing
Supports: Groq (primary) → OpenAI (alternate) → Ollama (offline fallback)
"""

import re
import time
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from src.config import config


# ─────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional customer support assistant for a company.
Your job is to answer customer questions accurately using ONLY the provided context.

STRICT RULES:
1. Answer ONLY from the context provided below. Do NOT use general knowledge.
2. If the context does not contain the answer, say exactly: "I don't have sufficient information to answer this question from our knowledge base."
3. Be concise, friendly, and professional.
4. Always cite the source (e.g., "According to our return policy on Page 3...").
5. At the very end of your response, on a new line, write your confidence rating in this exact format:
   CONFIDENCE: HIGH   (use HIGH if answer is clearly in context)
   CONFIDENCE: MEDIUM (use MEDIUM if partially answered)
   CONFIDENCE: LOW    (use LOW if context is thin or incomplete)

Do NOT add any text after the CONFIDENCE line."""


PROMPT_TEMPLATE = """{system_prompt}

---KNOWLEDGE BASE CONTEXT---
{context}
---END CONTEXT---

Customer Question: {query}

Your Answer:"""


# ─────────────────────────────────────────────
#  LLM FACTORY
# ─────────────────────────────────────────────

def get_llm():
    """
    Return configured LLM instance.
    Priority: Groq → OpenAI → Ollama (based on .env config)
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        return _get_groq_llm()
    elif provider == "openai":
        return _get_openai_llm()
    elif provider == "ollama":
        return _get_ollama_llm()
    else:
        print(f"[LLM] Unknown provider '{provider}', defaulting to Groq")
        return _get_groq_llm()


def _get_groq_llm():
    """Groq: ultra-fast inference, free tier available."""
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            groq_api_key=config.GROQ_API_KEY,
            max_tokens=1024,
        )
        print(f"[LLM] Using Groq: {config.LLM_MODEL}")
        return llm
    except Exception as e:
        print(f"[LLM] Groq initialization failed: {e}. Trying Ollama fallback.")
        return _get_ollama_llm()


def _get_openai_llm():
    """OpenAI: best quality, requires API key and billing."""
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=config.LLM_TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY,
            max_tokens=1024,
        )
        print("[LLM] Using OpenAI: gpt-3.5-turbo")
        return llm
    except Exception as e:
        print(f"[LLM] OpenAI initialization failed: {e}")
        return _get_ollama_llm()


def _get_ollama_llm():
    """Ollama: fully offline, no API key needed. Run: ollama pull llama3"""
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3", temperature=config.LLM_TEMPERATURE)
        print("[LLM] Using Ollama: llama3 (offline mode)")
        return llm
    except Exception as e:
        raise RuntimeError(
            f"All LLM options failed. Last error: {e}\n"
            "Please set GROQ_API_KEY in .env or install Ollama."
        )


# ─────────────────────────────────────────────
#  PROMPT BUILDER
# ─────────────────────────────────────────────

def build_prompt(query: str, chunks: List[Document]) -> str:
    """
    Construct the full augmented prompt from query + retrieved chunks.
    Each chunk is labeled with its source and page for attribution.
    """
    if not chunks:
        context = "[No relevant information found in the knowledge base]"
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "?")
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{chunk.page_content}"
            )
        context = "\n\n".join(context_parts)

    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context,
        query=query,
    )


# ─────────────────────────────────────────────
#  LLM CALL WITH RETRY
# ─────────────────────────────────────────────

def call_llm(prompt: str, llm=None, max_retries: int = 3) -> str:
    """
    Call the LLM with exponential backoff retry logic.
    Returns raw LLM response string.
    """
    if llm is None:
        llm = get_llm()

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            from langchain_core.messages import HumanMessage
            # Support both chat models and completion models
            if hasattr(llm, "invoke"):
                response = llm.invoke(prompt)
                # Handle ChatModel response (AIMessage) vs string response
                if hasattr(response, "content"):
                    return response.content
                return str(response)
            else:
                return llm(prompt)
        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt
            print(f"[LLM] Attempt {attempt}/{max_retries} failed: {e}. "
                  f"Retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts. Last error: {last_error}")


# ─────────────────────────────────────────────
#  RESPONSE PARSER
# ─────────────────────────────────────────────

def parse_response(raw_response: str) -> Tuple[str, float, str]:
    """
    Parse the LLM's raw response into (answer, confidence_score, confidence_label).

    Returns:
        answer: Cleaned response text without the confidence line
        confidence_score: Float 0.0–1.0
        confidence_label: "HIGH", "MEDIUM", or "LOW"
    """
    # Extract confidence label
    confidence_pattern = r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)'
    match = re.search(confidence_pattern, raw_response, re.IGNORECASE)

    if match:
        label = match.group(1).upper()
        # Remove the confidence line from the answer
        answer = raw_response[:match.start()].strip()
    else:
        # No confidence tag found — treat as LOW
        label = "LOW"
        answer = raw_response.strip()

    # Map label to numeric score
    score_map = {"HIGH": 0.9, "MEDIUM": 0.65, "LOW": 0.3}
    score = score_map.get(label, 0.3)

    # Check for explicit "insufficient information" statements → LOW
    insufficient_phrases = [
        "don't have sufficient information",
        "not mentioned in",
        "not available in",
        "cannot find",
        "not provided in the context",
    ]
    if any(phrase in answer.lower() for phrase in insufficient_phrases):
        score = min(score, 0.35)  # Cap at LOW range
        label = "LOW"

    return answer, score, label


# ─────────────────────────────────────────────
#  CONVENIENCE: FULL GENERATE FUNCTION
# ─────────────────────────────────────────────

def generate_response(
    query: str,
    chunks: List[Document],
    llm=None
) -> Tuple[str, float, str, str]:
    """
    End-to-end: build prompt → call LLM → parse response.

    Returns:
        (answer, confidence_score, confidence_label, augmented_prompt)
    """
    prompt = build_prompt(query, chunks)
    raw_response = call_llm(prompt, llm)
    answer, score, label = parse_response(raw_response)
    return answer, score, label, prompt
