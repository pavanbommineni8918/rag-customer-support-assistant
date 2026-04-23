"""
graph_engine.py — LangGraph workflow orchestration
Defines: RAGState → all nodes → conditional routing → compiled graph
"""

from typing import TypedDict, List, Optional, Annotated
from langchain_core.documents import Document
from src.config import config

# ─────────────────────────────────────────────
#  STATE DEFINITION
# ─────────────────────────────────────────────

class RAGState(TypedDict):
    """
    The single state object that flows through all LangGraph nodes.
    Every field is populated by one or more nodes.
    """
    # Input
    query: str                              # Original user query
    cleaned_query: str                      # Preprocessed query text
    intent: str                             # standard | sensitive | complex

    # Retrieval
    retrieved_chunks: List[Document]        # Top-k relevant chunks
    retrieval_scores: List[float]           # Similarity scores (lower = better)

    # Generation
    augmented_prompt: str                   # Full prompt sent to LLM
    raw_llm_response: str                   # Unprocessed LLM output
    answer: str                             # Cleaned answer text
    confidence: float                       # 0.0 – 1.0 confidence score
    confidence_label: str                   # HIGH | MEDIUM | LOW

    # Routing
    escalation_flag: bool                   # True → route to HITL
    escalation_reason: str                  # Reason for escalation

    # Sources
    sources: List[str]                      # "source.pdf p.3" style attribution

    # HITL
    human_response: Optional[str]           # Human agent response text
    final_response: str                     # Final answer delivered to user

    # Metadata
    total_time_ms: Optional[float]          # End-to-end latency in ms


# ─────────────────────────────────────────────
#  GRAPH NODES
# ─────────────────────────────────────────────

def input_node(state: RAGState) -> RAGState:
    """
    Node 1: Validate, clean, and classify the user query.
    Outputs: cleaned_query, intent
    """
    import re
    from src.hitl_handler import classify_intent

    query = state["query"].strip()

    # Basic sanitization
    cleaned = re.sub(r'\s+', ' ', query)          # Normalize whitespace
    cleaned = cleaned[:500]                         # Hard cap at 500 chars

    intent = classify_intent(cleaned)

    print(f"\n[Graph: InputNode] Query received | Intent: {intent}")

    return {
        **state,
        "cleaned_query": cleaned,
        "intent": intent,
        "retrieved_chunks": [],
        "retrieval_scores": [],
        "escalation_flag": False,
        "escalation_reason": "",
        "sources": [],
        "human_response": None,
        "final_response": "",
        "augmented_prompt": "",
        "raw_llm_response": "",
        "answer": "",
        "confidence": 0.0,
        "confidence_label": "LOW",
    }


def retrieval_node(state: RAGState) -> RAGState:
    """
    Node 2: Embed the query and retrieve top-k relevant chunks from ChromaDB.
    Outputs: retrieved_chunks, retrieval_scores, sources
    """
    from src.vector_store import retrieve_chunks, load_vector_store

    query = state["cleaned_query"]

    try:
        vector_store = load_vector_store()
        results = retrieve_chunks(query, vector_store)

        chunks = [doc for doc, _ in results]
        scores = [score for _, score in results]
        sources = list({
            f"{c.metadata.get('source', 'KB')} p.{c.metadata.get('page', '?')}"
            for c in chunks
        })

        print(f"[Graph: RetrievalNode] Retrieved {len(chunks)} chunks")

    except RuntimeError as e:
        print(f"[Graph: RetrievalNode] ERROR: {e}")
        chunks, scores, sources = [], [], []

    return {**state, "retrieved_chunks": chunks, "retrieval_scores": scores, "sources": sources}


def processing_node(state: RAGState) -> RAGState:
    """
    Node 3: Build prompt → call LLM → parse response → assess confidence.
    Outputs: augmented_prompt, raw_llm_response, answer, confidence, confidence_label
    """
    from src.llm_handler import generate_response
    from src.hitl_handler import should_escalate

    query = state["cleaned_query"]
    chunks = state["retrieved_chunks"]
    intent = state["intent"]

    # If sensitive/complex intent, skip LLM entirely (save cost) and escalate
    if intent in ("sensitive", "complex"):
        print(f"[Graph: ProcessingNode] Intent '{intent}' — skipping LLM, escalating directly")
        from src.hitl_handler import EscalationReason
        reason = (EscalationReason.SENSITIVE_INTENT if intent == "sensitive"
                  else EscalationReason.COMPLEX_QUERY)
        return {
            **state,
            "answer": "",
            "confidence": 0.1,
            "confidence_label": "LOW",
            "escalation_flag": True,
            "escalation_reason": reason,
        }

    # Call LLM
    try:
        answer, confidence, label, prompt = generate_response(query, chunks)
        print(f"[Graph: ProcessingNode] LLM response received | Confidence: {label} ({confidence:.2f})")
    except RuntimeError as e:
        print(f"[Graph: ProcessingNode] LLM FAILED: {e}")
        from src.hitl_handler import EscalationReason
        return {
            **state,
            "answer": "",
            "confidence": 0.0,
            "confidence_label": "LOW",
            "augmented_prompt": "",
            "raw_llm_response": str(e),
            "escalation_flag": True,
            "escalation_reason": EscalationReason.LLM_FAILURE,
        }

    # Routing decision
    escalate, reason = should_escalate(confidence, len(chunks), answer, intent)

    return {
        **state,
        "augmented_prompt": prompt,
        "raw_llm_response": answer,   # raw before post-processing
        "answer": answer,
        "confidence": confidence,
        "confidence_label": label,
        "escalation_flag": escalate,
        "escalation_reason": reason,
    }


def output_node(state: RAGState) -> RAGState:
    """
    Node 4a: Format and finalize the automated response.
    Outputs: final_response
    """
    answer = state["answer"]
    sources = state["sources"]
    label = state["confidence_label"]

    # Build source attribution footer
    if sources:
        source_text = "\n\n📎 Sources: " + " | ".join(sources)
    else:
        source_text = ""

    confidence_badge = {"HIGH": "✅", "MEDIUM": "⚠️", "LOW": "❓"}.get(label, "")

    final = f"{answer}{source_text}\n\n{confidence_badge} Confidence: {label}"

    print(f"[Graph: OutputNode] Finalizing automated response | Confidence: {label}")

    return {**state, "final_response": final}


def hitl_node(state: RAGState) -> RAGState:
    """
    Node 4b: Escalate to human agent, capture response, inject back.
    Outputs: final_response, human_response
    """
    from src.hitl_handler import handle_escalation

    print(f"[Graph: HITLNode] Escalating — Reason: {state['escalation_reason']}")

    response = handle_escalation(
        query=state["cleaned_query"],
        chunks=state["retrieved_chunks"],
        llm_answer=state.get("answer", ""),
        confidence=state["confidence"],
        reason=state["escalation_reason"],
    )

    return {**state, "final_response": response, "human_response": response}


# ─────────────────────────────────────────────
#  CONDITIONAL ROUTING
# ─────────────────────────────────────────────

def routing_logic(state: RAGState) -> str:
    """
    Conditional edge: decides whether to go to OutputNode or HITLNode.
    Called after ProcessingNode.
    """
    if state.get("escalation_flag"):
        return "hitl"
    return "output"


# ─────────────────────────────────────────────
#  GRAPH BUILDER
# ─────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the LangGraph StateGraph.

    Graph structure:
        START → input → retrieve → process ──→ output → END
                                          └──→ hitl   → END
    """
    from langgraph.graph import StateGraph, END

    g = StateGraph(RAGState)

    # Register nodes
    g.add_node("input",    input_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("process",  processing_node)
    g.add_node("output",   output_node)
    g.add_node("hitl",     hitl_node)

    # Set entry point
    g.set_entry_point("input")

    # Direct edges
    g.add_edge("input",    "retrieve")
    g.add_edge("retrieve", "process")
    g.add_edge("output",   END)
    g.add_edge("hitl",     END)

    # Conditional routing from process
    g.add_conditional_edges(
        "process",
        routing_logic,
        {"output": "output", "hitl": "hitl"},
    )

    compiled = g.compile()
    print("[Graph] LangGraph compiled successfully")
    return compiled


# ─────────────────────────────────────────────
#  CONVENIENCE RUNNER
# ─────────────────────────────────────────────

def run_query(query: str, graph=None) -> dict:
    """
    Run a single query through the full RAG graph.

    Args:
        query: Natural language question string
        graph: Pre-compiled graph (built if not provided)

    Returns:
        Final state dict with 'final_response' and all intermediate fields
    """
    import time

    if graph is None:
        graph = build_graph()

    initial_state: RAGState = {
        "query": query,
        "cleaned_query": "",
        "intent": "standard",
        "retrieved_chunks": [],
        "retrieval_scores": [],
        "augmented_prompt": "",
        "raw_llm_response": "",
        "answer": "",
        "confidence": 0.0,
        "confidence_label": "LOW",
        "escalation_flag": False,
        "escalation_reason": "",
        "sources": [],
        "human_response": None,
        "final_response": "",
        "total_time_ms": None,
    }

    start = time.time()
    final_state = graph.invoke(initial_state)
    elapsed_ms = (time.time() - start) * 1000

    final_state["total_time_ms"] = round(elapsed_ms, 1)
    print(f"[Graph] Query completed in {elapsed_ms:.0f}ms")

    return final_state
