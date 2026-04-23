"""
hitl_handler.py — Human-in-the-Loop escalation module
Handles: escalation detection → logging → human input capture → response injection
"""

import json
import os
import signal
import uuid
from datetime import datetime, timezone
from typing import Optional
from src.config import config

# ANSI colors (via colorama or fallback)
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    RED = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL
except ImportError:
    RED = YELLOW = CYAN = GREEN = BOLD = RESET = ""


# ─────────────────────────────────────────────
#  ESCALATION REASONS
# ─────────────────────────────────────────────

class EscalationReason:
    LOW_CONFIDENCE = "low_confidence"
    NO_CHUNKS = "no_relevant_chunks"
    MISSING_CONTEXT = "missing_context_in_answer"
    SENSITIVE_INTENT = "sensitive_or_complex_intent"
    LLM_FAILURE = "llm_failure"
    COMPLEX_QUERY = "complex_multi_part_query"


# ─────────────────────────────────────────────
#  INTENT CLASSIFIER
# ─────────────────────────────────────────────

def classify_intent(query: str) -> str:
    """
    Lightweight intent classification using keyword matching.

    Returns:
        'sensitive' — keywords suggest legal/dispute/complaint
        'complex'   — query is unusually long or multi-part
        'standard'  — normal customer query
    """
    query_lower = query.lower()

    # Check sensitive keywords
    for kw in config.SENSITIVE_KEYWORDS:
        if kw in query_lower:
            return "sensitive"

    # Check complexity
    word_count = len(query.split())
    question_count = query.count("?")
    if word_count > 80 or question_count >= 3:
        return "complex"

    return "standard"


# ─────────────────────────────────────────────
#  ESCALATION DECISION
# ─────────────────────────────────────────────

def should_escalate(
    confidence: float,
    chunk_count: int,
    answer: str,
    intent: str,
) -> tuple[bool, str]:
    """
    Determine if the query should be escalated to a human agent.

    Returns:
        (should_escalate: bool, reason: str)
    """
    # Sensitive/complex intent → always escalate
    if intent == "sensitive":
        return True, EscalationReason.SENSITIVE_INTENT

    if intent == "complex":
        return True, EscalationReason.COMPLEX_QUERY

    # No relevant chunks retrieved
    if chunk_count == 0:
        return True, EscalationReason.NO_CHUNKS

    # LLM explicitly says it doesn't know
    if "don't have sufficient information" in answer.lower():
        return True, EscalationReason.MISSING_CONTEXT

    # Low confidence score
    if confidence < config.CONFIDENCE_THRESHOLD:
        return True, EscalationReason.LOW_CONFIDENCE

    return False, ""


# ─────────────────────────────────────────────
#  ESCALATION LOGGER
# ─────────────────────────────────────────────

def log_escalation(
    query: str,
    chunks: list,
    llm_answer: str,
    confidence: float,
    reason: str,
    escalation_id: str,
    human_response: Optional[str] = None,
    resolved: bool = False,
) -> str:
    """
    Write escalation record to JSON log file.
    Returns the escalation_id.
    """
    os.makedirs(os.path.dirname(config.ESCALATION_LOG), exist_ok=True)

    # Load existing log
    existing = []
    if os.path.exists(config.ESCALATION_LOG):
        try:
            with open(config.ESCALATION_LOG, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    # Build record
    record = {
        "id": escalation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieved_chunks": [c.page_content[:200] + "..." for c in chunks],
        "chunk_sources": [c.metadata.get("source", "?") + f" p.{c.metadata.get('page','?')}" for c in chunks],
        "llm_attempt": llm_answer[:500] if llm_answer else None,
        "confidence_score": round(confidence, 3),
        "escalation_reason": reason,
        "human_response": human_response,
        "resolved": resolved,
    }

    existing.append(record)

    with open(config.ESCALATION_LOG, "w") as f:
        json.dump(existing, f, indent=2)

    return escalation_id


# ─────────────────────────────────────────────
#  HUMAN INPUT WITH TIMEOUT
# ─────────────────────────────────────────────

def _timeout_handler(signum, frame):
    raise TimeoutError("Human response timed out")


def get_human_input(timeout_seconds: int = None) -> Optional[str]:
    """
    Prompt for human agent input with a timeout.
    Returns None if timeout expires or agent skips.
    """
    timeout = timeout_seconds or config.HITL_TIMEOUT_SECONDS

    print(f"\n{YELLOW}{'─'*60}{RESET}")
    print(f"{BOLD}{YELLOW}⚠  HUMAN AGENT REQUIRED{RESET}")
    print(f"{YELLOW}{'─'*60}{RESET}")
    print(f"{CYAN}You have {timeout} seconds to respond.{RESET}")
    print(f"{CYAN}Type your response and press Enter.{RESET}")
    print(f"{CYAN}Press Enter without typing to trigger auto-response.{RESET}")
    print(f"{YELLOW}{'─'*60}{RESET}\n")

    # Use signal-based timeout (Unix only)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        user_input = input(f"{GREEN}Agent Response > {RESET}").strip()
        signal.alarm(0)  # Cancel alarm
        return user_input if user_input else None
    except (TimeoutError, AttributeError):
        # AttributeError on Windows (no SIGALRM)
        signal.alarm(0) if hasattr(signal, 'SIGALRM') else None
        print(f"\n{RED}[HITL] Response timeout. Auto-message will be sent.{RESET}")
        return None


# ─────────────────────────────────────────────
#  MAIN HITL HANDLER
# ─────────────────────────────────────────────

AUTO_RESPONSE = (
    "Thank you for reaching out. Your query has been escalated to our support team. "
    "A human agent will contact you within 24 hours with a detailed response. "
    "We apologize for any inconvenience."
)


def handle_escalation(
    query: str,
    chunks: list,
    llm_answer: str,
    confidence: float,
    reason: str,
) -> str:
    """
    Full HITL escalation flow:
    1. Generate escalation ID
    2. Display escalation notice with context
    3. Request human input
    4. Log and return final response

    Returns:
        Final response string (human-provided or auto-message)
    """
    escalation_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"

    # ── Display escalation notice ──
    print(f"\n{RED}{'═'*60}{RESET}")
    print(f"{BOLD}{RED}  ESCALATION TRIGGERED — ID: {escalation_id}{RESET}")
    print(f"{RED}{'═'*60}{RESET}")
    print(f"{BOLD}  Customer Query:{RESET}  {query}")
    print(f"{BOLD}  Reason:{RESET}          {reason.replace('_', ' ').title()}")
    print(f"{BOLD}  Confidence:{RESET}      {confidence:.0%}")
    print(f"{BOLD}  Chunks Found:{RESET}    {len(chunks)}")

    if llm_answer and "don't have sufficient" not in llm_answer.lower():
        preview = llm_answer[:200].replace('\n', ' ')
        print(f"{BOLD}  AI Draft:{RESET}        {preview}...")

    print(f"{RED}{'═'*60}{RESET}")

    # ── Get human input ──
    human_response = get_human_input()

    # ── Determine final response ──
    if human_response:
        final_response = f"[Support Agent]\n{human_response}"
        resolved = True
        print(f"\n{GREEN}[HITL] Human response captured. Delivering to customer.{RESET}")
    else:
        final_response = AUTO_RESPONSE
        resolved = False
        print(f"\n{YELLOW}[HITL] Auto-response sent. Ticket {escalation_id} remains open.{RESET}")

    # ── Log everything ──
    log_escalation(
        query=query,
        chunks=chunks,
        llm_answer=llm_answer,
        confidence=confidence,
        reason=reason,
        escalation_id=escalation_id,
        human_response=human_response,
        resolved=resolved,
    )

    return final_response


# ─────────────────────────────────────────────
#  LOG VIEWER
# ─────────────────────────────────────────────

def view_escalation_log(limit: int = 10):
    """Print recent escalation records in a readable format."""
    if not os.path.exists(config.ESCALATION_LOG):
        print("No escalation log found.")
        return

    with open(config.ESCALATION_LOG) as f:
        records = json.load(f)

    records = records[-limit:]  # Most recent N
    print(f"\n{'─'*50}")
    print(f"  ESCALATION LOG — Last {len(records)} entries")
    print(f"{'─'*50}")

    for r in records:
        status = "✅ RESOLVED" if r["resolved"] else "🔴 OPEN"
        print(f"\n  ID:       {r['id']}")
        print(f"  Time:     {r['timestamp'][:19]}")
        print(f"  Query:    {r['query'][:80]}...")
        print(f"  Reason:   {r['escalation_reason']}")
        print(f"  Status:   {status}")

    print(f"\n{'─'*50}")
    total = len(records)
    resolved = sum(1 for r in records if r["resolved"])
    print(f"  Resolved: {resolved}/{total} ({resolved/total*100:.0f}%)")
    print(f"{'─'*50}\n")
