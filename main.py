#!/usr/bin/env python3
"""
main.py — RAG-Based Customer Support Assistant
Interactive CLI interface for querying the knowledge base.

Usage:
    python main.py                # Start interactive session
    python main.py --query "..."  # Single query mode
    python main.py --log          # View escalation log
    python main.py --demo         # Run demo with sample queries
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
#  COLORS (graceful fallback)
# ─────────────────────────────────────────────
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    BLUE   = Fore.BLUE
    GREEN  = Fore.GREEN
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    RED    = Fore.RED
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
    DIM    = Style.DIM
except ImportError:
    BLUE = GREEN = YELLOW = CYAN = RED = BOLD = RESET = DIM = ""


# ─────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────

BANNER = f"""
{BLUE}{BOLD}
╔══════════════════════════════════════════════════════╗
║   🤖  RAG Customer Support Assistant  v1.0           ║
║   Powered by: LangGraph + ChromaDB + Llama 3         ║
║   with Human-in-the-Loop (HITL) Escalation           ║
╚══════════════════════════════════════════════════════╝
{RESET}"""

HELP_TEXT = f"""
{CYAN}Commands:{RESET}
  {BOLD}Type your question{RESET}  →  Get an answer from the knowledge base
  {BOLD}help{RESET}               →  Show this help text
  {BOLD}log{RESET}                →  View recent escalation log
  {BOLD}clear{RESET}              →  Clear the screen
  {BOLD}quit / exit{RESET}        →  Exit the assistant
"""

DEMO_QUERIES = [
    "What is your return policy?",
    "How do I track my order?",
    "What payment methods do you accept?",
    "Can I get a refund for a damaged product?",
    "How do I contact customer service?",
]


# ─────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────

def print_response(result: dict):
    """Pretty-print the final response and metadata."""
    response   = result.get("final_response", "No response generated.")
    confidence = result.get("confidence_label", "?")
    escalated  = result.get("escalation_flag", False)
    latency    = result.get("total_time_ms", 0)
    sources    = result.get("sources", [])

    print(f"\n{BOLD}{'─'*55}{RESET}")

    if escalated and result.get("human_response"):
        print(f"{YELLOW}{BOLD}🙋 Human Agent Response:{RESET}")
    elif escalated:
        print(f"{YELLOW}{BOLD}📋 Escalation Response:{RESET}")
    else:
        color = GREEN if confidence == "HIGH" else (YELLOW if confidence == "MEDIUM" else RED)
        print(f"{color}{BOLD}🤖 Assistant [{confidence} Confidence]:{RESET}")

    print()
    # Print response with word-wrapping at 70 chars
    for line in response.split("\n"):
        print(f"  {line}")

    print(f"\n{DIM}  ⏱  {latency:.0f}ms  |  Sources: {len(sources)}{RESET}")
    print(f"{BOLD}{'─'*55}{RESET}\n")


def print_status_check():
    """Show system status on startup."""
    from src.vector_store import index_exists
    from src.config import config

    print(f"{CYAN}System Status:{RESET}")

    # KB Index
    if index_exists():
        print(f"  {GREEN}✅ Knowledge Base{RESET}  — Index loaded from {config.CHROMA_PERSIST_DIR}")
    else:
        print(f"  {RED}❌ Knowledge Base{RESET}  — Not indexed! Run: python ingest.py --pdf your_doc.pdf")

    # API Key
    if config.GROQ_API_KEY and config.GROQ_API_KEY != "your_groq_api_key_here":
        print(f"  {GREEN}✅ LLM (Groq){RESET}     — API key configured")
    else:
        print(f"  {YELLOW}⚠️  LLM (Groq){RESET}    — No API key. Set GROQ_API_KEY in .env")
        print(f"     {DIM}→  Get free key: https://console.groq.com{RESET}")

    print()

    # Config warnings
    warnings = config.validate()
    for w in warnings:
        print(f"  {YELLOW}⚠  {w}{RESET}")
    if warnings:
        print()


# ─────────────────────────────────────────────
#  CORE QUERY RUNNER
# ─────────────────────────────────────────────

_graph = None

def get_graph():
    """Lazy-load the LangGraph compiled graph (cached)."""
    global _graph
    if _graph is None:
        from src.graph_engine import build_graph
        _graph = build_graph()
    return _graph


def ask(query: str) -> dict:
    """Run a single query through the RAG graph."""
    from src.graph_engine import run_query
    return run_query(query, graph=get_graph())


# ─────────────────────────────────────────────
#  INTERACTIVE SESSION
# ─────────────────────────────────────────────

def interactive_session():
    """Main interactive loop."""
    print(BANNER)
    print_status_check()
    print(HELP_TEXT)

    print(f"{CYAN}Ask me anything about our products and services.{RESET}")
    print(f"{DIM}Type 'help' for commands, 'quit' to exit.{RESET}\n")

    while True:
        try:
            query = input(f"{BOLD}{BLUE}You > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Goodbye! 👋{RESET}\n")
            break

        if not query:
            continue

        query_lower = query.lower()

        # Built-in commands
        if query_lower in ("quit", "exit", "q", "bye"):
            print(f"\n{YELLOW}Thank you for using the support assistant. Goodbye! 👋{RESET}\n")
            break

        elif query_lower == "help":
            print(HELP_TEXT)
            continue

        elif query_lower == "log":
            from src.hitl_handler import view_escalation_log
            view_escalation_log()
            continue

        elif query_lower == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)
            continue

        elif query_lower == "demo":
            run_demo()
            continue

        # Process query
        print(f"\n{DIM}Searching knowledge base...{RESET}")
        try:
            result = ask(query)
            print_response(result)
        except Exception as e:
            print(f"\n{RED}Error processing query: {e}{RESET}")
            print(f"{YELLOW}Please check your configuration and try again.{RESET}\n")


# ─────────────────────────────────────────────
#  DEMO MODE
# ─────────────────────────────────────────────

def run_demo():
    """Run a set of predefined demo queries to showcase the system."""
    print(f"\n{BOLD}{CYAN}Running demo with {len(DEMO_QUERIES)} sample queries...{RESET}\n")

    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n{BOLD}Demo Query {i}/{len(DEMO_QUERIES)}:{RESET}")
        print(f"{CYAN}Q: {query}{RESET}")

        try:
            result = ask(query)
            print_response(result)
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}\n")

        if i < len(DEMO_QUERIES):
            input(f"{DIM}Press Enter for next query...{RESET}")

    print(f"\n{GREEN}Demo complete!{RESET}\n")


# ─────────────────────────────────────────────
#  SINGLE QUERY MODE
# ─────────────────────────────────────────────

def single_query_mode(query: str):
    """Non-interactive single query — useful for scripting/testing."""
    print(BANNER)
    print(f"{BOLD}Query:{RESET} {query}\n")
    print(f"{DIM}Processing...{RESET}\n")
    result = ask(query)
    print_response(result)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG-Based Customer Support Assistant"
    )
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Run a single query and exit")
    parser.add_argument("--log", "-l", action="store_true",
                        help="View escalation log and exit")
    parser.add_argument("--demo", "-d", action="store_true",
                        help="Run demo queries and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.log:
        from src.hitl_handler import view_escalation_log
        view_escalation_log()

    elif args.query:
        single_query_mode(args.query)

    elif args.demo:
        print(BANNER)
        run_demo()

    else:
        interactive_session()
