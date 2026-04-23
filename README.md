# RAG-Based Customer Support Assistant
### with LangGraph Workflow Orchestration & Human-in-the-Loop (HITL) Escalation

> **Internship Project | VTU Evaluation**  
> Implements: RAG · LangGraph · ChromaDB · Sentence Transformers · Groq LLM · HITL Escalation

---

## 📌 Project Overview

This system is an **AI-powered customer support assistant** that:

- 📄 Processes a PDF knowledge base (policies, FAQs, manuals)
- 🔍 Retrieves relevant information using semantic embeddings
- 🤖 Generates contextually accurate answers using Llama 3 (via Groq)
- 🗺️ Orchestrates the entire workflow using **LangGraph**
- 🚨 Escalates uncertain or sensitive queries to a **human agent** (HITL)
- 📊 Logs all escalations for audit and continuous improvement

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Workflow Engine               │
│                                                      │
│  InputNode → RetrievalNode → ProcessingNode          │
│                                  │                   │
│                          ┌───────┴────────┐          │
│                          ▼                ▼          │
│                      OutputNode       HITLNode        │
│                          │                │          │
└──────────────────────────┼────────────────┼──────────┘
                           ▼                ▼
                    Final Answer    Human Agent Response
```

**Data Flow:**
```
PDF → Chunk → Embed → ChromaDB        (Offline Ingestion)
Query → Embed → Search → LLM → Route → Respond    (Online Query)
```

---

## 📁 Project Structure

```
rag_project/
├── main.py                  # CLI interface & entry point
├── ingest.py                # PDF ingestion script (run once)
├── create_sample_kb.py      # Generate sample knowledge base PDF
├── requirements.txt         # Python dependencies
├── .env.example             # Environment configuration template
│
├── src/
│   ├── config.py            # Centralized configuration
│   ├── document_processor.py # PDF loading & chunking
│   ├── vector_store.py      # ChromaDB embedding & retrieval
│   ├── llm_handler.py       # LLM integration & prompt engineering
│   ├── hitl_handler.py      # Human-in-the-Loop escalation
│   └── graph_engine.py      # LangGraph nodes, edges, state
│
├── tests/
│   └── test_system.py       # Unit & integration tests
│
├── data/
│   └── knowledge_base.pdf   # Your PDF knowledge base (add here)
│
└── logs/
    └── escalation_log.json  # Auto-created escalation records
```

---

## ⚡ Quick Start

### Step 1: Clone & Install Dependencies

```bash
git clone <your-repo-url>
cd rag_project

pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Get a free Groq API key:** https://console.groq.com  
> Groq provides **free tier** with 14,400 requests/day on Llama 3.

### Step 3: Add Your Knowledge Base

Place your company's PDF in the `data/` directory:
```bash
cp your_document.pdf data/knowledge_base.pdf
```

Or generate a **sample knowledge base** for testing:
```bash
python create_sample_kb.py
```

### Step 4: Ingest the PDF

```bash
python ingest.py
```

This will:
1. Load and clean the PDF
2. Chunk text into 500-character segments  
3. Generate embeddings with `all-MiniLM-L6-v2`
4. Store in ChromaDB (persisted to `./chroma_db/`)

### Step 5: Start the Assistant

```bash
python main.py
```

---

## 💬 Usage

### Interactive Mode (default)
```
python main.py
```

```
You > What is the return policy?
  The standard return window is 30 days from purchase...
  📎 Sources: knowledge_base.pdf p.2
  ✅ Confidence: HIGH

You > I want to sue your company!
  [ESCALATION TRIGGERED]
  Reason: Sensitive Intent
  → Human agent required...
```

### Single Query Mode
```bash
python main.py --query "How do I track my order?"
```

### Demo Mode
```bash
python main.py --demo
```

### View Escalation Log
```bash
python main.py --log
```

### Re-ingest with reset
```bash
python ingest.py --pdf data/new_document.pdf --reset
```

---

## 🧪 Running Tests

```bash
python tests/test_system.py
```

Expected output:
```
Tests: 20 | Passed: 20 | Failed: 0 | Errors: 0
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `LLM_MODEL` | `llama3-8b-8192` | Groq model ID |
| `TOP_K_CHUNKS` | `4` | Chunks retrieved per query |
| `CHUNK_SIZE` | `500` | Chunk character size |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `CONFIDENCE_THRESHOLD` | `0.6` | Below = HITL escalation |
| `HITL_TIMEOUT_SECONDS` | `60` | Human response timeout |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |

---

## 🗺️ LangGraph Flow

```
START
  │
  ▼
[input_node]          Validate + clean query + classify intent
  │
  ▼
[retrieval_node]      Embed query → ChromaDB similarity search → top-4 chunks
  │
  ▼
[processing_node]     Build prompt → LLM call → parse confidence
  │
  ├── confidence ≥ 0.6 AND chunks found ──→ [output_node] → END
  │
  └── confidence < 0.6 OR no chunks
      OR sensitive/complex intent     ──→ [hitl_node]   → END
```

### HITL Escalation Triggers

| Trigger | Condition |
|---|---|
| Low Confidence | LLM confidence < 0.6 |
| No Relevant Chunks | ChromaDB returns empty results |
| Missing Context | LLM says "I don't have enough information" |
| Sensitive Intent | Keywords: sue, lawyer, complaint, fraud, etc. |
| Complex Query | Query > 80 words or 3+ questions |
| LLM Failure | API timeout, rate limit, network error |

---

## 🔧 Technology Stack

| Component | Technology | Why |
|---|---|---|
| **LLM** | Llama 3 via Groq | Free tier, <300ms latency |
| **Embeddings** | all-MiniLM-L6-v2 | Offline, fast, 384-dim semantic vectors |
| **Vector DB** | ChromaDB | Persistent, LangChain-native, metadata support |
| **Orchestration** | LangGraph | Stateful graph, supports HITL interrupts |
| **PDF Loader** | PyPDF + LangChain | Reliable page-level extraction |
| **Chunker** | RecursiveCharacterTextSplitter | Paragraph-aware, configurable |

---

## 🚀 Future Enhancements

- [ ] Multi-document support with per-document filtering
- [ ] Web UI (Streamlit or React frontend)
- [ ] Feedback loop (thumbs up/down → KB improvement)
- [ ] Conversational memory (multi-turn context)
- [ ] Hybrid retrieval (BM25 + dense reranking)
- [ ] Production deployment (Docker + FastAPI)
- [ ] Fine-tuned LLM on domain-specific data

---

## 📋 Evaluation Criteria Mapping

| Criterion | Implementation Location |
|---|---|
| RAG Implementation | `document_processor.py`, `vector_store.py` |
| LangGraph Workflow | `graph_engine.py` (nodes, edges, state, routing) |
| HITL Escalation | `hitl_handler.py`, `graph_engine.py::hitl_node` |
| Customer Support Use Case | `main.py`, sample knowledge base |
| HLD & LLD | Separate documents (submitted separately) |

---

## 👤 Author

**[BOMMINENI PAVAN KUMAR]**    
6281800244

---

*This system is designed as a production-grade foundation. Every design decision — chunk size, embedding model, routing threshold — is justified by engineering trade-offs documented in the HLD/LLD.*
