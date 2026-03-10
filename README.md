# 🤖 PDF Chatbot — LangGraph RAG with Chainlit + LangSmith

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-RAG_Pipeline-FF6B35?style=for-the-badge)
![Chainlit](https://img.shields.io/badge/Chainlit-Chat_UI-2D9CDB?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00C7B7?style=for-the-badge)
![LangSmith](https://img.shields.io/badge/LangSmith-Observability-7C3AED?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**A production-ready Retrieval-Augmented Generation (RAG) chatbot that ingests PDFs, DOCX, XLSX, and JSON files into FAISS vectorstores and serves answers via a Qwen 7B LLM — with token-level streaming, secure auth, multi-turn memory, sidebar thread history, and full LangSmith observability.**

</div>

---

## 📖 Table of Contents

- [What It Does](#-what-it-does)
- [System Architecture](#-system-architecture)
- [How It Works — Step by Step](#-how-it-works--step-by-step)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Document Ingestion](#-document-ingestion)
- [Multi-Module Setup](#-multi-module-setup)
- [LangSmith Tracing & Observability](#-langsmith-tracing--observability)
- [Token-Level Streaming](#-token-level-streaming)
- [Database Architecture](#-database-architecture)
- [Tools & Retriever Integration](#-tools--retriever-integration)
- [Performance & Latency](#-performance--latency)
- [Docker Deployment](#-docker-deployment)
- [Login Credentials](#-login-credentials)
- [Troubleshooting](#-troubleshooting)
- [API References](#-api-references)

---

## 🎯 What It Does

Given one or more document files (PDF, DOCX, XLSX, JSON), this system:

1. 📥 **Ingests** documents into FAISS vectorstores — chunked, embedded, and indexed locally
2. 🧠 **Routes** each user query through a LangGraph agent that decides: answer directly or call a retriever tool
3. 🔍 **Retrieves** the top-k most semantically similar document chunks via FAISS
4. 💬 **Answers** using Qwen 7B with full multi-turn conversation memory per thread
5. 📡 **Streams** responses token-by-token into the Chainlit UI in real time
6. 📊 **Traces** every LLM call, tool invocation, and document fetch in LangSmith
7. 💾 **Persists** all conversations — resume any thread from the sidebar after closing the browser

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CHAINLIT UI (chainlit_app.py)                    │
│                                                                         │
│   Sidebar                         Chat Panel                            │
│  ┌─────────────────────┐         ┌──────────────────────────────────┐   │
│  │ 🧵 Thread History   │         │  💬 User message input           │   │
│  │  ─────────────────  │         │  📡 Token-by-token streaming     │   │
│  │  > Thread 1 (today) │         │  🔴 Live response rendering      │   │
│  │    Thread 2         │         │                                  │   │
│  │    Thread 3 ...     │         │  Auth  → on_chat_start()         │   │
│  └─────────────────────┘         │  Msg   → on_message()            │   │
│                                  └──────────────────────────────────┘   │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       │
                      app.astream_events(..., version="v2")
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH RAG AGENT (app.py)                         │
│                                                                         │
│   START                                                                 │
│     │                                                                   │
│     ▼                                                                   │
│  ┌────────────────────────────────────────────────┐                     │
│  │              LLM NODE (Qwen 7B)                │                     │
│  │  "Answer directly OR call a retriever tool"    │                     │
│  └────────────────────┬───────────────────────────┘                     │
│                       │                                                 │
│          ┌────────────┴────────────┐                                    │
│          │                         │                                    │
│   Direct Answer              Tool Call                                  │
│          │                         │                                    │
│          │              ┌──────────▼──────────┐                         │
│          │              │    TOOL NODE         │                         │
│          │              │  <module>_search()   │                         │
│          │              │  → FAISS retriever   │                         │
│          │              │  → top-k chunks      │                         │
│          │              └──────────┬───────────┘                         │
│          │                         │                                    │
│          └────────────┬────────────┘                                    │
│                       │                                                 │
│                       ▼                                                 │
│                 Stream response                                          │
│                  token-by-token                                          │
│                       │                                                 │
│                      END                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ (traces everything)
┌─────────────────────────────────────────────────────────────────────────┐
│                         LANGSMITH (Observability)                       │
│   LLM calls  ·  Tool invocations  ·  Documents retrieved  ·  Latency   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 How It Works — Step by Step

### Step 1 — User Logs In
```
Chainlit shows login form
  → auth_callback() validates credentials against VALID_USERS dict
  → Returns User(id="username")
  → User's thread history loaded from chainlit_threads.db
  → Sidebar populated with past conversations
```

### Step 2 — Sidebar Loads Thread History
```
SQLiteDataLayer.list_threads(user_id)
  → Queries chainlit_threads.db
  → Returns all threads sorted by most recent activity
  → Sidebar updates live (no page reload needed)
```

### Step 3 — User Sends a Message
```
on_message() triggered
  → Message saved to chainlit_threads.db  (UI persistence)
  → Message saved to chat_history.db      (LangGraph memory)
  → Thread auto-titled from first 60 chars of first message
  → Sidebar updated via emitter.emit("update_thread", ...)
```

### Step 4 — LangGraph Agent Decides
```
LLM (Qwen 7B) receives: system prompt + full conversation history
  │
  ├─► Direct answer → no tool needed → streams response
  │
  └─► Tool call → LLM emits: { tool: "<module>_search", query: "..." }
        → FAISS retriever: similarity search, k=4 chunks returned
        → Chunk text + metadata (source, page, module) sent back to LLM
        → LLM refines and streams final answer
```

### Step 5 — Response Streams Token-by-Token
```
astream_events(..., version="v2")
  → fires "on_chat_model_stream" once per token
  → each token: await response_msg.stream_token(chunk.content)
  → user sees response appearing character by character
```

### Step 6 — Conversation Persisted
```
Full turn saved to both databases:
  chat_history.db     → LangGraph checkpointer (LLM memory per thread_id)
  chainlit_threads.db → Chainlit UI (sidebar + message history)

User can close browser, return later, click any thread → full history restored
```

---

## 📁 Project Structure

```
RAG_Project/
│
├── chainlit_app.py           ← Chainlit UI, auth, thread/message persistence, dataLayer
├── app.py                    ← LangGraph RAG graph, LLM, tools, FAISS retriever, tracing
├── ingest.py                 ← Document ingestion pipeline (load → chunk → embed → index)
│
├── requirements.txt          ← Python dependencies
├── DockerFile                ← Docker image definition
├── docker-compose.yml        ← Multi-container orchestration
├── .dockerignore
├── config.toml               ← Chainlit UI config
├── custom.css                ← Custom Chainlit styling
│
├── data/                     ← Drop your PDFs / DOCX / XLSX / JSON here
│   └── (your documents)
│
├── processed/                ← Files moved here after successful ingestion
│   ├── default/
│   └── project_a/
│
├── vectorstore/              ← FAISS indices (output of ingest.py)
│   ├── default/
│   │   ├── index.faiss       ← Binary vector index
│   │   ├── index.pkl         ← Index metadata
│   │   └── docstore.pkl      ← Document text + metadata
│   └── project_a/
│       └── ...
│
├── chat_history.db           ← LangGraph checkpointer (multi-turn LLM memory)
├── chainlit_threads.db       ← Chainlit UI threads + messages (sidebar)
└── Project_Update.txt        ← Changelog
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/pratiksinha312511/RAG_Project.git
cd RAG_Project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure `.env`

Create a `.env` file in the project root:

```env
# HuggingFace (required for LLM)
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

# LangSmith Tracing (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_pt_your_api_key_here
LANGCHAIN_PROJECT=pdf-rag-chatbot

# Chainlit Auth (optional)
CHAINLIT_AUTH_SECRET=your_secret_key_here

# Google Gemini (optional)
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Create Folder Structure

```bash
mkdir data processed vectorstore
```

### 4. Ingest Your Documents

```bash
# Place files in data/ folder, then:
python ingest.py

# For named modules:
MODULE=company_docs python ingest.py
MODULE=faq python ingest.py
```

### 5. Run the App

```bash
chainlit run chainlit_app.py -w
```

Open → `http://localhost:8000` → Login with credentials below.

---

## 📥 Document Ingestion

The `ingest.py` pipeline processes any file in `data/` and converts it into a searchable FAISS vectorstore:

```
data/ folder
    │
    ├── document.pdf   ┐
    ├── report.docx    ├─► Loaded by LangChain document loaders
    ├── data.xlsx      │
    └── config.json    ┘
            │
            ▼
    ┌───────────────────────┐
    │    Text Chunking      │  chunk_size    = 1000 tokens
    │  RecursiveCharacter   │  chunk_overlap = 100 tokens
    │    TextSplitter       │
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐
    │      Embedding        │  BAAI/bge-small-en-v1.5
    │      FastEmbed        │  384-dimensional vectors
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐
    │     FAISS Index       │  Saved to vectorstore/<module>/
    │     + Docstore        │  index.faiss · index.pkl · docstore.pkl
    └──────────┬────────────┘
               │
               ▼
    processed/<module>/     ← Files moved here after indexing
```

---

## 🗂️ Multi-Module Setup

Organise documents into separate knowledge bases — each gets its own vectorstore and named retriever tool:

```bash
MODULE=company_docs  python ingest.py   →  vectorstore/company_docs/
MODULE=faq           python ingest.py   →  vectorstore/faq/
MODULE=engineering   python ingest.py   →  vectorstore/engineering/
```

At runtime, `app.py` auto-discovers all vectorstore folders and creates a named LangGraph tool for each:

```
vectorstore/company_docs/  →  tool: company_docs_search
vectorstore/faq/           →  tool: faq_search
vectorstore/engineering/   →  tool: engineering_search
```

The LLM automatically decides which tool to call based on the user's query — no manual routing needed:

```
User: "What's our refund policy?"
  └─► LLM calls: faq_search("refund policy")

User: "Show me the Q3 engineering spec for module X"
  └─► LLM calls: engineering_search("Q3 engineering spec module X")
```

---

## 📊 LangSmith Tracing & Observability

LangSmith gives you **full visibility** into every step of the RAG pipeline.

```
┌──────────────────────────────────────────────────────────┐
│               LangSmith Run Timeline                     │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  LLM Step (Qwen 7B)                               │   │
│  │  • Tokens streamed : 312                          │   │
│  │  • Temperature     : 0.7                          │   │
│  │  • Latency         : 1.52s                        │   │
│  └───────────────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────────────┐   │
│  │  Tool / Retriever Step                            │   │
│  │  • Input query          : "transformer attention" │   │
│  │  • Documents retrieved  : 4                       │   │
│  │  • Chunk sources        : paper.pdf (p.3, p.7)    │   │
│  │  • Retriever latency    : 67ms                    │   │
│  └───────────────────────────────────────────────────┘   │
│  Metadata: module=default · tokens=312 · cost=$0.00      │
└──────────────────────────────────────────────────────────┘
```

### Setup

Ensure `.env` has LangSmith credentials. Traces are **auto-captured** — no code changes needed. The app initialises a `LangSmithTracer` + `CallbackManager` on startup and passes it to both the LLM and all retriever tools.

### View Traces

1. Open [smith.langchain.com](https://smith.langchain.com)
2. Select project: `pdf-rag-chatbot`
3. Go to **"Runs"** tab → click any run to inspect the full timeline

---

## 📡 Token-Level Streaming

Responses stream **token-by-token** as the LLM generates them — not chunk-by-chunk.

```python
# chainlit_app.py
async for event in app.astream_events(
    {"messages": [HumanMessage(content=message.content)]},
    config={
        "configurable": {"thread_id": thread_id},
        "checkpointer": memory,
    },
    version="v2",                          # ← fires once per token
):
    if event["event"] == "on_chat_model_stream":
        chunk = event.get("data", {}).get("chunk")
        if chunk and hasattr(chunk, "content"):
            if isinstance(chunk.content, str):
                await response_msg.stream_token(chunk.content)
```

`version="v2"` fires `on_chat_model_stream` **hundreds of times per response** — once per token — making the UI feel instant and conversational.

---

## 🗄️ Database Architecture

The app maintains **two separate SQLite databases** with distinct responsibilities:

```
┌──────────────────────────────────────────────────────────────────────┐
│  chat_history.db  (LangGraph Checkpointer)                           │
│                                                                      │
│  Purpose  : LLM conversation memory, scoped per thread               │
│  Created  : on_chat_start() via AsyncSqliteSaver                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ checkpoints                                                  │    │
│  │ thread_id │ checkpoint_id │ timestamp │ channel │ values BLOB│    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  • LLM sees full prior conversation via thread_id automatically      │
│  • No manual context injection needed                                │
│  • Isolated per thread — different thread_id = fresh memory          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  chainlit_threads.db  (Chainlit UI Persistence)                      │
│                                                                      │
│  Purpose  : Sidebar thread list + message history display            │
│  Updated  : in on_message() after each user/assistant turn           │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ cl_threads                                                   │    │
│  │ id (PK) │ name (title) │ user_id │ created_at │ metadata     │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ cl_messages                                                  │    │
│  │ id (PK) │ thread_id (FK) │ role │ content │ created_at       │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  Indexes: idx_messages_thread · idx_threads_user                     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  vectorstore/<module>/  (FAISS Vector Index)                         │
│                                                                      │
│  Purpose  : Semantic document search (read-only at runtime)          │
│  Created  : by ingest.py — never modified during chat                │
│                                                                      │
│  ├── index.faiss    ← Binary vector index + search tree              │
│  ├── index.pkl      ← Index metadata                                 │
│  └── docstore.pkl   ← Document text + source / page / module tags    │
│                                                                      │
│  Embedding : BAAI/bge-small-en-v1.5  (384 dims)                      │
│  Search    : cosine similarity · default k=4                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tools & Retriever Integration

`app.py` auto-creates one **named LangGraph tool per vectorstore module**:

```python
for module_name, retriever in retrievers.items():
    tool = StructuredTool.from_function(
        func=search,
        name=f"{module_name}_search",
        description=f"Search inside {module_name} knowledge base"
    )
```

Each tool:
- Takes a query string as input
- Returns concatenated text of top-k retrieved document chunks
- Is **fully traced in LangSmith** (documents returned + retrieval latency)

The LLM autonomously picks the correct tool and query based on conversation context — no hardcoded routing.

---

## ⚡ Performance & Latency

Live metrics from LangSmith traces:

```
┌──────────────────────────────────────────────────────┐
│              End-to-End Latency Breakdown            │
│                                                      │
│  P50 :  1.47s – 1.58s                               │
│  P99 :  2.56s                                       │
│                                                      │
│  Component breakdown:                               │
│  ├── LLM inference (Qwen 7B via HF) : ~1.5–2.5s  ← bottleneck
│  ├── FAISS retriever                : ~50–100ms   │
│  └── Streaming overhead             : ~100–300ms  │
└──────────────────────────────────────────────────────┘
```

### Optimization Options

| Option | Latency Gain | Effort | Notes |
|--------|-------------|--------|-------|
| Reduce `k` (4 → 2) | ~50ms | ⭐ Easy | Less context for LLM to process |
| MMR retriever | Neutral | ⭐ Easy | Better quality, same speed |
| Larger chunks (1000 → 2000) | Neutral | ⭐ Easy | Fewer reads, higher info density |
| Better embeddings (bge-base) | +100ms | ⭐ Easy | Better ranking, requires re-ingest |
| **Local LLM (Ollama)** | **40–50%** | ⭐⭐ Medium | **Biggest single improvement** |
| Quantized model (int8/int4) | 15–30% | ⭐⭐ Medium | Slight quality tradeoff |
| Reranker (CrossEncoder) | Slower overall | ⭐⭐ Medium | Better relevance, +200–500ms |

**Quick win** — reduce `k` in `app.py`:
```python
retrievers[module.lower()] = db.as_retriever(search_kwargs={"k": 2})
```

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# App available at:
# http://localhost:8000
```

Add your `.env` file before running. The `DockerFile` and `docker-compose.yml` handle all dependencies and environment wiring automatically.

---

## 🔐 Login Credentials

| Username | Password |
|----------|----------|
| `admin` | `admin123` |
| `user` | `user123` |

**To change credentials**, edit `VALID_USERS` in `chainlit_app.py`:

```python
VALID_USERS = {
    "yourname": "yourpassword",
    "alice":    "alice_secure_pass",
}
```

---

## 🔧 Troubleshooting

**App won't start / module not found**
```bash
pip install -r requirements.txt
pip install langsmith
python -c "import chainlit; import langgraph; import langsmith; print('OK')"
```

**PDFs not ingesting**
```bash
ls data/           # Confirm files are present
python ingest.py   # Run with verbose output
# Expected output: "Indexed 1200 chunks" or similar
```

**Sidebar is empty after login**
- You must send at least one message — sidebar only shows threads with messages
- Check `chainlit_threads.db` exists and is readable
- Open browser console (F12) for JS errors

**LangSmith traces not appearing**
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('LANGCHAIN_TRACING_V2'))"
# Should print: true
```
- Check startup logs for: `[LangSmith] Tracing enabled → project: pdf-rag-chatbot`
- Verify API key is valid at [smith.langchain.com](https://smith.langchain.com)

**Retriever returning no results**
- Ensure `vectorstore/default/index.faiss` exists — run `ingest.py` first
- Reduce query to key terms (semantic search needs topic overlap with documents)
- Temporarily increase `k`: `search_kwargs={"k": 8}`

**LLM responses are slow**
- Switch to a local LLM via Ollama — biggest impact
- Reduce `k` to 2 for faster responses
- Try `mistralai/Mistral-7B-Instruct-v0.3` (often faster on HF free tier)

---

## 📦 API References

### HuggingFace LLM Models

| Model | HF ID | Notes |
|-------|-------|-------|
| **Qwen 2.5 7B** *(default)* | `Qwen/Qwen2.5-7B-Instruct` | Current default |
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` | Good balance |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B-Instruct` | Strong reasoning |
| Zephyr 7B | `HuggingFaceH4/zephyr-7b-beta` | Instruction following |

### Embedding Model

| Model | Dims | Speed | Quality |
|-------|------|-------|---------|
| **BAAI/bge-small-en-v1.5** *(default)* | 384 | ⚡ Fast | Good |
| BAAI/bge-base-en-v1.5 | 768 | Moderate | Better |
| all-MiniLM-L6-v2 | 384 | ⚡ Fast | Weaker |

---

## 📜 License & Contributing

Production-ready template designed for learning and extension. Fork freely, adapt to your use case.

Reference docs:
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Chainlit Docs](https://docs.chainlit.io/)
- [LangSmith Docs](https://docs.smith.langchain.com/)
- [FAISS Docs](https://faiss.ai/)

---

<div align="center">

Built with ❤️ using **LangGraph** · **Chainlit** · **FAISS** · **LangSmith** · **HuggingFace**

</div>
