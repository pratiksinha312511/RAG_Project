# PDF Chatbot — LangGraph RAG with Chainlit + LangSmith Tracing

A production-ready RAG (Retrieval-Augmented Generation) chatbot using **LangGraph**, **Chainlit**, **FAISS vectorstores**, and **LangSmith tracing**. Features secure authentication, multi-threaded conversations with sidebar history, token-level streaming, and full observability into retriever & LLM activity.

---

## Quick Start

### 1. Environment Setup

Clone the repo and create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# or: source .venv/bin/activate  # Mac/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages (also in `requirements.txt`):
- `chainlit` — UI framework
- `langgraph` — Agent/RAG orchestration
- `langchain` & `langchain-community` — LLM chains, tools, retrievers
- `langchain-huggingface` — HuggingFace LLM endpoint
- `fastembed` — Fast embeddings (BAAI/bge-small-en-v1.5)
- `python-dotenv` — Environment variable loading
- `langsmith` — LangSmith SDK for tracing (auto-installed with langchain)

### 3. Configure `.env` File

Create `.env` in the project root with your API keys:

```dotenv
# HuggingFace API (required for LLM)
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

# LangSmith Tracing (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_pt_your_api_key_here
LANGCHAIN_PROJECT=pdf-rag-chatbot

# Chainlit Auth (optional)
CHAINLIT_AUTH_SECRET=your_secret_key_here

# Google Gemini (optional if using Google LLMs)
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Ingest PDFs

Place PDF files (or `.docx`, `.xlsx`, `.json`) in `data/` folder:

```bash
python ingest.py
```

This will:
- Load documents from `data/`
- Split into 1000-token chunks (100-token overlap)
- Embed using `BAAI/bge-small-en-v1.5`
- Save FAISS index to `vectorstore/default/`
- Move processed files to `processed/default/`

To ingest into a different module (vectorstore):
```bash
MODULE=project_a python ingest.py
MODULE=project_b python ingest.py
```

### 5. Run the App

```bash
chainlit run chainlit_app.py -w
```

Open browser → `http://localhost:8000` → Login (see credentials below)

---

## Login Credentials (Default)

| Username | Password  |
|----------|-----------|
| admin    | admin123  |
| user     | user123   |

**Change credentials** by editing `VALID_USERS` in [chainlit_app.py](chainlit_app.py#L28):

```python
VALID_USERS = {
    "yourname": "yourpassword",
    "alice": "alice_password",
}
```

---

## Architecture

### System Flow

```
User Chat Input
    ↓
[Chainlit UI] ← Display + Auth
    ↓
[chainlit_app.py] ← Thread management, message persistence
    ↓
[app.py] LangGraph Agent
    ├── Query LLM w/ tools
    ├── LLM decides: answer directly OR call retriever tool
    └── Retriever tool searches FAISS vectorstore
    ↓
[LangSmith] ← Traces every step (LLM, retriever, tool calls)
    ↓
Stream Response Tokens Back to UI
```

### Why This Architecture?

- **LangGraph**: Handles agent loop, tools, state management reliably
- **FAISS**: Fast, local similarity search (no external API latency)
- **Chainlit**: Polished UI, auth, thread history out-of-box
- **LangSmith**: Full visibility into RAG pipeline (critical for debugging retriever quality)
- **Token-level streaming**: Real-time token delivery via `astream_events(..., version="v2")`

### Key Components

| File | Purpose |
|------|---------|
| [chainlit_app.py](chainlit_app.py) | Chainlit UI, auth, thread/message persistence, dataLayer |
| [app.py](app.py) | LangGraph RAG graph, LLM, tools, FAISS retriever setup |
| [ingest.py](ingest.py) | PDF/DOCX/XLSX loader, chunking, embedding, FAISS indexing |
| [chat_history.db](chat_history.db) | LangGraph checkpointer (LLM memory per thread) |
| [chainlit_threads.db](chainlit_threads.db) | Chainlit UI DB (sidebar threads + messages) |

---

## LangSmith Tracing

### Why LangSmith?

See **exactly** which documents were retrieved, token-by-token LLM streaming, latency breakdown, and error traces in one dashboard.

### Setup

1. **Ensure `.env` has LangSmith credentials:**
   ```dotenv
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=lsv2_pt_your_api_key_here
   LANGCHAIN_PROJECT=pdf-rag-chatbot
   ```

2. **Traces are auto-captured** — no code changes needed in `app.py` or `chainlit_app.py`:
   - `app.py` loads `.env` early and initializes `LangSmithTracer` + `CallbackManager`
   - Callback manager is passed to LLM and retriever tools
   - All LLM calls, tool invocations, and document fetches are traced

3. **View traces in LangSmith UI:**
   - Open https://smith.langchain.com
   - Select your project (`pdf-rag-chatbot`)
   - Go to **"Runs"** tab
   - Click any run to see timeline with:
     - **LLM step**: Tokens streamed, model, temperature, latency
     - **Tool/Retriever step**: Input query, retrieved documents, chunk sources
     - **Metadata**: Module name, token count, cost estimates

### Latency Metrics (Current)

From live traces:
- **P50 latency**: 1.47s – 1.58s
- **P99 latency**: 2.56s
- **Breakdown**:
  - LLM inference (Qwen 7B via HF): **~1.5–2.5s** ← bottleneck
  - Retriever (FAISS): **~50–100ms**
  - Streaming overhead: **~100–300ms**

**Bottleneck**: LLM inference via HuggingFace API.

### Optimization Options

| Option | Latency Gain | Effort | Notes |
|--------|--------------|--------|-------|
| Reduce `k` (4 → 2) | ~50ms | ⭐ Easy | Less context to process |
| MMR retriever | Neutral | ⭐ Easy | Better quality, same speed |
| Larger chunks (1000 → 2000) | Neutral | ⭐ Easy | Fewer doc reads, info density |
| Better embeddings (bge-base) | +100ms | ⭐ Easy | Better ranking; requires re-ingest |
| **Local LLM (Ollama)** | **40–50%** | ⭐⭐ Medium | **Biggest impact** |
| Quantized model (int8/int4) | 15–30% | ⭐⭐ Medium | Slightly lower quality |
| Reranker (CrossEncoder) | Slower | ⭐⭐ Medium | Better relevance, +200–500ms |

**Quick win**: Reduce `k` to 2–3 in [app.py](app.py#L68) (search_kwargs):
```python
retrievers[module.lower()] = db.as_retriever(search_kwargs={"k": 2})
```

**Best latency cut**: Run local LLM with Ollama or use a faster model.

---

## Token-Level Streaming

Responses stream **token-by-token** as the LLM generates them (not chunk-by-chunk).

Implementation in [chainlit_app.py](chainlit_app.py#L427):

```python
async for event in app.astream_events(
    {"messages": [HumanMessage(content=message.content)]},
    config={
        "configurable": {"thread_id": thread_id},
        "checkpointer": memory,
    },
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        chunk = event.get("data", {}).get("chunk")
        if chunk and hasattr(chunk, "content"):
            if isinstance(chunk.content, str):
                await response_msg.stream_token(chunk.content)
```

Why `v2`?
- `astream_events(..., version="v2")` fires `on_chat_model_stream` **hundreds of times per response** (once per token)
- Each token appears in the UI instantly, creating a natural streaming effect
- Traces in LangSmith show each token delivery for debugging

---

## Project Structure

```
f:\langgraph\
├── chainlit_app.py              ← Chainlit UI + Auth + DataLayer
├── app.py                        ← LangGraph RAG + LLM + Tools + Tracing
├── ingest.py                     ← Document ingestion pipeline
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
├── .env                          ← API keys (not in git, create locally)
├── .chainlit/
│   └── config.toml              ← Chainlit UI config
├── public/
│   └── custom.css               ← Custom styling
├── data/                         ← Place PDFs/DOCX/XLSX here (ingestion input)
├── processed/                    ← Files moved here after ingestion
│   └── default/                 ← Processed docs for "default" module
│   └── project_a/               ← Processed docs for "project_a" module
├── vectorstore/                  ← FAISS indices (output of ingest.py)
│   ├── default/
│   │   ├── index.faiss
│   │   ├── index.pkl
│   │   └── docstore.pkl
│   └── project_a/
│       ├── index.faiss
│       ├── index.pkl
│       └── docstore.pkl
├── chat_history.db              ← LangGraph checkpointer (multi-turn memory)
├── chainlit_threads.db          ← Chainlit UI threads + messages (sidebar)
└── __pycache__/
```

### Create the folder structure (one-time):

```bash
mkdir data processed vectorstore
```

---

## How It Works (Step-by-Step)

### 1. **User Logs In**
   - Chainlit shows login form
   - User enters username + password
   - `auth_callback` in [chainlit_app.py](chainlit_app.py#L307) validates
   - Returns `User(id="username")` → user cell session initialized

### 2. **Sidebar Loads**
   - Chainlit calls `SQLiteDataLayer.list_threads(user_id)`
   - Returns all threads for this user from `chainlit_threads` DB
   - User sees list in sidebar (sorted by most recent activity)

### 3. **User Sends Message**
   - Message typed in chat box
   - `on_message()` in [chainlit_app.py](chainlit_app.py#L409) triggered
   - Message saved to `chainlit_threads.db` + `chat_history.db`
   - Thread titled (if first message): "where and how does..." (first 60 chars)
   - Sidebar updated live without page reload (via `emitter.emit("update_thread", ...)`)

### 4. **LLM + Retriever Invoked**
   - [app.py](app.py#L170) LangGraph agent receives message
   - LLM (Qwen 7B) is asked: *"Answer this question or call the search tool if you need docs"*
   - LLM decides:
     - **No tool call**: Returns answer directly
     - **Tool call**: Calls `<module>_search` tool with query
   - If tool called:
     - Retriever searches FAISS for top-k (default k=4) similar documents
     - Document chunks + metadata returned to LLM
     - LLM sees context and refines answer
   - **All of this is traced** in LangSmith (LLM calls, tool invokes, docs retrieved)

### 5. **Response Streamed**
   - LLM output streamed token-by-token via `astream_events(..., version="v2")`
   - Each token fires `on_chat_model_stream` event
   - Token appended to UI in real-time
   - User sees response appearing character by character

### 6. **Conversation Persisted**
   - Full conversation saved to `chat_history.db` (LangGraph checkpoints)
   - LLM has access to full history next message (via `thread_id` + checkpointer)
   - User can close browser, come back later, click thread → full history loaded
   - All turns searchable/resumable

---

## Database Schema

### **chainlit_threads.db** (Chainlit UI Persistence)

Stores thread list and messages shown in sidebar:

```sql
CREATE TABLE cl_threads (
    id         TEXT PRIMARY KEY,
    name       TEXT,               -- Thread title (first 60 chars of first user msg)
    user_id    TEXT,               -- User who owns this thread
    created_at TEXT,               -- ISO 8601 timestamp
    metadata   TEXT DEFAULT '{}'   -- JSON blob for future extensions
);

CREATE TABLE cl_messages (
    id         TEXT PRIMARY KEY,
    thread_id  TEXT,               -- Foreign key to cl_threads
    role       TEXT,               -- "user" or "assistant"
    content    TEXT,               -- Full message text
    created_at TEXT                -- ISO 8601 timestamp
);

CREATE INDEX idx_messages_thread ON cl_messages(thread_id);
CREATE INDEX idx_threads_user    ON cl_threads(user_id);
```

**Lifecycle**:
- Created by `chainlit_app.py` on startup
- Updated in `on_message()` with each user/assistant turn
- Read by sidebar when user logs in
- Deleted when user deletes a thread

---

### **chat_history.db** (LangGraph Checkpointer)

Stores LLM conversation memory and agent state per thread:

```sql
CREATE TABLE checkpoints (
    thread_id  TEXT,
    checkpoint_id TEXT,
    timestamp  TEXT,
    channel    TEXT,
    version    INT,
    values     BLOB,  -- Serialized LangGraph state (pickled)
    metadata   TEXT   -- JSON config metadata
);
```

**Why it exists**:
- LLM sees full prior conversation automatically
- No manual context injection needed
- Thread-scoped: different `thread_id` = different conversation memory
- Created on `on_chat_start()` via `AsyncSqliteSaver`

---

### **vectorstore/\<module\>/index.faiss** (FAISS Vector Index)

Binary vector database created by `ingest.py`:

```
vectorstore/
└── default/                 ← Module "default"
    ├── index.faiss         ← FAISS index (binary vectors + search tree)
    ├── index.pkl           ← Metadata about index
    └── docstore.pkl        ← Document text + metadata (source, page, module)
```

**Content**:
- One vector per document chunk (1000 tokens, 100-token overlap)
- Embedding model: `BAAI/bge-small-en-v1.5` (384 dims)
- Metadata: source file, page number, module name
- Supports similarity search: `k=4` closest chunks returned

**Never modified by runtime** — only read during retriever calls.

---

## Tools & Retriever Integration

### How Tools Work

In [app.py](app.py#L80), a **tool is created per vectorstore module**:

```python
for module_name, retriever in retrievers.items():
    tool = StructuredTool.from_function(
        func=search,
        name=f"{module_name}_search",
        description=f"Search inside {module_name} knowledge base"
    )
```

Example: if you have 2 modules:
- `vectorstore/company_docs/` → tool named `company_docs_search`
- `vectorstore/faq/` → tool named `faq_search`

LLM can call either tool independently. Each tool:
- Takes query string as input
- Returns concatenated text of top-k documents
- **Traced in LangSmith** (documents logged + retrieval latency)

### Callback Manager Integration

`app.py` initializes `LangSmithTracer` + `CallbackManager` if `LANGCHAIN_TRACING_V2=true`:

```python
tracer = LangSmithTracer(project_name=os.getenv("LANGCHAIN_PROJECT", "default"))
cb_manager = CallbackManager(tracers=[tracer])
```

The callback manager is passed to:
1. **LLM** (ChatHuggingFace) → traces every LLM invocation
2. **Retriever** (FAISS.as_retriever) → traces document fetches
3. **Tools** via tool node → traces tool execution

Result: **Full visibility** into what docs were retrieved and why LLM made decisions.

---

## Troubleshooting

### **App won't start / Module not found errors**

```bash
# Ensure all deps installed
pip install -r requirements.txt
pip install langsmith
```

Check if imports work:
```bash
python -c "import chainlit; import langgraph; import langsmith; print('OK')"
```

### **PDFs not ingesting**

```bash
# Check if data/ folder exists and has files
ls data/

# Run ingest with verbose output
python ingest.py

# If files don't move to processed/, check permissions
# Expected output: "Indexed 1200 chunks" or similar
```

### **Sidebar is empty after login**

- Ensure you've logged in (no sidebar without auth)
- Ensure you have at least one message in the thread (sidebar only shows threads with messages)
- Check `chainlit_threads.db` exists and is readable
- Check browser console (F12) for JS errors

### **LangSmith traces not appearing**

1. **Check env vars loaded:**
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('LANGCHAIN_TRACING_V2'))"
   ```
   Should print: `true`

2. **Check console output on startup:**
   ```
   [LangSmith] Tracing enabled → project: pdf-rag-chatbot
   [LangSmith] LangSmithTracer initialized and callback manager created
   ```
   If not present → SDK not installed or env var not set

3. **Verify API key is valid** on https://smith.langchain.com (try signing in manually)

4. **Check network** — app must reach `https://api.smith.langchain.com`

### **Retriever returning no results**

- Ensure `vectorstore/default/index.faiss` exists (run `ingest.py`)
- Check embedding model loaded: `FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")`
- Check query is similar to doc content (semantic search requires good overlap)
- Try reducing query to key terms instead of full sentences
- Increase `k` in [app.py](app.py#L68) to see more results (e.g., `k=8`)

### **LLM response is slow** 

See [Optimization Options](#optimization-options) above. Quick wins:
1. Reduce `k` (fewer docs to process)
2. Switch to local LLM (Ollama) — biggest impact
3. Use smaller model (e.g., `mistralai/Mistral-7B-Instruct-v0.3`)

---

## Multi-Module Setup (Advanced)

You can organize PDFs into different modules (separate vectorstores):

```bash
# Ingest company docs into "company" module
MODULE=company python ingest.py

# Ingest FAQs into "faq" module  
MODULE=faq python ingest.py
```

Files created:
- `vectorstore/company/index.faiss` (company docs)
- `vectorstore/faq/index.faiss` (FAQ docs)
- Each gets its own tool: `company_search`, `faq_search`
- Sidebar shows both tools when LLM deliberates

LLM automatically decides which module to search based on query context.

---

## Performance Tips

| Action | Impact | How |
|--------|--------|-----|
| Reduce chunk size 1000 → 500 | More precise, slower ingest | Edit `ingest.py` line 145 |
| Increase chunk size 1000 → 2000 | Less precise, faster retriever | Edit `ingest.py` line 145 |
| Reduce k: 4 → 2 | Fewer docs to LLM (~100ms faster) | Edit `app.py` line 68 |
| Use better embeddings | Higher quality retrievals, +100ms | Change `bge-small` → `bge-base` in [app.py](app.py#L51) + re-ingest |
| Local LLM (Ollama) | 40–60% latency cut | Setup Ollama, change `HuggingFaceEndpoint` → local endpoint |

---

## API References

### HuggingFace LLM Models (in `app.py`)

Current: `Qwen/Qwen2.5-7B-Instruct`

Alternatives:
- `mistralai/Mistral-7B-Instruct-v0.3` — Good balance
- `meta-llama/Meta-Llama-3-8B-Instruct` — Strong reasoning (needs access)
- `HuggingFaceH4/zephyr-7b-beta` — Instruction-following
- Local via Ollama — Run on your machine

### Embedding Model (in `app.py` + `ingest.py`)

Current: `BAAI/bge-small-en-v1.5` (384-dim, fast)

Alternatives:
- `BAAI/bge-base-en-v1.5` (768-dim, higher quality, slower)
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast but weaker)

### Vector Store (FAISS)

Fast local similarity search. No external API calls.
- Supports: cosine, L2, inner product
- Pre-loaded on app startup
- Module-scoped per folder

---

## License & Contributing

This is a template for demonstration and learning. Feel free to fork, modify, and extend for your use case.

For bugs or questions, check:
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Chainlit Docs](https://docs.chainlit.io/)
- [LangSmith Docs](https://docs.smith.langchain.com/)