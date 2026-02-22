# PDF Chatbot — Chainlit with Sidebar Thread History

## Why authentication is required

Chainlit's thread history sidebar **only activates when there is a logged-in user**.
Without authentication, `list_threads()` is never called and the sidebar stays hidden.
This is a hard requirement in Chainlit's architecture, not a config option.

## Setup

```bash
# 1. Install dependencies
pip install chainlit langgraph langchain langchain-google-genai \
            langchain-community fastembed pymupdf python-dotenv literalai

# 2. Add PDFs to data/ and ingest
python ingest.py

# 3. Run
chainlit run chainlit_app.py -w
```

## Login credentials (default)

| Username | Password  |
|----------|-----------|
| admin    | admin123  |
| user     | user123   |

To change credentials, edit `VALID_USERS` at the top of `chainlit_app.py`:
```python
VALID_USERS = {
    "yourname": "yourpassword",
}
```

## .env file

```
GOOGLE_API_KEY=your_google_api_key_here
```

## File structure

```
project/
├── chainlit_app.py         ← Chainlit app (with auth + DataLayer)
├── app.py                  ← LangGraph RAG pipeline
├── ingest.py               ← PDF ingestion
├── .chainlit/
│   └── config.toml
├── public/
│   └── custom.css
├── vectorstore/db_faiss/   ← built by ingest.py
├── data/                   ← put PDFs here
├── chat_history.db         ← LangGraph checkpoints
├── chainlit_threads.db     ← Chainlit sidebar data
└── .env
```

## How it works

1. User logs in → Chainlit calls `auth_callback` → returns `User` object
2. Sidebar appears and calls `list_threads(filters)` with the user's identifier
3. Each chat session gets a `thread_id` from `cl.context.session.thread_id`
4. First message titles the thread; all messages saved to `cl_messages` table
5. Clicking a sidebar thread calls `on_chat_resume(thread)` with `thread["id"]`
6. We use `thread["id"]` (not the session id) to load the correct messages
7. LangGraph uses the same `thread_id` as config key for checkpoint lookup