"""
chainlit_app.py  -  PDF RAG Chatbot with sidebar thread history
Run with:  chainlit run chainlit_app.py -w

DATABASE MAP:
  chainlit_threads.db  → Chainlit UI: thread list + messages shown in sidebar
                         Tables: cl_threads, cl_messages
  chat_history.db      → LangGraph checkpointer: LLM multi-turn memory
                         Tables: checkpoints (internal langgraph schema)
  vectorstore/<mod>/   → FAISS embeddings — created by ingest.py, never touched here
"""

import uuid, json, sqlite3, os, traceback
from typing import Dict, List, Optional, Any

import chainlit as cl
import chainlit.data as cl_data
from chainlit.user import User
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app import app

# ── Credentials ────────────────────────────────────────────────────────────
VALID_USERS = {
    "admin": "admin123",
    "user":  "user123",
}

# ── Version-safe type imports ──────────────────────────────────────────────
def _safe_import(*paths):
    import importlib
    for mp, name in paths:
        try:
            obj = getattr(importlib.import_module(mp), name, None)
            if obj is not None:
                return obj
        except Exception:
            pass
    return None

ThreadDict    = _safe_import(("chainlit.types", "ThreadDict"),        ("chainlit.data", "ThreadDict"))       or dict
PageInfo      = _safe_import(("chainlit.types", "PageInfo"),          ("chainlit.data", "PageInfo"))
PaginatedResp = _safe_import(("chainlit.types", "PaginatedResponse"), ("chainlit.data", "PaginatedResponse"))

try:
    from literalai.helper import utc_now
except ImportError:
    from datetime import datetime, timezone
    def utc_now(): return datetime.now(timezone.utc).isoformat()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CL_DB_PATH   = os.path.join(BASE_DIR, "chainlit_threads.db")  # UI / sidebar db
LANGGRAPH_DB = os.path.join(BASE_DIR, "chat_history.db")       # LLM memory db

# ── SQLite: Chainlit UI persistence ───────────────────────────────────────
_db = sqlite3.connect(CL_DB_PATH, check_same_thread=False)
_db.row_factory = sqlite3.Row
_db.executescript("""
    CREATE TABLE IF NOT EXISTS cl_threads (
        id         TEXT PRIMARY KEY,
        name       TEXT,
        user_id    TEXT,
        created_at TEXT,
        metadata   TEXT DEFAULT '{}'
    );
    CREATE TABLE IF NOT EXISTS cl_messages (
        id         TEXT PRIMARY KEY,
        thread_id  TEXT,
        role       TEXT,
        content    TEXT,
        created_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_messages_thread ON cl_messages(thread_id);
    CREATE INDEX IF NOT EXISTS idx_threads_user    ON cl_threads(user_id);
""")
_db.commit()

# ── One-time migration: retitle "New Chat" threads that already have messages ──
# Fixes threads created before this patch that were stored as "New Chat" forever.
def _migrate_retitle_threads():
    rows = _db.execute(
        "SELECT id FROM cl_threads WHERE name = 'New Chat' OR name IS NULL OR name = ''"
    ).fetchall()
    for row in rows:
        first_user_msg = _db.execute(
            "SELECT content FROM cl_messages WHERE thread_id=? AND role='user' ORDER BY created_at ASC LIMIT 1",
            (row["id"],)
        ).fetchone()
        if first_user_msg:
            _db.execute(
                "UPDATE cl_threads SET name=? WHERE id=?",
                (first_user_msg["content"][:60], row["id"])
            )
    _db.commit()

_migrate_retitle_threads()

# ── DB helpers ─────────────────────────────────────────────────────────────

def db_upsert_thread(thread_id: str, name: str, user_id: str = "anonymous"):
    exists = _db.execute("SELECT id FROM cl_threads WHERE id=?", (thread_id,)).fetchone()
    if exists:
        _db.execute("UPDATE cl_threads SET name=? WHERE id=?", (name, thread_id))
    else:
        _db.execute(
            "INSERT INTO cl_threads (id,name,user_id,created_at,metadata) VALUES (?,?,?,?,?)",
            (thread_id, name, user_id, utc_now(), "{}"),
        )
    _db.commit()


def db_save_message(thread_id: str, role: str, content: str):
    """Persist only real, non-empty conversation turns."""
    if not content or not content.strip():
        return
    _db.execute(
        "INSERT INTO cl_messages (id,thread_id,role,content,created_at) VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), thread_id, role, content, utc_now()),
    )
    _db.commit()


def db_get_messages(thread_id: str) -> list:
    return _db.execute(
        "SELECT * FROM cl_messages WHERE thread_id=? ORDER BY created_at ASC",
        (thread_id,)
    ).fetchall()


def db_list_threads(user_id: str) -> list:
    """Threads with at least one message, ordered by most recent activity first."""
    return _db.execute("""
        SELECT t.*,
               MAX(m.created_at) AS last_message_at
          FROM cl_threads t
          JOIN cl_messages m ON m.thread_id = t.id
         WHERE t.user_id = ?
         GROUP BY t.id
         ORDER BY last_message_at DESC
    """, (user_id,)).fetchall()


def db_get_thread(thread_id: str):
    return _db.execute("SELECT * FROM cl_threads WHERE id=?", (thread_id,)).fetchone()


# ── DataLayer helpers ──────────────────────────────────────────────────────

def _steps_from_messages(messages) -> List[Dict]:
    """
    Convert cl_messages rows into the step dicts Chainlit's frontend expects.

    *** ROOT CAUSE OF THE INVISIBLE USER MESSAGES BUG ***
    Chainlit renders the visible bubble text from `output` for ALL step types —
    both user messages AND assistant messages.
    `input` is only shown for tool-call argument display (e.g. function params),
    it is NEVER rendered as a chat bubble.

    The previous code put user content in `input` and left `output` as "" —
    so every user message rendered as a blank invisible bubble when switching threads.
    Fix: always put content in `output` for every message, leave `input` empty.
    """
    steps = []
    for m in messages:
        is_user = m["role"] == "user"
        steps.append({
            "id":        m["id"],
            "threadId":  m["thread_id"],
            "name":      "User"          if is_user else "Assistant",
            "type":      "user_message"  if is_user else "assistant_message",
            "input":     "",             # always blank — chat text never goes here
            "output":    m["content"],   # ← ALL bubble text lives in output
            "createdAt": m["created_at"],
            "isError":   False,
            "metadata":  {},
            "tags":      [],
        })
    return steps


def _build_thread_dict(row, steps) -> dict:
    d = dict(
        id=row["id"],
        name=row["name"] or "New Chat",
        userId=row["user_id"],
        userIdentifier=row["user_id"],
        createdAt=row["created_at"],
        metadata=json.loads(row["metadata"] or "{}"),
        steps=steps,
        elements=[],
        tags=[],
    )
    if ThreadDict is not dict:
        try:
            return ThreadDict(**d)
        except Exception:
            pass
    return d


# ── DataLayer ──────────────────────────────────────────────────────────────

class SQLiteDataLayer(cl_data.BaseDataLayer):

    async def get_thread_author(self, thread_id: str) -> str:
        r = _db.execute("SELECT user_id FROM cl_threads WHERE id=?", (thread_id,)).fetchone()
        return r["user_id"] if r else "anonymous"

    async def get_thread(self, thread_id: str = None, tid: str = None) -> Optional[Dict]:
        """
        Called by Chainlit when a thread is clicked in the sidebar.
        The returned steps dict is what Chainlit renders as the full chat history.
        """
        _id = thread_id or tid
        if not _id:
            return None
        row = db_get_thread(_id)
        if not row:
            return None
        steps = _steps_from_messages(db_get_messages(_id))
        return _build_thread_dict(row, steps)

    async def update_thread(self, thread_id, name=None, user_id=None, metadata=None, tags=None):
        # If Chainlit passes name=None (it does this on session init), do NOT
        # overwrite an existing real title with "New Chat". Only write the name
        # if one is actually provided, otherwise preserve what's already in DB.
        if name:
            db_upsert_thread(thread_id, name, user_id or "anonymous")
        elif not db_get_thread(thread_id):
            # Thread doesn't exist yet — create it with placeholder
            db_upsert_thread(thread_id, "New Chat", user_id or "anonymous")

    async def delete_thread(self, thread_id: str):
        _db.execute("DELETE FROM cl_messages WHERE thread_id=?", (thread_id,))
        _db.execute("DELETE FROM cl_threads  WHERE id=?",        (thread_id,))
        _db.commit()

    async def list_threads(self, pagination=None, filters=None) -> Any:
        """Populate the sidebar with the current user's threads."""
        user_id = "anonymous"
        if filters:
            try:
                user_id = (
                    filters.get("userId") if isinstance(filters, dict)
                    else getattr(filters, "userId", None)
                        or getattr(filters, "userIdentifier", None)
                        or "anonymous"
                )
            except Exception:
                pass

        threads = []
        for row in db_list_threads(user_id):
            steps = _steps_from_messages(db_get_messages(row["id"]))
            threads.append(_build_thread_dict(row, steps))

        if PaginatedResp and PageInfo:
            try:
                return PaginatedResp(
                    data=threads,
                    pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
                )
            except Exception:
                pass
        return threads

    async def get_user(self, identifier: str):
        if identifier in VALID_USERS:
            try:
                from chainlit.user import PersistedUser
                return PersistedUser(
                    id=identifier,
                    identifier=identifier,
                    createdAt=utc_now(),
                )
            except ImportError:
                return {"id": identifier, "identifier": identifier}
        return None

    async def create_user(self, user):
        return user

    # ── Stubs required by BaseDataLayer ───────────────────────────────────
    async def create_step(self, s: Dict):                pass
    async def update_step(self, s: Dict):                pass
    async def delete_step(self, sid: str):               pass
    async def get_element(self, tid, eid):               return None
    async def create_element(self, el):                  pass
    async def delete_element(self, eid, thread_id=None): pass
    async def upsert_feedback(self, feedback=None):      return str(uuid.uuid4())
    async def delete_feedback(self, fid):                return True
    async def build_debug_url(self):                     return ""
    async def close(self):                               pass
    async def get_favorite_steps(self):                  return []


cl_data._data_layer = SQLiteDataLayer()


# ── Authentication ─────────────────────────────────────────────────────────

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[User]:
    if VALID_USERS.get(username) == password:
        return User(identifier=username, metadata={"role": "user"})
    return None


# ── Shared helper: initialise LangGraph memory for a thread ───────────────

async def _init_memory(thread_id: str):
    """
    Open AsyncSqliteSaver against chat_history.db scoped to thread_id.
    Called from BOTH on_chat_start and on_chat_resume so memory is always valid.
    """
    saver_cm = AsyncSqliteSaver.from_conn_string(LANGGRAPH_DB)
    memory   = await saver_cm.__aenter__()
    cl.user_session.set("memory_cm", saver_cm)
    cl.user_session.set("memory",    memory)
    cl.user_session.set("thread_id", thread_id)  # single authoritative assignment
    return memory


# ── Chainlit hooks ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Called for every brand-new chat session (never called on resume)."""
    thread_id = cl.context.session.thread_id   # authoritative source — never overwrite
    user      = cl.user_session.get("user")
    user_id   = user.identifier if user else "anonymous"

    cl.user_session.set("user_id",   user_id)
    cl.user_session.set("msg_count", 0)

    await _init_memory(thread_id)
    db_upsert_thread(thread_id, "New Chat", user_id)

    # Welcome message is intentionally NOT saved to db_save_message.
    # If it were saved, it would appear as a stale assistant bubble every time
    # the user resumes this thread — exactly like the "Resuming..." bug.
    await cl.Message(
        content=(
            "## 📄 PDF Chatbot\n\n"
            "Ask me anything about your uploaded documents.\n\n"
            "Your previous conversations are saved in the **sidebar on the left**. "
            "Click any thread to resume it, or use the ✏️ icon to start a new chat."
        ),
        author="Assistant",
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Called when the user clicks an existing thread in the sidebar.

    Chainlit already re-renders the full message history from the steps returned
    by get_thread() BEFORE this hook fires. So we must NOT send any cl.Message()
    here — it would show up as an extra phantom bubble at the top of the resumed chat.

    We only need to restore session state and re-initialise LangGraph memory.
    """
    thread_id = thread["id"] if isinstance(thread, dict) else thread.id
    user      = cl.user_session.get("user")
    user_id   = user.identifier if user else "anonymous"

    cl.user_session.set("user_id", user_id)

    # Restore msg_count so the title-update logic in on_message doesn't fire again
    messages = db_get_messages(thread_id)
    cl.user_session.set("msg_count", len(messages))

    # Re-init LangGraph memory so the LLM remembers context from prior turns
    await _init_memory(thread_id)

    # No cl.Message().send() — Chainlit handles history rendering automatically


@cl.on_message
async def on_message(message: cl.Message):
    """Handle an incoming user message — works for both new and resumed threads."""

    thread_id = cl.user_session.get("thread_id")
    user_id   = cl.user_session.get("user_id", "anonymous")
    memory    = cl.user_session.get("memory")

    # Safety net: should never be None, but recover gracefully if it is
    if memory is None:
        memory    = await _init_memory(thread_id or cl.context.session.thread_id)
        thread_id = cl.user_session.get("thread_id")

    # Persist user message to UI db
    db_save_message(thread_id, "user", message.content)

    # Name the thread after the first user message (like ChatGPT / Claude).
    # We must do TWO things:
    #   1. db_upsert_thread  → persists the title to SQLite for future loads
    #   2. emit "update_thread" → tells Chainlit's live frontend to update the
    #      sidebar title RIGHT NOW without a page refresh
    msg_count = cl.user_session.get("msg_count", 0)
    if msg_count == 0:
        title = message.content[:60]
        db_upsert_thread(thread_id, title, user_id)
        # Notify the live sidebar to rename this thread immediately
        try:
            await cl.context.emitter.emit(
                "update_thread",
                {"threadId": thread_id, "name": title}
            )
        except Exception:
            pass  # safe fallback — title will be correct on next page load
    cl.user_session.set("msg_count", msg_count + 1)

    # Stream the assistant response
    response_msg = cl.Message(content="", author="Assistant")
    await response_msg.send()

    try:
        # LangGraph restores the full prior conversation from chat_history.db
        # automatically via checkpointer + thread_id. We pass only the new message.
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
                    elif isinstance(chunk.content, list):
                        for part in chunk.content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                await response_msg.stream_token(part.get("text", ""))

        await response_msg.update()

    except Exception as e:
        traceback.print_exc()
        await response_msg.stream_token(f"\n\n⚠️ Error: {str(e)}")
        await response_msg.update()

    # Persist assistant reply to UI db
    db_save_message(thread_id, "assistant", response_msg.content)


@cl.on_chat_end
async def on_chat_end():
    """Close AsyncSqliteSaver to flush LangGraph checkpoints to disk."""
    saver_cm = cl.user_session.get("memory_cm")
    if saver_cm:
        try:
            await saver_cm.__aexit__(None, None, None)
        except Exception:
            pass


if __name__ == "__main__":
    print("Run this app with:  chainlit run chainlit_app.py -w")