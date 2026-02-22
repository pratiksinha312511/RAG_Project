"""
chainlit_app.py  -  PDF RAG Chatbot with sidebar thread history
Run with:  chainlit run chainlit_app.py -w

IMPORTANT: The thread history sidebar in Chainlit ONLY appears when there is
an authenticated user. Without a logged-in user the DataLayer's list_threads()
is never called and the sidebar stays hidden.

We add a simple password-based login (no database needed for auth — just an
env var or hardcoded credentials). The user object is then passed through
every DataLayer call so threads are scoped per user.
"""

import uuid, json, sqlite3, os
from typing import Dict, List, Optional, Any

import chainlit as cl
import chainlit.data as cl_data
from chainlit.user import User
from langchain_core.messages import HumanMessage, AIMessage
from app import app

# ── Credentials — change these or load from .env ───────────────────────────
# Simple single-user setup. Add more entries for multi-user.
VALID_USERS = {
    "admin": "admin123",   # username: password
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

ThreadDict    = _safe_import(("chainlit.types","ThreadDict"),        ("chainlit.data","ThreadDict"))       or dict
PageInfo      = _safe_import(("chainlit.types","PageInfo"),          ("chainlit.data","PageInfo"))
PaginatedResp = _safe_import(("chainlit.types","PaginatedResponse"), ("chainlit.data","PaginatedResponse"))

try:
    from literalai.helper import utc_now
except ImportError:
    from datetime import datetime, timezone
    def utc_now(): return datetime.now(timezone.utc).isoformat()

# ── SQLite persistence ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CL_DB_PATH = os.path.join(BASE_DIR, "chainlit_threads.db")

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
""")
_db.commit()

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
    _db.execute(
        "INSERT INTO cl_messages (id,thread_id,role,content,created_at) VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), thread_id, role, content, utc_now()),
    )
    _db.commit()

def db_get_messages(thread_id: str) -> list:
    return _db.execute(
        "SELECT * FROM cl_messages WHERE thread_id=? ORDER BY created_at",
        (thread_id,)
    ).fetchall()

def db_list_threads(user_id: str) -> list:
    """Threads for this user that have at least one message."""
    return _db.execute("""
        SELECT t.* FROM cl_threads t
        WHERE t.user_id = ?
          AND EXISTS (SELECT 1 FROM cl_messages m WHERE m.thread_id = t.id)
        ORDER BY t.created_at DESC
    """, (user_id,)).fetchall()

def db_get_thread(thread_id: str):
    return _db.execute("SELECT * FROM cl_threads WHERE id=?", (thread_id,)).fetchone()

# ── DataLayer helpers ──────────────────────────────────────────────────────

def _steps_from_messages(messages) -> List[Dict]:
    steps = []
    for m in messages:
        is_user = m["role"] == "user"
        steps.append({
            "id":        m["id"],
            "threadId":  m["thread_id"],
            "name":      "user" if is_user else "Assistant",
            "type":      "user_message" if is_user else "assistant_message",
            "input":     m["content"] if is_user else "",
            "output":    m["content"] if not is_user else "",
            "createdAt": m["created_at"],
            "metadata":  {},
        })
    return steps

def _build_thread_dict(row, steps) -> dict:
    d = dict(
        id=row["id"],
        name=row["name"] or row["id"],
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

    async def get_thread_author(self, tid: str) -> str:
        r = _db.execute("SELECT user_id FROM cl_threads WHERE id=?", (tid,)).fetchone()
        return r["user_id"] if r else "anonymous"

    async def get_thread(self, thread_id: str = None, tid: str = None) -> Optional[Dict]:
        # Accept both calling conventions across Chainlit versions:
        #   get_thread("abc")            -- positional (old Chainlit)
        #   get_thread(thread_id="abc")  -- keyword (new Chainlit)
        _id = thread_id or tid
        if not _id:
            return None
        row = db_get_thread(_id)
        if not row:
            return None
        steps = _steps_from_messages(db_get_messages(_id))
        return _build_thread_dict(row, steps)

    async def update_thread(self, thread_id, name=None, user_id=None, metadata=None, tags=None):
        db_upsert_thread(thread_id, name or "New Chat", user_id or "anonymous")

    async def delete_thread(self, tid: str):
        _db.execute("DELETE FROM cl_messages WHERE thread_id=?", (tid,))
        _db.execute("DELETE FROM cl_threads  WHERE id=?",        (tid,))
        _db.commit()

    async def list_threads(self, pagination=None, filters=None) -> Any:
        # Get current user from filters if available, else fall back to "anonymous"
        user_id = "anonymous"
        if filters:
            try:
                # filters may be a ThreadFilter object or dict
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
        """Return a PersistedUser if the identifier exists in our VALID_USERS."""
        if identifier in VALID_USERS:
            try:
                from chainlit.user import PersistedUser
                return PersistedUser(
                    id=identifier,
                    identifier=identifier,
                    createdAt=utc_now(),
                )
            except ImportError:
                # Older Chainlit — return a plain dict
                return {"id": identifier, "identifier": identifier}
        return None

    async def create_user(self, user):
        # We don't persist users to DB — VALID_USERS dict is the source of truth
        return user

    # Stubs
    async def create_step(self, s: Dict): pass
    async def update_step(self, s: Dict): pass
    async def delete_step(self, sid: str): pass
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
    """
    Called by Chainlit's login screen.
    Returns a User object on success, None on failure.
    The sidebar ONLY appears when this returns a valid User.
    """
    if VALID_USERS.get(username) == password:
        return User(identifier=username, metadata={"role": "user"})
    return None


# ── Chainlit hooks ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Called for every brand-new chat session."""
    thread_id = cl.context.session.thread_id
    user      = cl.user_session.get("user")
    user_id   = user.identifier if user else "anonymous"

    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("user_id",   user_id)
    cl.user_session.set("msg_count", 0)

    # Register thread in DB (titled on first message)
    db_upsert_thread(thread_id, "New Chat", user_id)

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
    Called when the user clicks a thread in the sidebar.
    thread["id"] is the ORIGINAL thread_id — use this, not the session id.
    """
    thread_id = thread["id"] if isinstance(thread, dict) else thread.id
    thread_name = thread["name"] if isinstance(thread, dict) else thread.name

    user    = cl.user_session.get("user")
    user_id = user.identifier if user else "anonymous"

    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("user_id",   user_id)

    messages = db_get_messages(thread_id)
    cl.user_session.set("msg_count", len(messages))

    if messages:
        await cl.Message(
            content=f"🔄 Resuming **{thread_name}**",
            author="Assistant",
        ).send()
    else:
        await cl.Message(
            content="No previous messages found for this thread.",
            author="Assistant",
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle an incoming user message."""
    thread_id = cl.user_session.get("thread_id") or cl.context.session.thread_id
    user_id   = cl.user_session.get("user_id",   "anonymous")
    cl.user_session.set("thread_id", thread_id)

    config = {"configurable": {"thread_id": thread_id}}

    # Persist user message
    db_save_message(thread_id, "user", message.content)

    # Title thread from first message
    msg_count = cl.user_session.get("msg_count", 0)
    if msg_count == 0:
        title = message.content.strip().replace("\n", " ")
        title = title[:60] + ("…" if len(title) > 60 else "")
        db_upsert_thread(thread_id, title, user_id)
    cl.user_session.set("msg_count", msg_count + 1)

    # Stream AI response
    response_msg = cl.Message(content="", author="Assistant")
    await response_msg.send()

    try:
        result = app.invoke(
            {"messages": [HumanMessage(content=message.content)]},
            config,
        )
        last          = result["messages"][-1]
        response_text = last.content if isinstance(last, AIMessage) else str(last)

        for chunk in response_text:
            await response_msg.stream_token(chunk)

        ctx = result.get("context", "")
        if ctx:
            response_msg.elements = [
                cl.Text(name="📄 Source Chunks", content=ctx, display="side")
            ]

    except Exception as e:
        await response_msg.stream_token(f"\n\n⚠️ Error: {e}")

    await response_msg.update()

    # Persist assistant reply
    db_save_message(thread_id, "assistant", response_msg.content)


if __name__ == "__main__":
    print("Run this app with:  chainlit run chainlit_app.py -w")