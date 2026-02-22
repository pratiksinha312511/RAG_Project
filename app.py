import os
import sqlite3
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
DB_PATH    = os.path.join(BASE_DIR, "chat_history.db")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ── 1. STATE ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str


# ── 2. COMPONENTS ──────────────────────────────────────────────────────────
if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError(
        f"Vector database not found at {FAISS_PATH}. Run ingest.py first."
    )

embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
vector_db  = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)


# ── 3. NODES ───────────────────────────────────────────────────────────────
def retrieve_docs(state: AgentState):
    """Retrieve relevant chunks from the FAISS vector store."""
    query = state["messages"][-1].content
    docs  = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    return {"context": context}


def generate_response(state: AgentState):
    """Generate an answer using conversation history + retrieved context."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a helpful assistant that answers questions strictly "
                "based on the provided document context. If the answer is not "
                "in the context, say so honestly.\n\n"
                "CONTEXT:\n{context}"
            ),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        text = chain.invoke({
            "messages": state["messages"],
            "context":  state["context"],
        })
        return {"messages": [AIMessage(content=text)]}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"messages": [AIMessage(content=f"Sorry, I encountered an error: {e}")]}


# ── 4. GRAPH ───────────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate", generate_response)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# SqliteSaver gives us PERSISTENT thread memory across restarts.
# Every thread_id gets its own isolated conversation history stored on disk.
conn   = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

app = workflow.compile(checkpointer=memory)


# ── 5. THREAD MANAGEMENT HELPERS (used by Chainlit) ────────────────────────
def list_threads() -> list[dict]:
    """
    Return all saved threads as [{"thread_id": str, "title": str}, ...].
    Title = first human message in the thread (truncated to 60 chars).
    """
    try:
        cur = conn.cursor()
        # LangGraph SqliteSaver stores checkpoints in 'checkpoints' table
        cur.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        rows = cur.fetchall()
        threads = []
        for (tid,) in rows:
            title = _get_thread_title(tid)
            threads.append({"thread_id": tid, "title": title})
        return threads
    except Exception:
        return []


def _get_thread_title(thread_id: str) -> str:
    """Fetch the first user message of a thread to use as its display title."""
    try:
        config  = {"configurable": {"thread_id": thread_id}}
        history = app.get_state(config)
        msgs    = history.values.get("messages", [])
        for m in msgs:
            if isinstance(m, HumanMessage) and m.content.strip():
                title = m.content.strip().replace("\n", " ")
                return title[:60] + ("…" if len(title) > 60 else "")
    except Exception:
        pass
    return thread_id  # fallback


def get_thread_history(thread_id: str) -> list[BaseMessage]:
    """Return the full message list for a thread."""
    try:
        config  = {"configurable": {"thread_id": thread_id}}
        state   = app.get_state(config)
        return state.values.get("messages", [])
    except Exception:
        return []


# ── 6. CLI (optional) ──────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "cli_session"}}
    print("\n--- PDF Chatbot (type 'quit' to exit) ---")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        result   = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        last_msg = result["messages"][-1]
        if isinstance(last_msg, AIMessage):
            print(f"\nAI: {last_msg.content}")