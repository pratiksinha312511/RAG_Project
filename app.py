import os
import inspect
import sqlite3

from typing import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from langchain_core.tools import StructuredTool, tool

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
load_dotenv()

# ── LangSmith tracing ────────────────────────────────────────────────────────
# Reads LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT from .env
# No code changes needed beyond this import — all LangGraph runs are traced.
# Dashboard: https://smith.langchain.com
try:
    from langsmith import traceable  # noqa: F401  triggers SDK initialisation
    import os
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        print(f"[LangSmith] Tracing enabled → project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    else:
        print("[LangSmith] LANGCHAIN_TRACING_V2 not set — tracing disabled")
except ImportError:
    print("[LangSmith] SDK not installed. Run: pip install langsmith")
else:
    # Try to create a LangSmith tracer and callback manager for LangChain
    try:
        from langchain.tracers import LangSmithTracer
        from langchain.callbacks.manager import CallbackManager
        tracer = LangSmithTracer(project_name=os.getenv("LANGCHAIN_PROJECT", "default"))
        cb_manager = CallbackManager(tracers=[tracer])
        print("[LangSmith] LangSmithTracer initialized and callback manager created")
    except Exception:
        tracer = None
        cb_manager = None

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_BASE = os.path.join(BASE_DIR, "vectorstore")
SQLITE_PATH = os.path.join(BASE_DIR, "chat_history.db")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ============================================================
# LOAD VECTORSTORES
# ============================================================

embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
retrievers = {}

for module in os.listdir(VECTORSTORE_BASE):
    module_path = os.path.join(VECTORSTORE_BASE, module)
    index_file = os.path.join(module_path, "index.faiss")
    if os.path.exists(index_file):
        print(f"Loading vectorstore: {module}")
        db = FAISS.load_local(
            module_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retrievers[module.lower()] = db.as_retriever(search_kwargs={"k": 4})

if not retrievers:
    raise Exception("No vectorstores found. Run ingest.py first.")

# ============================================================
# CREATE TOOLS
# ============================================================

tools = []

for module_name, retriever in retrievers.items():

    def make_func(retriever, module_name=module_name):
        def search(query: str) -> str:
            print(f"\n[TOOL CALLED]: {module_name}\n")
            try:
                invoke_fn = getattr(retriever, "invoke", None)
                if invoke_fn is not None:
                    sig = None
                    try:
                        sig = inspect.signature(invoke_fn)
                    except Exception:
                        sig = None
                    if sig and "callback_manager" in sig.parameters and cb_manager is not None:
                        docs = invoke_fn(query, callback_manager=cb_manager)
                    else:
                        docs = invoke_fn(query)
                elif hasattr(retriever, "get_relevant_documents"):
                    get_docs = getattr(retriever, "get_relevant_documents")
                    sig = None
                    try:
                        sig = inspect.signature(get_docs)
                    except Exception:
                        sig = None
                    if sig and "run_manager" in sig.parameters and cb_manager is not None:
                        docs = get_docs(query, run_manager=cb_manager)
                    else:
                        docs = get_docs(query)
                else:
                    raise AttributeError("Retriever has no known invoke/get_relevant_documents method")
            except Exception as e:
                print(f"[TOOL ERROR]: {e}")
                docs = []
            return "\n\n".join(d.page_content for d in docs)
        return search

    func = make_func(retriever)
    tool = StructuredTool.from_function(
        func=func,
        name=f"{module_name}_search",
        description=f"Search inside {module_name} knowledge base"
    )

    tools.append(tool)

# ============================================================
# LLM BLOCK ──────────────────────────────────────
# Model options (pick one, set the repo_id below):
#   - "mistralai/Mistral-7B-Instruct-v0.3"   (good general-purpose)
#   - "meta-llama/Meta-Llama-3-8B-Instruct"  (strong reasoning, needs HF access)
#   - "HuggingFaceH4/zephyr-7b-beta"         (instruction-tuned, no gating)
#   - "Qwen/Qwen2.5-7B-Instruct"             (strong multilingual)
# ============================================================

_hf_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",   # ← swap repo_id here
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.1,
    streaming=True,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

# Attach callback manager to LLM if available so traces include LLM activity
llm_kwargs = {}
try:
    if cb_manager:
        llm_kwargs["callback_manager"] = cb_manager
    llm_kwargs["verbose"] = True
except NameError:
    pass

llm = ChatHuggingFace(llm=_hf_endpoint, **llm_kwargs)
# ── END BLOCK ────────────────────────────────────────

llm_with_tools = llm.bind_tools(tools)

# ============================================================
# SQLITE MEMORY
# ============================================================
conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)

# ============================================================
# THREAD HELPERS
# ============================================================

def list_threads():
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
    return [row[0] for row in cursor.fetchall()]

def get_thread_history(thread_id):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT checkpoint FROM checkpoints WHERE thread_id=?",
        (thread_id,)
    )
    return cursor.fetchall()

def _get_thread_title(thread_id):
    history = get_thread_history(thread_id)
    if not history:
        return "New Chat"
    return thread_id

# ============================================================
# STATE
# ============================================================
class AgentState(TypedDict):

    messages: Annotated[Sequence[BaseMessage], add_messages]

# ============================================================
# NODES
# ============================================================

def assistant_node(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)


# ============================================================
# GRAPH
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("assistant", assistant_node)

workflow.add_node("tools", tool_node)

workflow.set_entry_point("assistant")

workflow.add_conditional_edges(
    "assistant",
    tools_condition,
)

workflow.add_edge("tools", "assistant")
app = workflow.compile()
