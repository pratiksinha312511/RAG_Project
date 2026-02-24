import os
import sqlite3

from typing import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from langchain_core.tools import StructuredTool, tool

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
load_dotenv()

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

    def make_func(retriever):
        def search(query: str) -> str:
            print(f"\n[TOOL CALLED]: {module_name}\n")
            docs = retriever.invoke(query)
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
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
    streaming=True
)
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