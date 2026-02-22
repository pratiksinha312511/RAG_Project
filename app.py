import sqlite3
import os
from typing import Annotated, TypedDict, List

from langchain_community.chat_models import ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LLM
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from typing import Optional, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI





EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"        # must match the ingest model
DEFAULT_INDEX_DIR = "./faiss_index"
DEFAULT_TOP_K = 4

load_dotenv()

# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    # Annotated with add_messages so history is appended, not overwritten
    # This automatically tracks roles (HumanMessage vs AIMessage)
    messages: Annotated[list[BaseMessage], add_messages]
    context: str # To store retrieved PDF chunks

# --- 2. INITIALIZE COMPONENTS ---
# Load FAISS database created by ingest.py
embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
if not os.path.exists("vectorstore/db_faiss"):
    print("Error: Vector database not found. Run ingest.py first.")
    exit()

vector_db = FAISS.load_local(
    "vectorstore/db_faiss", 
    embeddings, 
    allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=1.8)

# --- 3. DEFINE NODES ---

def retrieve_docs(state: AgentState):
    """
    Step 1: Get the latest user message and fetch relevant PDF chunks.
    """
    last_message = state["messages"][-1].content
    
    # Advanced tip: In a production app, you'd use the LLM to 'rewrite' 
    # the query based on history before retrieving. 
    # For now, we fetch docs based on the latest input.
    docs = retriever.invoke(last_message)
    formatted_docs = "\n\n".join([d.page_content for d in docs])
    
    return {"context": formatted_docs}

def generate_response(state: AgentState):
    """
    Step 2: Use history + context + user query to generate a response.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are helpful assistant.\nCONTEXT:\n{context}"),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt_template | llm | StrOutputParser()

    try:
        response_text = chain.invoke({
            "messages": state["messages"],
            "context": state["context"]
        })

        response = AIMessage(content=response_text)

    except Exception as e:

        import traceback
        print(f"DEBUG Exception: {type(e).__name__}")
        print(f"DEBUG Message: {str(e)}")
        traceback.print_exc()

        response = AIMessage(
            content="Sorry, I encountered an error."
        )

    return {"messages": [response]}

# --- 4. BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate", generate_response)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Persistence: SQLite to store threads
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
memory = SqliteSaver(conn)

app = workflow.compile(checkpointer=memory)

# --- 5. INTERACTIVE CHATBOT INTERFACE ---
def run_chatbot():
    # thread_id separates different users or sessions
    config = {"configurable": {"thread_id": "session_1"}}
    
    print("\n--- PDF Chatbot Initialized (Type 'quit' to exit) ---")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        # Prepare the input for the graph
        # We pass a HumanMessage object to satisfy the Role requirement
        input_data = {"messages": [HumanMessage(content=user_input)]}
        
        try:
            # Use invoke instead of stream to avoid StopIteration issues
            result = app.invoke(input_data, config)
            
            # Get the latest message from the result
            last_msg = result["messages"][-1]
            
            if isinstance(last_msg, AIMessage):
                print(f"\nAI: {last_msg.content}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_chatbot()