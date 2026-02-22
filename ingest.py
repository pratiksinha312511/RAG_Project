import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def create_vector_db():
    # 1. Load PDFs from the directory
    print(f"Loading PDFs from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyMuPDFLoader)
    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    if not documents:
        print("No documents found. Exiting.")
        return

    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks.")
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return

    if not texts:
        print("No text chunks generated. Exiting.")
        return

    # 3. Create Embeddings & Store in FAISS
    print("Generating embeddings and building FAISS index...")
    try:
        embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        print(f"Error generating embeddings or building FAISS index: {e}")
        return

    # 4. Save the database locally
    try:
        db.save_local(DB_FAISS_PATH)
        print(f"Vector database saved to {DB_FAISS_PATH}")
    except Exception as e:
        print(f"Error saving vector database: {e}")
        return

if __name__ == "__main__":
    create_vector_db()