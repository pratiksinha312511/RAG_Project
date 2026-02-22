import os
# Loaders
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    Docx2txtLoader, 
    UnstructuredExcelLoader,
    JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
import shutil
import time


# Configuration
# Use absolute paths to be safe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
PROCESSED_PATH = os.path.join(BASE_DIR, "processed")
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# Create directories if they don't exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)


def get_loader(file_path: str):
    """Returns the appropriate loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return PyMuPDFLoader(file_path)
    elif ext == ".docx" or ext == ".doc":
        return Docx2txtLoader(file_path)
    elif ext == ".xlsx" or ext == ".xls":
        # Requires 'unstructured' and 'openpyxl'
        return UnstructuredExcelLoader(file_path, mode="elements")
    elif ext == ".json":
        # This basic schema loads all text values from the JSON
        # Requires 'jq' package installed in the system/env
        return JSONLoader(
            file_path=file_path,
            jq_schema=".[]",
            text_content=False
        )
    else:
        return None
    

def update_vector_db():
    # 1. Get list of all files in data folder
    all_files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    
    if not all_files:
        print("No new files found in 'data/' folder.")
        return

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = None

    # 2. Load existing DB if it exists
    if os.path.exists(DB_FAISS_PATH):
        print("Loading existing FAISS index...")
        vector_db = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for file_name in all_files:
        file_path = os.path.join(DATA_PATH, file_name)
        loader = get_loader(file_path)

        if loader is None:
            print(f"Skipping unsupported file type: {file_name}")
            continue

        print(f"\n--- Processing {os.path.splitext(file_name)[1].upper()}: {file_name} ---")

        try:
            # 3. Extract and Split
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            
            # 4. Update Vector Store
            if vector_db is None:
                vector_db = FAISS.from_documents(chunks, embeddings)
            else:
                vector_db.add_documents(chunks)
            
            # Save after every successful file
            vector_db.save_local(DB_FAISS_PATH)
            print(f"Indexed {len(chunks)} chunks from {file_name}")

            # 5. Cleanup and Move
            # Explicitly delete objects to release file locks (important for Windows)
            del loader
            del documents
            
            time.sleep(1.5) # Buffer for OS to release file handle
            
            dest_path = os.path.join(PROCESSED_PATH, file_name)
            if os.path.exists(dest_path):
                dest_path = os.path.join(PROCESSED_PATH, f"{int(time.time())}_{file_name}")

            shutil.move(file_path, dest_path)
            print(f"File moved to 'processed' folder.")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    print("\nAll new documents have been integrated into the Vector DB.")

if __name__ == "__main__":
    update_vector_db()