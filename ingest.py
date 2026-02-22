import os
<<<<<<< HEAD
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, DirectoryLoader
=======
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
>>>>>>> ff0fca72db1cc08e4d457370de0a0540508b2ad2
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


<<<<<<< HEAD
# Create directories if they don't exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

def update_vector_db():
    # 1. Get list of files
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("No new PDF files found.")
=======
def create_vector_db():
    # 1. Load PDFs from the directory
    print(f"Loading PDFs from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyMuPDFLoader)
    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
>>>>>>> ff0fca72db1cc08e4d457370de0a0540508b2ad2
        return

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = None

    # 2. Load existing DB if it exists
    if os.path.exists(DB_FAISS_PATH):
        vector_db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for file_name in pdf_files:
        file_path = os.path.join(DATA_PATH, file_name)
        print(f"\n--- Processing: {file_name} ---")

        try:
            # 3. Load PDF and immediately extract data
            # Using a temporary variable to ensure the loader object doesn't persist
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            # 4. Split and Add to DB
            chunks = text_splitter.split_documents(documents)
            
            if vector_db is None:
                vector_db = FAISS.from_documents(chunks, embeddings)
            else:
                vector_db.add_documents(chunks)
            
            # Save DB after every file to be safe
            vector_db.save_local(DB_FAISS_PATH)
            print(f"Added to Vector DB: {file_name}")

            # 5. CLEAR FILE HANDLES
            # We explicitly delete the loader and documents to release the file lock
            del loader
            del documents
            
            # 6. ATTEMPT TO MOVE
            # On Windows, we need to wait a tiny bit for the OS to acknowledge the file is free
            time.sleep(1.5) 
            
            dest_path = os.path.join(PROCESSED_PATH, file_name)
            if os.path.exists(dest_path):
                dest_path = os.path.join(PROCESSED_PATH, f"{int(time.time())}_{file_name}")

            shutil.move(file_path, dest_path)
            print(f"Successfully moved to 'processed' folder.")

        except PermissionError:
            print(f"ERROR: Could not move {file_name}. Is the PDF open in another program (Chrome/Adobe)?")
        except Exception as e:
            print(f"CRITICAL ERROR: {str(e)}")

    print("\nAll tasks finished.")

if __name__ == "__main__":
    update_vector_db()