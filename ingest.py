import os
import shutil
import time

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


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data")

# processed folder per module (prevents mixing)
MODULE_NAME = os.environ.get("MODULE", "default")
PROCESSED_PATH = os.path.join(BASE_DIR, "processed", MODULE_NAME)

VECTORSTORE_BASE = os.path.join(BASE_DIR, "vectorstore")
DB_FAISS_PATH = os.path.join(VECTORSTORE_BASE, MODULE_NAME)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ============================================================
# Create directories
# ============================================================

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(VECTORSTORE_BASE, exist_ok=True)
os.makedirs(DB_FAISS_PATH, exist_ok=True)


# ============================================================
# Loader selector
# ============================================================

def get_loader(file_path: str):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyMuPDFLoader(file_path)

    elif ext in [".docx", ".doc"]:
        return Docx2txtLoader(file_path)

    elif ext in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(file_path, mode="elements")

    elif ext == ".json":
        return JSONLoader(
            file_path=file_path,
            jq_schema=".",
            text_content=False
        )

    return None


# ============================================================
# Main ingestion logic
# ============================================================

def update_vector_db():

    print(f"\n===== INGESTING FOR MODULE: {MODULE_NAME} =====\n")

    all_files = [
        f for f in os.listdir(DATA_PATH)
        if os.path.isfile(os.path.join(DATA_PATH, f))
    ]

    if not all_files:

        print("No new files found in data folder")
        return


    embeddings = FastEmbedEmbeddings(
        model_name=EMBEDDING_MODEL
    )


    index_file = os.path.join(DB_FAISS_PATH, "index.faiss")


    # ========================================================
    # Load existing vectorstore safely
    # ========================================================

    if os.path.exists(index_file):

        print("Loading existing vectorstore...")

        vector_db = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        print("Creating new vectorstore...")

        vector_db = None


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )


    # ========================================================
    # Process each file
    # ========================================================

    for file_name in all_files:

        file_path = os.path.join(DATA_PATH, file_name)

        loader = get_loader(file_path)


        if loader is None:

            print(f"Skipping unsupported file: {file_name}")
            continue


        print(f"\nProcessing: {file_name}")


        try:

            documents = loader.load()

            chunks = splitter.split_documents(documents)


            # add module metadata
            for chunk in chunks:
                chunk.metadata["module"] = MODULE_NAME


            if vector_db is None:

                vector_db = FAISS.from_documents(
                    chunks,
                    embeddings
                )

            else:

                vector_db.add_documents(chunks)


            vector_db.save_local(DB_FAISS_PATH)


            print(f"Indexed {len(chunks)} chunks")


            # =================================================
            # Move file to processed
            # =================================================

            del loader
            del documents

            time.sleep(1)

            dest_path = os.path.join(PROCESSED_PATH, file_name)


            if os.path.exists(dest_path):

                dest_path = os.path.join(
                    PROCESSED_PATH,
                    f"{int(time.time())}_{file_name}"
                )


            shutil.move(file_path, dest_path)


            print("Moved to processed folder")


        except Exception as e:

            print(f"Error processing {file_name}: {e}")


    print("\n===== INGEST COMPLETE =====\n")


# ============================================================

if __name__ == "__main__":

    update_vector_db()
