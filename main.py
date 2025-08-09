# main.py

# --- Imports ---
# We no longer need dotenv for this local approach
# from dotenv import load_dotenv 

# For loading the PDF
from langchain_community.document_loaders import PyPDFLoader
# For splitting the text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# NEW: For creating the local embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# For creating the vector database
from langchain_community.vectorstores import Chroma

# --- Constants ---
PDF_PATH = "google-2023-environmental-report.pdf"
DB_PATH = "chroma_db_local" # Using a new DB path for our local model

def main():
    """
    This is the main function where our script will run.
    """
    # --- Step 1: Load and Chunk the Document ---
    print("--- Starting Document Loading and Chunking ---")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} pages from the PDF.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split the document into {len(chunks)} chunks.")

    # --- Step 2: Create Embeddings using a Local Model and Store in ChromaDB ---
    print("\n--- Creating Vector Store with Local HuggingFace Model ---")
    
    # NEW: We are now creating an instance of HuggingFaceEmbeddings.
    # We specify the model name from the Hugging Face model hub.
    # 'all-MiniLM-L6-v2' is a popular, small, and fast model that runs well on a CPU.
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # This part of the code remains exactly the same!
    # It takes our chunks, uses the new local embedding model,
    # and stores everything in our ChromaDB.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

    print(f"--- Vector Store Created and Saved to '{DB_PATH}' ---")


# This is a standard Python practice. It means the main() function will only run
# when you execute this script directly.
if __name__ == "__main__":
    main()
