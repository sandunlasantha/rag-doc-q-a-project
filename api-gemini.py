# api.py

# --- Imports ---
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Import components from our RAG pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# Suppress all warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Constants ---
DB_PATH = "chroma_db_local"

# --- FastAPI App Initialization ---
# Load environment variables to get the API key
load_dotenv()

# Create the FastAPI app instance
app = FastAPI(
    title="RAG Q&A API",
    description="An API for asking questions to a RAG pipeline based on a document.",
    version="1.0.0",
)

# --- Data Models for API ---
# This defines the structure of the request body for the /ask endpoint
class Query(BaseModel):
    question: str

# This will hold our RAG chain so it's loaded only once
rag_chain = None

# --- Application Startup Event ---
@app.on_event("startup")
def load_rag_pipeline():
    """
    This function is called when the FastAPI application starts.
    It loads all the models and sets up the RAG pipeline.
    This is a crucial optimization to avoid reloading models on every request.
    """
    global rag_chain
    print("--- Loading models and building RAG pipeline... ---")

    # Initialize the Gemini LLM
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=gemini_api_key)

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the vector store
    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embedding_model
    )

    # Set up the re-ranking logic
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    cross_encoder = HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-base')
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Create the final QA Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True
    )
    print("--- RAG Pipeline Ready! ---")

# --- API Endpoint ---
@app.post("/ask")
def ask_question(query: Query):
    """
    This is the main endpoint to ask a question.
    It takes a question in a JSON request and returns the answer.
    """
    if not rag_chain:
        return {"error": "RAG pipeline is not ready."}

    # Invoke the RAG chain with the user's question
    result = rag_chain.invoke({"query": query.question})

    # Return the answer and the source documents
    return {
        "answer": result["result"],
        "source_documents": [
            {"source": doc.metadata.get('source'), "page": doc.metadata.get('page')} 
            for doc in result["source_documents"]
        ]
    }

