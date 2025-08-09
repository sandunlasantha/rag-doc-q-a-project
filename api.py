# api.py

# --- Imports ---
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import torch

# Import components from our RAG pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import PromptTemplate
from prompt import RAG_PROMPT_TEMPLATE

# Suppress all warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Constants ---
DB_PATH = "chroma_db_local"

# --- FastAPI App Initialization ---
# Load environment variables (still good practice for other potential keys)
load_dotenv()

# Create the FastAPI app instance
app = FastAPI(
    title="RAG Q&A API",
    description="An API for asking questions to a RAG pipeline based on a document.",
    version="1.0.0",
)

# --- Data Models for API ---
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
    """
    global rag_chain
    print("--- Loading models and building RAG pipeline... ---")

    # --- Set up the Local LLM for Generation ---
    # NEW: Using a powerful, non-gated model to avoid authentication issues.
    model_id = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=512
    )
    llm = HuggingFacePipeline(pipeline=pipe)

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
    prompt = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    print("--- RAG Pipeline Ready! ---")

# --- API Endpoint ---
@app.post("/ask")
def ask_question(query: Query):
    """
    This is the main endpoint to ask a question.
    """
    if not rag_chain:
        return {"error": "RAG pipeline is not ready."}

    result = rag_chain.invoke({"query": query.question})

    return {
        "answer": result["result"],
        "source_documents": [
            {"source": doc.metadata.get('source'), "page": doc.metadata.get('page')} 
            for doc in result["source_documents"]
        ]
    }
