# evaluate.py

# --- Imports ---
import warnings
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
import os # Import the os module

# Import components for our RAG pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Suppress all warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Constants ---
DB_PATH = "chroma_db_local"
GOLDEN_DATASET_PATH = "golden_dataset.csv"

def build_rag_pipeline(llm):
    """
    Builds the RAG pipeline using a provided LLM.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain, embedding_model

def main():
    """
    Main function to run the evaluation.
    """
    load_dotenv()

    # --- Set up the Gemini LLM ---
    print("--- Initializing Gemini LLM ---")
    # Explicitly use the API key from the .env file.
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
        
    # FIX: Use the correct model name for the Google AI Studio API
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
    print("--- Gemini LLM Initialized ---")

    print("\n--- Building RAG Pipeline with Gemini ---")
    qa_chain, embedding_model = build_rag_pipeline(gemini_llm)
    print("--- RAG Pipeline Built Successfully ---")

    # --- Prepare Data for Evaluation ---
    print("\n--- Preparing Evaluation Dataset ---")
    golden_df = pd.read_csv(GOLDEN_DATASET_PATH)
    
    results = []
    for index, row in golden_df.iterrows():
        question = row['question']
        print(f"Processing question: '{question}'")
        result = qa_chain.invoke({"query": question})
        
        results.append({
            "question": question,
            "answer": result["result"],
            "contexts": [doc.page_content for doc in result["source_documents"]],
            "ground_truth": row["ground_truth"]
        })

    results_dataset = Dataset.from_list(results)
    print("--- Evaluation Dataset Prepared ---")

    # --- Run the Evaluation ---
    print("\n--- Running RAGAs Evaluation with Gemini ---")
    metrics = [
        faithfulness,
        answer_relevancy,
    ]
    
    ragas_llm = LangchainLLMWrapper(gemini_llm)

    result = evaluate(
        dataset=results_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=embedding_model
    )
    
    results_df = result.to_pandas()

    print("\n--- Evaluation Results ---")
    print(results_df)
    print("--------------------------")

if __name__ == "__main__":
    main()
