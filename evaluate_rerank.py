# evaluate_rerank.py

# --- Imports ---
import warnings
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
import os
import time
import sys
import threading
import itertools

# Import components for our RAG pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
# NEW: Import the Google Generative AI model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# Suppress all warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Helper Class for Colors & Animation ---
class BColors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'

class Spinner:
    def __init__(self, message="Processing..."):
        self.spinner = itertools.cycle(['-', '/', '|', '\\'])
        self.message = message
        self.stop_running = threading.Event()
        self.spinner_thread = threading.Thread(target=self._spin)

    def _spin(self):
        while not self.stop_running.is_set():
            sys.stdout.write(f"\r{BColors.YELLOW}{self.message} {next(self.spinner)}{BColors.ENDC}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r') # Clear the line
        sys.stdout.flush()

    def __enter__(self):
        self.spinner_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop_running.set()
        self.spinner_thread.join()

def log_step(message):
    print(f"\n{BColors.CYAN}➡️ {message}{BColors.ENDC}")

# --- Constants ---
DB_PATH = "chroma_db_local"
GOLDEN_DATASET_PATH = "golden_dataset.csv"

def build_reranking_pipeline(llm):
    """
    Builds the RAG pipeline with a re-ranking step using a provided LLM.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embedding_model
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    cross_encoder = HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-base')
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True
    )
    return qa_chain, embedding_model

def main():
    load_dotenv()

    log_step("Initializing Gemini LLM for generation and evaluation...") 
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    
    # Using the latest Gemini Flash model for speed and cost-effectiveness
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
    print(f"{BColors.GREEN}✅ Gemini LLM Initialized.{BColors.ENDC}")

    log_step("Building the advanced RAG pipeline with a re-ranking step...")
    qa_chain, embedding_model = build_reranking_pipeline(gemini_llm)
    print(f"{BColors.GREEN}✅ RAG Pipeline Built Successfully.{BColors.ENDC}")

    log_step("Loading the golden dataset to begin evaluation...")
    golden_df = pd.read_csv(GOLDEN_DATASET_PATH)
    print(f"{BColors.GREEN}✅ Golden Dataset loaded with {len(golden_df)} questions.{BColors.ENDC}")

    log_step("Generating answers for each question in the dataset...")
    results = []
    for index, row in golden_df.iterrows():
        question = row['question']
        print(f"\n{BColors.WHITE}Processing question {index+1}/{len(golden_df)}: '{question}'{BColors.ENDC}")
        
        with Spinner("   Thinking..."):
            result = qa_chain.invoke({"query": question})
        
        results.append({
            "question": question,
            "answer": result["result"],
            "contexts": [doc.page_content for doc in result["source_documents"]],
            "ground_truth": row["ground_truth"]
        })
        print(f"{BColors.GREEN}   Answer generated.{BColors.ENDC}")
        
        if index < len(golden_df) - 1:
            print(f"{BColors.YELLOW}   Pausing for 20 seconds to respect API rate limits...{BColors.ENDC}")
            time.sleep(20)

    results_dataset = Dataset.from_list(results)
    print(f"\n{BColors.GREEN}✅ All answers generated and prepared for evaluation.{BColors.ENDC}")

    log_step("Running RAGAs evaluation using Gemini as the judge...")
    metrics = [faithfulness, answer_relevancy]
    ragas_llm = LangchainLLMWrapper(gemini_llm)

    with Spinner("   Judging the results... This may take a few minutes."):
        result = evaluate(
            dataset=results_dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=embedding_model
        )
    
    results_df = result.to_pandas()

    print(f"\n\n{BColors.BLUE}--- 📊 Final Evaluation Results (with Re-ranking) ---{BColors.ENDC}")
    print(results_df)
    print(f"{BColors.BLUE}----------------------------------------------------{BColors.ENDC}")

if __name__ == "__main__":
    main()
