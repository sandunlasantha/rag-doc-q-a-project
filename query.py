# query.py

# --- Imports ---
# For creating the local embeddings
from langchain_huggingface import HuggingFaceEmbeddings
# For creating the vector database
from langchain_community.vectorstores import Chroma
# For the QA chain
from langchain.chains import RetrievalQA
# For the local LLM
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import warnings

# Suppress all warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Helper Class for Colors ---
class BColors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ENDC = '\033[0m' # Resets the color

# --- Constants ---
DB_PATH = "chroma_db_local"

def main():
    """
    This is the main function where our script will run.
    """
    # --- Load the local database ---
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embedding_model
    )

    # --- Set up the local LLM for generation ---
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=512
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)

    # --- Create the Retriever ---
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # --- Create the QA Chain ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # --- Ask a Question and Print the Final Result ---
    question = "what do you know just summarise this pdf in 50 words"
    
    # Run the chain to get the result
    result = qa_chain.invoke({"query": question})

    # Print only the question and answer with colors
    print(f"\n{BColors.BLUE}Question: {question}{BColors.ENDC}")
    print(f"{BColors.GREEN}Answer: {result['result']}{BColors.ENDC}\n")


# This is a standard Python practice.
if __name__ == "__main__":
    main()
