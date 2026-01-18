import os
import json
import re
import ast
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# V1.0 COMPATIBLE IMPORTS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base", "ml_rules.txt")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "knowledge_base", "faiss_index")

def build_vector_store():
    print("Loading Knowledge Base...")
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print("Error: Rules file missing.")
        return

    loader = TextLoader(KNOWLEDGE_BASE_PATH)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    print("Creating Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print("Vector Store Saved!")

from functools import lru_cache

@lru_cache(maxsize=1)
def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError("Vector Store not found! Run llm_rag_core.py directly first.")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 3})

load_dotenv()

class ModelAdvisor:
    def __init__(self):
        self.retriever = get_rag_chain()
        
        # Initialize Groq Chat Model
        # Ensure GROQ_API_KEY is in your .env or environment variables
        self.llm = ChatGroq(
            temperature=0.3, 
            model_name="llama-3.1-8b-instant"
        )
        
        # --- RELAXED PROMPT FOR GROQ ---
        template = """
        You are a helpful Data Science Assistant. 
        Your goal is to suggest the best machine learning models for a given dataset.
        
        CONTEXT (Optional Guidelines):
        {context}
        
        DATASET INSIGHTS:
        {question}
        
        TASK:
        Suggest 3-5 Python-compatible machine learning models that would work well for this specific data.
        You can be creative but realistic. Consider the size of data and the nature of the target variable.
        
        OUTPUT FORMAT:
        Return ONLY a Python list of strings.
        Example: ["Random Forest", "Neural Network", "XGBoost"]
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.retriever, chain_type_kwargs={"prompt": self.prompt}
        )

    def get_recommendations(self, stats_json):
        query = f"Dataset Statistics: {str(stats_json)}"
        
        # --- 1. SMART DEFAULT (If AI fails, we use this) ---
        # We pre-calculate a safe recommendation based on the flags
        smart_fallback = ["XGBoost", "Random Forest"]
        if isinstance(stats_json, dict) and stats_json.get('is_time_series', False):
            # If Time Series, we MUST suggest Prophet
            smart_fallback = ["Prophet", "XGBoost", "Random Forest"]
        elif isinstance(stats_json, dict) and stats_json.get('task_type') == "Regression":
            smart_fallback.append("Linear Regression")
            
        try:
            print(f"--- SENDING STATS TO LLM: ---\n{stats_json}")
            
            response_dict = self.chain.invoke(query)
            response = response_dict['result']
            
            print(f"--- RAW LLM RESPONSE: ---\n{response}")
            
            # --- 2. PARSING ---
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match:
                list_str = match.group(0)
                try:
                    recs = ast.literal_eval(list_str)
                    return recs
                except:
                    return json.loads(list_str)
            
            # Fallback Keyword Search
            found_models = []
            if "Prophet" in response: found_models.append("Prophet")
            if "XGBoost" in response: found_models.append("XGBoost")
            if "Random Forest" in response: found_models.append("Random Forest")
            if "ARIMA" in response: found_models.append("ARIMA")
            if "LSTM" in response: found_models.append("LSTM")
            
            if found_models: 
                return found_models
            
            # If essentially nothing found, use smart fallback
            return smart_fallback

        except Exception as e:
            print(f"LLM Error: {e}")
            return smart_fallback

if __name__ == "__main__":
    build_vector_store()