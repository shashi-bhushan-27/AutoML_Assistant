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
        
        # --- RELAXED PROMPT FOR GROQ (JSON FORMAT) ---
        template = """
        You are a helpful Data Science Assistant. 
        Your goal is to suggest the best machine learning models for a given dataset and EXPLAIN WHY.
        
        CONTEXT (Optional Guidelines):
        {context}
        
        DATASET INSIGHTS:
        {question}
        
        TASK:
        Suggest 3-5 Python-compatible machine learning models that would work well for this specific data.
        Provide a specific reason for each recommendation based on the data stats (e.g. "Good for high skew", "Handles imbalanced data").
        
        OUTPUT FORMAT:
        Return ONLY a JSON object with two keys: "recommendations" (list of strings) and "reasoning" (list of strings corresponding to models).
        Example: {{"recommendations": ["XGBoost", "SVM"], "reasoning": ["Robust to outliers", "Effective for high-dimensional data"]}}
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.retriever, chain_type_kwargs={"prompt": self.prompt}
        )

    def get_recommendations(self, stats_json, supported_models=None, similar_workspaces=None):
        query = f"Dataset Statistics: {str(stats_json)}"
        
        # --- 0. HISTORICAL CONTEXT (Meta-Learning) ---
        if similar_workspaces:
            # Extract winning models
            history_str = ", ".join([f"{w['best_model']} (Score: {w['best_score']})" for w in similar_workspaces])
            query += f"\n\nHISTORICAL INTELLIGENCE:\nOn similar datasets in the past, the following models were most successful: {history_str}.\nPlease considering biasing your recommendation towards these proven winners."

        # --- 1. SMART DEFAULT (If AI fails, we use this) ---
        # We pre-calculate a safe recommendation based on the flags
        if supported_models:
             query += f"\n\nCONSTRAINT: You must ONLY suggest models from this available list: {', '.join(supported_models)}."
             
        smart_fallback = ["XGBoost", "Random Forest"]
        if isinstance(stats_json, dict) and stats_json.get('is_time_series', False):
            # If Time Series, we MUST suggest Prophet
            smart_fallback = ["Prophet", "XGBoost", "Random Forest"]
        elif isinstance(stats_json, dict) and stats_json.get('task_type') == "Regression":
            smart_fallback.append("Linear Regression")
            
        # Filter fallback against supported models if provided
        if supported_models:
             smart_fallback = [m for m in smart_fallback if m in supported_models]
             if not smart_fallback and supported_models:
                  smart_fallback = supported_models[:3] # Fallback to first 3 supported if no match

        try:
            print(f"--- SENDING STATS TO LLM: ---\n{stats_json}")
            
            response_dict = self.chain.invoke(query)
            response = response_dict['result']
            
            print(f"--- RAW LLM RESPONSE: ---\n{response}")
            
            # --- 2. PARSING ---
            # Attempt to find JSON
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    data = json.loads(json_str)
                    return data # Expects {"recommendations": [], "reasoning": []}
                except:
                     pass
            
            # Legacy Parser (Fall back to list search if JSON fails)
            match_list = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match_list:
                list_str = match_list.group(0)
                try:
                    recs = ast.literal_eval(list_str)
                    # Generate generic reasoning if missing
                    return {"recommendations": recs, "reasoning": ["Selected based on general performance."]*len(recs)}
                except:
                    pass
            
            # Fallback Keyword Search
            found_models = []
            if supported_models:
                for model in supported_models:
                     if model in response:
                          found_models.append(model)
            else:
                 if "Prophet" in response: found_models.append("Prophet")
                 if "XGBoost" in response: found_models.append("XGBoost")
                 if "Random Forest" in response: found_models.append("Random Forest")
                 if "ARIMA" in response: found_models.append("ARIMA")
                 if "LSTM" in response: found_models.append("LSTM")
            
            if found_models: 
                return {"recommendations": found_models, "reasoning": ["Extracted from AI response."]*len(found_models)}
            
            # If essentially nothing found, use smart fallback
            return {"recommendations": smart_fallback, "reasoning": ["Fallback selection."]*len(smart_fallback)}

        except Exception as e:
            print(f"LLM Error: {e}")
            return {"recommendations": smart_fallback, "reasoning": ["System fallback due to error."]*len(smart_fallback)}

if __name__ == "__main__":
    build_vector_store()