"""
Model recommendation: FAISS retrieval over knowledge_base/ml_rules.txt + a Groq LLM.

* The Groq model comes from the ``GROQ_MODEL`` environment variable (default
  ``openai/gpt-oss-120b``). The previously hard-coded ``llama-3.3-70b-versatile`` is no
  longer served, so every call silently fell back to defaults.
* ``check_llm_health`` looks the model up with the Groq API (no tokens used) and is shown
  as the LLM status chip in the UI.
* ``get_recommendations`` always returns ``source`` ("llm" or "fallback") and ``error``;
  the output is validated with a pydantic schema (``recommendations: list[str]``,
  ``reasoning: list[str]``) with one retry on invalid output.
* The retrieved rules are returned so the UI can show them.

Run ``python app_backend/llm_rag_core.py`` to rebuild the FAISS index; the app also
rebuilds it automatically when ml_rules.txt changes.
"""
import hashlib
import json
import logging
import os
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator

load_dotenv()
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base", "ml_rules.txt")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "knowledge_base", "faiss_index")
RULES_HASH_FILE = os.path.join(VECTOR_STORE_PATH, "rules.sha256")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_K = 3
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
MAX_RECOMMENDATIONS = 6
FALLBACK_REASON_ERROR = "System fallback due to error."
FALLBACK_REASON_INVALID = "Fallback selection."


def get_llm_model() -> str:
    return os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL


def get_report_model() -> str:
    return os.environ.get("GROQ_REPORT_MODEL") or get_llm_model()


# ── health check ────────────────────────────────────────────────────────────
_HEALTH_CACHE: Dict[str, Dict[str, Any]] = {}
HEALTH_TTL_S = 300


def check_llm_health(model: str = None, force: bool = False, timeout: float = 8.0) -> Dict[str, Any]:
    """Is ``model`` served for this API key? Uses the Groq models endpoint (no completion tokens)."""
    model = model or get_llm_model()
    cached = _HEALTH_CACHE.get(model)
    if cached and not force and time.time() - cached["checked_at"] < HEALTH_TTL_S:
        return cached
    result = {"model": model, "ok": False, "error": None, "checked_at": time.time(), "latency_ms": None}
    if not os.environ.get("GROQ_API_KEY"):
        result["error"] = "GROQ_API_KEY is not set"
    else:
        try:
            import groq

            t0 = time.perf_counter()
            groq.Groq(timeout=timeout, max_retries=0).models.retrieve(model)
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            result["ok"] = True
        except Exception as exc:  # NotFoundError (404), AuthenticationError, connection errors ...
            status = getattr(exc, "status_code", None)
            result["error"] = (f"model '{model}' is not served for this key (HTTP 404)" if status == 404
                               else f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")
                               + f": {str(exc)[:160]}")
    _HEALTH_CACHE[model] = result
    return result


# ── vector store ────────────────────────────────────────────────────────────
def _rules_hash() -> str:
    with open(KNOWLEDGE_BASE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store():
    """(Re)build the FAISS index from ml_rules.txt (one chunk per rule)."""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    with open(KNOWLEDGE_BASE_PATH, encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    docs = [Document(page_content=c, metadata={"rule": _rule_name(c)}) for c in chunks]
    store = FAISS.from_documents(docs, _embeddings())
    store.save_local(VECTOR_STORE_PATH)
    with open(RULES_HASH_FILE, "w") as f:
        f.write(_rules_hash())
    logger.info("Vector store rebuilt with %d rule chunks", len(docs))
    return store


def _rule_name(text: str) -> str:
    m = re.search(r"\[RULE:\s*([^\]]+)\]", text)
    return m.group(1).strip() if m else text.strip().splitlines()[0][:60]


def _index_is_current() -> bool:
    if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")) or not os.path.exists(RULES_HASH_FILE):
        return False
    with open(RULES_HASH_FILE) as f:
        return f.read().strip() == _rules_hash()


@lru_cache(maxsize=1)
def get_rag_chain():
    """Retriever over the rules (k=3). Rebuilds the index if it is missing or older than ml_rules.txt."""
    from langchain_community.vectorstores import FAISS

    if not _index_is_current():
        logger.info("FAISS index missing or stale; rebuilding")
        store = build_vector_store()
    else:
        # the index is created locally by build_vector_store (not downloaded), so its pickle is trusted
        store = FAISS.load_local(VECTOR_STORE_PATH, _embeddings(), allow_dangerous_deserialization=True)
    return FactsRetriever(store=store, k=RETRIEVAL_K)


try:
    from langchain_core.retrievers import BaseRetriever

    class FactsRetriever(BaseRetriever):
        """Embeds only the 'RETRIEVAL FACTS:' line of the query (if present), not the whole stats dump,
        so rules are retrieved for the conditions that hold instead of for column names."""

        store: Any
        k: int = RETRIEVAL_K

        def _get_relevant_documents(self, query: str, *, run_manager=None):
            m = re.search(r"RETRIEVAL FACTS:\s*(.+)", query)
            return self.store.similarity_search(m.group(1) if m else query, k=self.k)

        @property
        def vectorstore(self):
            return self.store
except ImportError:  # pragma: no cover
    FactsRetriever = None


def retrieval_facts(stats: Dict[str, Any]) -> str:
    """Short description of the conditions that hold (absent conditions are not mentioned)."""
    if not isinstance(stats, dict):
        return str(stats)[:300]
    facts = []
    task = stats.get("task_type")
    if task:
        facts.append(f"{task.lower()} task")
    rows = stats.get("rows", 0)
    facts.append("small dataset under 1,000 rows" if rows < 1000 else
                 "large dataset over 100,000 rows" if rows > 100_000 else "medium dataset")
    n_cat, n_num = len(stats.get("categorical_columns") or []), len(stats.get("numerical_columns") or [])
    if n_cat > n_num:
        facts.append("mostly categorical features")
    elif n_cat == 0:
        facts.append("all numeric features")
    share = stats.get("minority_class_share")
    if share is not None and share < 0.10:
        facts.append(f"severe class imbalance, minority class {share:.2%}")
    if (stats.get("n_classes") or 0) > 2:
        facts.append("multi-class target")
    if stats.get("missing_values", 0) > 0:
        facts.append("missing values present")
    prof = stats.get("profile") or {}
    if prof.get("mean_abs_skew", 0) > 1:
        facts.append("skewed features with outliers")
    lin = prof.get("linearity")
    if lin is not None:
        facts.append("linear relationship with target" if lin > 0.5 else "weak or nonlinear relationship")
    if stats.get("is_time_series"):
        facts.append("time series ordered by date")
    return "; ".join(facts)


# ── output validation ───────────────────────────────────────────────────────
class LLMRecommendation(BaseModel):
    recommendations: List[str]
    reasoning: List[str] = []

    @field_validator("recommendations", mode="before")
    @classmethod
    def _coerce_names(cls, v):
        if isinstance(v, (str, dict)):
            v = [v]
        if not isinstance(v, list):
            raise ValueError("recommendations must be a list")
        out = []
        for item in v:
            if isinstance(item, dict):
                item = item.get("name") or item.get("model") or item.get("model_name")
            if isinstance(item, str) and item.strip():
                out.append(item.strip()[:80])
        if not out:
            raise ValueError("no model names in recommendations")
        return out[:MAX_RECOMMENDATIONS]

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasons(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if isinstance(v, dict):
            v = list(v.values())
        return [str(x).strip()[:400] for x in v] if isinstance(v, list) else []


def parse_llm_output(text: str):
    """(LLMRecommendation, None) or (None, error message)."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty response"
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, "no JSON object in the response"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON ({exc.msg})"
    if not isinstance(data, dict) or "recommendations" not in data:
        return None, "JSON has no 'recommendations' key"
    try:
        rec = LLMRecommendation(**{k: data.get(k) for k in ("recommendations", "reasoning")})
    except ValidationError as exc:
        return None, f"schema validation failed: {exc.errors()[0]['msg']}"
    n = len(rec.recommendations)
    rec.reasoning = (rec.reasoning + [""] * n)[:n]
    return rec, None


PROMPT_TEMPLATE = """
You are a data science assistant. Suggest machine learning models for the dataset below and explain why.

CONTEXT (retrieved guidelines):
{context}

DATASET INSIGHTS:
{question}

TASK:
Suggest 3-5 models that would work well for this data. Give one specific reason per model based on the
statistics (e.g. "handles severe class imbalance with class weights").

OUTPUT FORMAT:
Return ONLY a JSON object with two keys: "recommendations" (list of model names, exactly as written in
the allowed list) and "reasoning" (list of strings, one per model, same order).
Example: {{"recommendations": ["XGBoost", "Random Forest"], "reasoning": ["...", "..."]}}
"""


class ModelAdvisor:
    def __init__(self, model_name: str = None, temperature: float = 0.3):
        from langchain_core.prompts import PromptTemplate

        self.model_name = model_name or get_llm_model()
        self.temperature = temperature
        self.init_error: Optional[str] = None
        self.retriever = get_rag_chain()
        self.prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        self.llm = None
        self.chain = None
        if not os.environ.get("GROQ_API_KEY"):
            self.init_error = "GROQ_API_KEY is not set"
            return
        try:
            from langchain_classic.chains import RetrievalQA
            from langchain_groq import ChatGroq

            self.llm = ChatGroq(temperature=temperature, model_name=self.model_name, max_retries=1, timeout=90)
            self.chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever,
                                                     chain_type_kwargs={"prompt": self.prompt})
        except Exception as exc:
            self.init_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def build_query(stats_json, supported_models=None, similar_workspaces=None) -> str:
        query = f"RETRIEVAL FACTS: {retrieval_facts(stats_json)}\nDataset Statistics: {stats_json}"
        if similar_workspaces:
            history = ", ".join(f"{w['best_model']} ({w.get('metric') or 'score'} {w.get('best_score')}, "
                                f"similarity {w.get('similarity', 'n/a')})" for w in similar_workspaces)
            query += ("\n\nHISTORICAL INTELLIGENCE:\nOn similar past datasets these models were the best: "
                      f"{history}.\nConsider them, but judge this dataset on its own statistics.")
        if supported_models:
            query += f"\n\nCONSTRAINT: You must ONLY suggest models from this available list: {', '.join(supported_models)}."
        return query

    @staticmethod
    def fallback_models(stats_json, supported_models=None) -> List[str]:
        fallback = ["XGBoost", "Random Forest"]
        if isinstance(stats_json, dict) and stats_json.get("is_time_series") and supported_models \
                and "Prophet" in supported_models:
            fallback = ["Prophet", "XGBoost", "Random Forest"]
        elif isinstance(stats_json, dict) and stats_json.get("task_type") == "Regression":
            fallback.append("Linear Regression")
        if supported_models:
            fallback = [m for m in fallback if m in supported_models] or list(supported_models[:3])
        return fallback

    def retrieve_rules(self, query: str) -> List[Dict[str, str]]:
        try:
            docs = self.retriever.invoke(query)
        except Exception as exc:  # shown in the UI, not hidden
            return [{"rule": "retrieval failed", "text": f"{type(exc).__name__}: {exc}"}]
        return [{"rule": d.metadata.get("rule") or _rule_name(d.page_content), "text": d.page_content}
                for d in docs]

    def get_recommendations(self, stats_json, supported_models=None, similar_workspaces=None) -> Dict[str, Any]:
        query = self.build_query(stats_json, supported_models, similar_workspaces)
        base = {"model": self.model_name, "history_used": bool(similar_workspaces),
                "similar_workspaces": list(similar_workspaces or []), "retrieved_rules": self.retrieve_rules(query),
                "retrieval_facts": retrieval_facts(stats_json), "raw_response": None, "attempts": 0}
        fallback = self.fallback_models(stats_json, supported_models)
        if self.chain is None:
            return {**base, "recommendations": fallback, "reasoning": [FALLBACK_REASON_ERROR] * len(fallback),
                    "source": "fallback", "error": self.init_error or "LLM unavailable"}
        error, reason = None, FALLBACK_REASON_ERROR
        for attempt in range(2):
            base["attempts"] = attempt + 1
            q = query if attempt == 0 else (query + f"\n\nYour previous answer was invalid ({error}). Reply with "
                                            "ONLY the JSON object described above.")
            try:
                out = self.chain.invoke(q)
                raw = out.get("result") if isinstance(out, dict) else str(out)
            except Exception as exc:  # 404 model, auth, network: no retry, fall back with the reason
                error, reason = f"{type(exc).__name__}: {str(exc)[:300]}", FALLBACK_REASON_ERROR
                logger.warning("LLM call failed: %s", error)
                break
            base["raw_response"] = raw
            parsed, perr = parse_llm_output(raw)
            if parsed:
                return {**base, "recommendations": parsed.recommendations, "reasoning": parsed.reasoning,
                        "source": "llm", "error": None}
            error, reason = f"invalid LLM output: {perr}", FALLBACK_REASON_INVALID
        return {**base, "recommendations": fallback, "reasoning": [reason] * len(fallback),
                "source": "fallback", "error": error}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_vector_store()
    print("Vector store saved to", VECTOR_STORE_PATH)
