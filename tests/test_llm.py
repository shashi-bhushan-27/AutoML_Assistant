"""B-4 (configurable model, health check, visible fallback) and B-9 (validated LLM output). No network."""
import pytest
from langchain_core.documents import Document

import app_backend.llm_rag_core as rag


class FakeRetriever:
    def invoke(self, query):
        return [Document(page_content="[RULE: SEVERE_CLASS_IMBALANCE]\nuse F1", metadata={"rule": "SEVERE_CLASS_IMBALANCE"})]


class FakeChain:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return {"result": out}


class NotFound(Exception):
    status_code = 404


STATS = {"task_type": "Classification", "rows": 1000, "minority_class_share": 0.02,
         "categorical_columns": [], "numerical_columns": ["a"], "missing_values": 0}
SUPPORTED = ["Logistic Regression", "Random Forest", "XGBoost"]


@pytest.fixture
def advisor(monkeypatch):
    monkeypatch.setattr(rag, "get_rag_chain", lambda: FakeRetriever())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    a = rag.ModelAdvisor()
    return a


def test_default_model_is_configurable(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert rag.get_llm_model() == "openai/gpt-oss-120b"
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    assert rag.get_llm_model() == "qwen/qwen3.8-27b"


def test_missing_api_key_falls_back_with_reason(advisor):
    out = advisor.get_recommendations(STATS, SUPPORTED)
    assert out["source"] == "fallback" and out["error"] == "GROQ_API_KEY is not set"
    assert out["recommendations"] == ["XGBoost", "Random Forest"]
    assert out["retrieved_rules"][0]["rule"] == "SEVERE_CLASS_IMBALANCE"


def test_model_404_is_reported_as_fallback(advisor):
    """B-4 acceptance: a 404 from Groq gives source == 'fallback' and the error text."""
    advisor.chain = FakeChain([NotFound("Error code: 404 - model `llama-3.3-70b-versatile` does not exist")])
    out = advisor.get_recommendations(STATS, SUPPORTED)
    assert out["source"] == "fallback"
    assert "404" in out["error"]
    assert out["reasoning"][0] == rag.FALLBACK_REASON_ERROR
    assert len(advisor.chain.calls) == 1   # API errors are not retried


def test_valid_output_is_parsed(advisor):
    advisor.chain = FakeChain(['thinking... {"recommendations": ["XGBoost", "Random Forest"], '
                               '"reasoning": ["handles imbalance", "robust"]}'])
    out = advisor.get_recommendations(STATS, SUPPORTED)
    assert out["source"] == "llm" and out["error"] is None
    assert out["recommendations"] == ["XGBoost", "Random Forest"]
    assert out["reasoning"] == ["handles imbalance", "robust"]


def test_missing_key_is_retried_once_then_falls_back(advisor):
    """B-9: a JSON object without 'recommendations' used to be returned as-is (empty selection)."""
    advisor.chain = FakeChain(['{"models": ["XGBoost"]}', '{"answer": 42}'])
    out = advisor.get_recommendations(STATS, SUPPORTED)
    assert len(advisor.chain.calls) == 2 and "previous answer was invalid" in advisor.chain.calls[1]
    assert out["source"] == "fallback" and "recommendations" in out["error"]
    assert out["reasoning"][0] == rag.FALLBACK_REASON_INVALID


def test_retry_can_recover(advisor):
    advisor.chain = FakeChain(["no json here", '{"recommendations": ["XGBoost"], "reasoning": ["ok"]}'])
    out = advisor.get_recommendations(STATS, SUPPORTED)
    assert out["source"] == "llm" and out["attempts"] == 2


@pytest.mark.parametrize("payload,expected", [
    ('{"recommendations": [{"name": "XGBoost"}, {"model": "SVM"}, 7, ""], "reasoning": "one"}', ["XGBoost", "SVM"]),
    ('{"recommendations": "XGBoost"}', ["XGBoost"]),
    ('{"recommendations": ["a","b","c","d","e","f","g","h"]}', ["a", "b", "c", "d", "e", "f"]),
])
def test_output_coercion(payload, expected):
    """B-9: dict entries (which crashed the trainer with 'unhashable') are coerced; lists are truncated."""
    rec, err = rag.parse_llm_output(payload)
    assert err is None and rec.recommendations == expected
    assert len(rec.reasoning) == len(expected) and all(isinstance(r, str) for r in rec.reasoning)


@pytest.mark.parametrize("payload", ["", "no json", "{bad json}", '{"recommendations": []}',
                                     '{"recommendations": [1, 2]}'])
def test_invalid_outputs_are_rejected(payload):
    rec, err = rag.parse_llm_output(payload)
    assert rec is None and err


def test_history_goes_into_the_prompt():
    q = rag.ModelAdvisor.build_query(STATS, SUPPORTED, [{"best_model": "XGBoost", "best_score": 0.9,
                                                        "metric": "F1 Score", "similarity": 0.97}])
    assert "HISTORICAL INTELLIGENCE" in q and "similarity 0.97" in q and q.startswith("RETRIEVAL FACTS:")
    assert "missing values" not in rag.retrieval_facts(STATS)  # absent conditions are not mentioned
    assert "severe class imbalance" in rag.retrieval_facts(STATS)


def test_health_check_reports_404(monkeypatch):
    import groq

    class Models:
        def retrieve(self, model):
            raise NotFound("not found")

    class Client:
        def __init__(self, **kw):
            self.models = Models()

    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(groq, "Groq", Client)
    rag._HEALTH_CACHE.clear()
    h = rag.check_llm_health("llama-3.3-70b-versatile", force=True)
    assert h["ok"] is False and "404" in h["error"]
    monkeypatch.delenv("GROQ_API_KEY")
    assert rag.check_llm_health("x", force=True)["error"] == "GROQ_API_KEY is not set"


def test_report_generator_says_template(monkeypatch):
    from app_backend.report_generator import ReportGenerator

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    out = ReportGenerator().generate({"dataset_name": "d.csv", "metric": "F1 Score", "best_model": "XGBoost"})
    assert out["source"] == "template" and "template" in out["text"]
