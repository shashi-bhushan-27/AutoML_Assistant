"""
Experiment report: a Groq-written summary when the LLM is available, otherwise a
template. ``generate`` always says which one produced the text.
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

from app_backend.llm_rag_core import get_report_model

load_dotenv()
logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or get_report_model()
        self.llm = None
        self.init_error = None
        self.last_source = None
        self.last_error = None
        if not os.environ.get("GROQ_API_KEY"):
            self.init_error = "GROQ_API_KEY is not set"
            return
        try:
            from langchain_groq import ChatGroq

            self.llm = ChatGroq(temperature=0.3, model_name=self.model_name, max_retries=1, timeout=90)
        except Exception as exc:
            self.init_error = f"{type(exc).__name__}: {exc}"

    def _prompt(self, d: Dict[str, Any]) -> str:
        targets = d.get("target_cols") or [d.get("target_col")]
        return f"""Write a concise experiment report in Markdown for this AutoML run. Use ONLY the facts below;
do not invent numbers, models or steps. If something is not given, do not mention it.

Dataset: {d.get('dataset_name')} (shape {d.get('dataset_shape')})
Task: {d.get('task_type')}; target(s): {', '.join(map(str, targets))}
Split: {d.get('split')}
Selection metric: {d.get('metric')} ({d.get('metric_reason', '')})
Leaderboard (test split): {d.get('leaderboard')}
Majority-class baseline: {d.get('baseline')}
Failed models: {d.get('failed_models') or 'none'}
Preprocessing steps (fitted on the training rows unless stated): {d.get('preprocessing_steps')}
Recommendation source: {d.get('recommendation_source')}

Sections: 1. Summary  2. Data  3. Preprocessing  4. Model comparison  5. Caveats and next steps"""

    def generate(self, workspace_data: Dict[str, Any]) -> Dict[str, Any]:
        """{"text", "source" ("llm" | "template"), "model", "error"}."""
        if self.llm is not None:
            try:
                text = self.llm.invoke(self._prompt(workspace_data)).content
                self.last_source, self.last_error = "llm", None
                return {"text": text, "source": "llm", "model": self.model_name, "error": None}
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                logger.warning("LLM report failed: %s", self.last_error)
        else:
            self.last_error = self.init_error
        self.last_source = "template"
        return {"text": self._template(workspace_data), "source": "template", "model": None,
                "error": self.last_error}

    def generate_summary_report(self, workspace_data: Dict) -> str:
        return self.generate(workspace_data)["text"]

    def _template(self, d: Dict[str, Any]) -> str:
        targets = d.get("target_cols") or [d.get("target_col")]
        lines = [f"# Experiment report: {d.get('dataset_name', 'dataset')}",
                 f"_Generated {datetime.now():%Y-%m-%d %H:%M} from a template (no LLM)._", "",
                 "## Data", f"- Shape: {d.get('dataset_shape')}", f"- Task: {d.get('task_type')}",
                 f"- Target(s): {', '.join(map(str, targets))}", f"- Split: {d.get('split')}", "",
                 "## Model comparison (test split)",
                 f"- Selection metric: **{d.get('metric')}** ({d.get('metric_reason', '')})",
                 f"- Best model: **{d.get('best_model', 'n/a')}** ({d.get('metric')} = {d.get('best_score', 'n/a')})"]
        if d.get("baseline"):
            lines.append(f"- Majority-class baseline: {d['baseline']}")
        for row in d.get("leaderboard") or []:
            lines.append(f"  - {row}")
        if d.get("failed_models"):
            lines += ["", "## Failed models"] + [f"- {f}" for f in d["failed_models"]]
        lines += ["", "## Preprocessing", self._format_steps(d.get("preprocessing_steps", []))]
        lines += ["", "## Recommendations", f"- Source: {d.get('recommendation_source', 'n/a')}",
                  f"- Models: {', '.join(d.get('recommendations') or []) or 'n/a'}"]
        return "\n".join(lines)

    @staticmethod
    def _format_steps(steps: List[Dict]) -> str:
        if not steps:
            return "- No preprocessing steps recorded"
        return "\n".join(f"- **{s.get('step')}**: {s.get('action')} (fitted on {s.get('fitted_on', 'n/a')})"
                         for s in steps[:25])
