#!/usr/bin/env python
"""
Build the merged paper: original draft + ablation subsection + new Future-Work
paragraph + comment-only flags (``% [ABLATION CHECK] ...``) next to every claim
that the ablation results or the code contradict. Flags are LaTeX comments, so
the compiled PDF is unchanged apart from the inserted material.

    python ablation/merge_paper.py --original draft.tex [--tail tail.tex] --out merged.tex

The draft itself is not part of this repository.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FLAG = "% [ABLATION CHECK] "


def _load(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def measured():
    s = json.load(open(os.path.join(HERE, "results", "summary.json")))
    c = json.load(open(os.path.join(HERE, "results", "code_checks.json")))
    return s, c


def build_flags(s, c):
    """(anchor substring, comment) pairs; the comment is inserted above the first line containing the anchor."""
    h = s["headline"]["measured"]
    fx = s.get("flag_values", {})

    def g(key, default="n/a"):
        return fx.get(key, default)

    acc = f"measured best Adult accuracy {g('adult_best_acc')} ({g('adult_best_acc_model')}); Credit accuracy is {g('credit_best_acc')} but the all-negative baseline is {g('credit_majority_acc')}"
    r2 = f"measured best California R^2 {g('california_best_r2')} ({g('california_best_r2_model')})"
    red = f"measured training-time reduction of the Full RAG selection vs. brute force: {g('reduction_by_dataset')}"
    lat = f"measured /predict (TestClient, single row): {g('latency_by_dataset')}"
    no_ens = "ModelTrainer has no ensemble/stacking model, so an 'AI-selected ensemble' cannot be produced by the code"
    return [
        ("our AI-selected ensemble achieves 90.4", f"{acc}. {r2}. {red}. {lat}. {no_ens}."),
        ("validated by a fuzzy matcher with deterministic fallback",
         "the matcher is exact / case-space-normalised / substring matching in main_ui.py (no Levenshtein, no tau); "
         "fallbacks are [XGBoost, Random Forest] (+Linear Regression) in the advisor and [XGBoost, RF, GB|LogReg] in the UI."),
        ("A stateful AutoPreprocessor fits all transformers strictly on training data",
         "only the scaler is fit on the training split; outlier capping, skew transforms, imputer, encoder (incl. target "
         f"encoding with the full y) and selector are fit on all rows. Parity is not exact: {g('parity_summary')}."),
        ("enabling crash-free feature attribution", f"SHAP with the shipped routing: {g('shap_full_summary')}."),
        ("achieving an 86.4\\% reduction in search compute time", red + ". " + g("meta_summary") + "."),
        ("by fitting all statistical transformers strictly on training partitions",
         "see the flag in the abstract: only the scaler is fit on the training split; transform() does not replay the fitted steps."),
        ("demonstrating 90.4\\% classification accuracy", f"{acc}. {r2}. {lat}."),
        ("RAG-grounded recommendation with natural language justifications; 86.4\\% compute reduction", red + "."),
        ("topo-deterministic fallback (Prophet/ XGBoost/ RandomForest)",
         "code: advisor fallback [XGBoost, Random Forest] (+Linear Regression for regression; Prophet only for time series)."),
        ("\\mathrm{sim}(P_{new}, P_{prior}) \\geq 0.95",
         "code (WorkspaceManager.find_similar_workspaces): same task AND (rows within 50% OR columns within 30%); no profile "
         f"vector or Euclidean similarity. {g('meta_summary')}."),
        ("The prompt is processed by a large language model (Llama 3.1 via Groq)",
         "code uses llama-3.3-70b-versatile, which Groq no longer serves (404 model_not_found) -> every call falls back. "
         "The JSON returned by the LLM is not validated (no hyperparameters are parsed)."),
        ("A runtime fuzzy matcher computes the Levenshtein similarity",
         "no Levenshtein distance or tau=0.85 exists in the code; see ablation_section.tex for the matcher actually used."),
        ("\\text{XGBoost}, & \\text{if } t = \\text{tabular} \\wedge n > 10{,}000",
         "code fallback does not depend on n: [XGBoost, Random Forest] (+Linear Regression for regression)."),
        ("Critically, all imputation statistics are computed exclusively on the training partition",
         "SmartImputer.fit_transform runs on all rows before the split (engine.py)."),
        ("\\text{One-Hot}, & \\text{if } |c_j| \\leq 15",
         "code (SmartEncoder): <=2 levels label, 3-10 one-hot, 11-100 target encoding (full y), >100 frequency."),
        ("---are fitted strictly and exclusively on $X_{train}$",
         "only SmartScaler is fit on X_train; the other fitted steps see the test rows."),
        ("If the class imbalance ratio exceeds $1{:}3$",
         "SMOTE is an opt-in UI checkbox (default off); ImbalanceHandler flags imbalance when the minority share < 10%. "
         "All ablation runs use the default (off)."),
        ("guaranteeing that $\\mathcal{T}_{serve} \\equiv \\mathcal{T}_{train}$ and eliminating training-serving skew",
         f"contradicted: {g('parity_summary')}."),
        ("returning prediction labels and probabilities with sub-15~ms latency", lat + "."),
        ("yielding 13,062 test rows for the classification datasets and 5,162 test observations",
         f"an 80/20 split gives {g('test_rows')} test rows (after the app's de-duplication); Credit Card Fraud on OpenML (1597) "
         "has 29 features (no Time column)."),
        ("We compare the AI-Selected RAG Ensemble against ten individual algorithms",
         "Ridge Classifier is not in ModelTrainer.get_supported_models(); no ensemble/stacking exists in the code."),
        ("LLM inference uses Llama 3.1 via the Groq API",
         "code: llama-3.3-70b-versatile (no longer served). Ablation used openai/gpt-oss-120b as a documented substitute."),
        ("The AI-Selected RAG Ensemble achieves the highest performance across all metrics: 90.4",
         f"{acc}. {g('adult_capital_note')}. Full per-model results: ablation/results/brute_force_summary.csv."),
        ("The AI-Selected RAG Stacking ensemble achieves the lowest RMSE (38,910)",
         f"{r2}; RMSE in $100k: {g('california_best_rmse')}."),
        ("The RAG-based algorithm selection reduces total search compute time by 86.4",
         red + ". " + g("meta_summary") + "."),
        ("The FastAPI serving engine achieves 12.4~ms average inference latency",
         f"{lat}. FAISS retrieval: {g('retrieval_ms')}. SHAP TreeExplainer runs on 200 rows in the app (not 2k)."),
        ("The AI-Selected RAG Ensemble achieves 90.4\\% classification accuracy and 0.884",
         f"{acc}. {r2}. SMOTE is off by default, so it cannot explain the MCC/Kappa values."),
        ("The stateful AutoPreprocessor guarantees exact mathematical parity",
         f"contradicted: {g('parity_summary')}."),
        ("The automated SHAP routing engine successfully computes feature attributions across",
         f"{g('shap_full_summary')}."),
        ("The FastAPI serving engine delivers 12.4~ms inference latency", lat + "."),
        ("Eliminate LLM & Validated algorithm rate", g("valid_summary") + "."),
        ("Reduce search & RAG vs. brute-force", red + "."),
        ("Prevent training & Online-offline parity", f"contradicted: {g('parity_summary')}."),
        ("Crash-free  & SHAP routing success", g("shap_full_summary") + "."),
        ("Ultra-low  & API latency", lat + "."),
        ("Superior predictive  & Accuracy", f"{acc}. {r2}."),
        ("This eliminates hallucinations entirely while reducing search compute time by 86.4",
         g("valid_summary") + ". " + red + "."),
        ("guaranteeing $\\mathcal{T}_{serve} \\equiv \\mathcal{T}_{train}$ and eliminating training-serving skew with absolute",
         f"contradicted: {g('parity_summary')}."),
        ("The AI-Selected RAG Ensemble achieved 90.4\\% classification accuracy, 88.9", f"{acc}. {r2}. {lat}."),
        ("Fourth, the reported results represent single-run experiments",
         "the ablation subsection now reports mean+-std over 5 seeds; the main result tables are still single-run."),
        ("Section~\\ref{sec:experiments} presents experimental results",
         "pre-existing: label sec:experiments is undefined (the section is labelled sec:results)."),
        ("As illustrated in Figure~\\ref{fig:architecture}",
         "pre-existing: fig:architecture is not defined in the draft."),
    ]


def merge(original, tail, section, future_par, flags, out):
    text = _load(original)
    if tail and "%@@TEMPLATE_TAIL@@" in text:
        text = text.replace("%@@TEMPLATE_TAIL@@\n", _load(tail))

    # 1) packages: graphicx for the figures; algorithm/algpseudocode are used by the draft but not loaded
    pkg_anchor = "\\usepackage{hyperref}"
    add = ("\\usepackage{amssymb}    % added: the draft uses \\mathbb without loading amssymb\n"
           "\\usepackage{graphicx}   % added for the ablation figures\n"
           "\\usepackage{array}      % added for the ablation table (ragged-right notes column)\n"
           "\\usepackage{algorithm}  % added: the draft uses algorithm/algorithmic environments\n"
           "\\usepackage{algpseudocode}\n")
    assert pkg_anchor in text
    text = text.replace(pkg_anchor, add + pkg_anchor, 1)

    # 2) ablation subsection after the "Online Inference Latency" results (before Discussion)
    anchor = "\\subsection{Discussion}\n\\label{sec:discussion}"
    assert anchor in text, "Discussion anchor not found"
    text = text.replace(anchor, _load(section).rstrip() + "\n\n" + anchor, 1)

    # 3) replace the Future Work "Ablation Studies" paragraph
    start = text.index("\\textbf{Ablation Studies and Broader Evaluation.}")
    end = text.index("\n\n", start)
    text = text[:start] + _load(future_par).strip() + text[end:]

    # 4) comment-only flags
    lines = text.split("\n")
    placed, missing = 0, []
    for anchor, note in flags:
        for i, ln in enumerate(lines):
            if anchor in ln and not ln.lstrip().startswith("%"):
                lines.insert(i, FLAG + note)
                placed += 1
                break
        else:
            missing.append(anchor)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return placed, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--tail", default=None)
    ap.add_argument("--section", default=os.path.join(HERE, "ablation_section.tex"))
    ap.add_argument("--future", default=os.path.join(HERE, "future_work_paragraph.tex"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s, c = measured()
    placed, missing = merge(a.original, a.tail, a.section, a.future, build_flags(s, c), a.out)
    print(f"wrote {a.out}: {placed} flags placed; anchors not found: {missing}")


if __name__ == "__main__":
    main()
