"""The exported training script runs and reproduces the app's features and metric (B-16: one pipeline)."""
import os
import subprocess
import sys

from app_backend.code_generator import generate_training_code
from app_backend.model_trainer import ModelTrainer
from app_backend.preprocessing_engine.engine import AutoPreprocessor
from tests.conftest import ROOT


def test_exported_script_reproduces_app_result(adult_df, tmp_path):
    df = adult_df.head(1500)
    csv = tmp_path / "adult.csv"
    df.to_csv(csv, index=False)
    p = AutoPreprocessor(target_col="class", verbose=False, random_state=4)
    out = p.fit_transform(df=df)
    tr = ModelTrainer(df, "class", p.task_type, random_state=4)
    tr.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
    acc = tr.run_selected_models(["Random Forest"]).iloc[0]["Accuracy"]

    code = generate_training_code(str(csv).replace("\\", "/"), "class", "Random Forest",
                                  {"n_estimators": 100}, p.task_type, preprocessor=p)
    script = tmp_path / "train.py"
    script.write_text(code, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=ROOT)
    run = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    printed = next(line for line in run.stdout.splitlines() if line.startswith("Accuracy:"))
    assert round(float(printed.split(":")[1]), 4) == acc   # the app reports metrics to 4 decimals
    assert (tmp_path / "preprocessor_pipeline.pkl").exists()
