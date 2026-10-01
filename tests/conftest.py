"""Shared fixtures. No test touches the network: the Groq client and the FAISS retriever are mocked."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "samples")
ABLATION_CACHE = os.path.join(ROOT, "ablation", ".cache")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the full benchmark datasets (ablation/.cache) or takes > 30 s")


@pytest.fixture(scope="session")
def adult_df():
    """4,000-row stratified sample of Adult (OpenML 1590) with the original string labels."""
    return pd.read_csv(os.path.join(DATA, "adult_income_sample.csv"))


@pytest.fixture(scope="session")
def california_df():
    """3,000-row sample of California Housing (target MedHouseVal)."""
    return pd.read_csv(os.path.join(DATA, "california_housing_sample.csv"))


@pytest.fixture(scope="session")
def credit_like_df():
    """Synthetic stand-in for Credit Card Fraud: numeric features, ~1% positives, skewed amount."""
    rng = np.random.default_rng(1)
    n = 6000
    X = rng.normal(size=(n, 8))
    logit = 3.5 * X[:, 0] - 2.5 * X[:, 1] - 6.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 9)])
    df["Amount"] = rng.lognormal(3, 1.2, n)
    df["Class"] = y
    return df


@pytest.fixture(scope="session")
def mixed_df():
    """Small synthetic frame with every column kind the pipeline handles."""
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": rng.lognormal(10, 1, n),
        "gain": np.where(rng.random(n) < 0.9, 0, rng.integers(100, 10000, n)),
        "city": rng.choice(list("ABCDE"), n),
        "job": rng.choice([f"j{i}" for i in range(30)], n),
        "zip": rng.choice([f"z{i}" for i in range(150)], n),
        "flag": rng.choice(["yes", "no"], n),
        "when": pd.date_range("2021-01-01", periods=n, freq="h").astype(str),
    })
    df.loc[rng.choice(n, 80, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "city"] = np.nan
    df["target"] = np.where(df["age"] + (df["city"] == "A") * 10 + rng.normal(0, 8, n) > 55, "yes_buy", "no_buy")
    return df


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    """Redirect the workspace store to a temporary directory."""
    import app_backend.workspace_manager as wsm

    monkeypatch.setattr(wsm, "WORKSPACE_DIR", str(tmp_path / "workspaces"))
    monkeypatch.setattr(wsm, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(wsm, "UPLOADS_DIR", str(tmp_path / "data" / "uploads"))
    monkeypatch.setattr(wsm, "WORKSPACE_INDEX", str(tmp_path / "workspaces" / "index.json"))
    return wsm


def full_dataset(name):
    """Full benchmark dataset from the ablation cache, or skip."""
    path = os.path.join(ABLATION_CACHE, f"{name}.pkl")
    if not os.path.exists(path):
        pytest.skip(f"{path} not available (run: python ablation/run_ablation.py prepare)")
    return pd.read_pickle(path)
