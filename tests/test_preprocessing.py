"""B-1 (train/serve parity), B-2 (no leakage), B-3 (zero-inflated columns), B-7 (label encoding),
B-8 (pickle without data), dedup reporting, schema handling."""
import pickle

import numpy as np
import pandas as pd
import pytest

from app_backend.preprocessing_engine.engine import AutoPreprocessor, SchemaError
from tests.conftest import full_dataset

BATCH_SIZES = [1, 16, 256, "full"]


def _fit(df, target, **kw):
    p = AutoPreprocessor(target_col=target, verbose=False, **kw)
    out = p.fit_transform(df=df)
    return p, out


def _serve(p, raw, batch):
    size = len(raw) if batch == "full" else batch
    return pd.concat([p.transform(raw.iloc[i:i + size]) for i in range(0, len(raw), size)])


@pytest.mark.parametrize("dataset,target", [("adult_df", "class"), ("california_df", "MedHouseVal"),
                                            ("credit_like_df", "Class"), ("mixed_df", "target")])
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_transform_reproduces_training_representation(request, dataset, target, batch):
    """B-1: transform(raw test rows) == fit_transform's representation of the same rows, any batch size."""
    df = request.getfixturevalue(dataset)
    p, out = _fit(df, target, random_state=3)
    raw_test = df.loc[out["X_test"].index].drop(columns=[target])
    served = _serve(p, raw_test, batch)
    assert list(served.columns) == list(out["X_test"].columns) == p.feature_names_
    np.testing.assert_allclose(served.to_numpy(), out["X_test"].to_numpy(), rtol=1e-9, atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("name,target", [("adult", "class"), ("credit", "Class"), ("california", "MedHouseVal")])
def test_parity_full_benchmarks(name, target):
    """B-1 acceptance on the full benchmark datasets (skipped if not downloaded)."""
    df = full_dataset(name)
    p, out = _fit(df, target, random_state=0)
    raw_test = df.loc[out["X_test"].index].drop(columns=[target])
    for batch in [1, 16, 256]:
        sub = raw_test.iloc[:512]
        np.testing.assert_allclose(_serve(p, sub, batch).to_numpy(), out["X_test"].loc[sub.index].to_numpy(),
                                   rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(p.transform(raw_test).to_numpy(), out["X_test"].to_numpy(), rtol=1e-9, atol=1e-9)


def _fitted_statistics(p):
    enc = {}
    for col, info in p.encoder.encoders.items():
        if info["type"] == "target":
            enc[col] = [np.asarray(e).tolist() for e in info["encoder"].encodings_]
        elif info["type"] == "frequency":
            enc[col] = sorted(info["mapping"].items())
        else:
            enc[col] = list(info["categories"])
    return {
        "caps": dict(p.transformer.outlier_caps), "skew": dict(p.transformer.transforms),
        "impute": {c: v["value"] for c, v in p.imputer.imputers.items()}, "encoders": enc,
        "selected": list(p.selector.selected_features),
        "scaler": (np.asarray(getattr(p.scaler.scaler, "center_", getattr(p.scaler.scaler, "mean_", []))).tolist()),
    }


def test_no_statistic_depends_on_test_features(mixed_df):
    """B-2: rewriting the features of the test rows leaves every fitted statistic unchanged
    (the stratified split depends only on the labels, so the partition is identical)."""
    p1, out = _fit(mixed_df, "target", random_state=5)
    tampered = mixed_df.copy()
    test_idx = out["X_test"].index
    tampered.loc[test_idx, "income"] = 1e9
    tampered.loc[test_idx, "age"] = -1000
    tampered.loc[test_idx, "city"] = "NEVER_SEEN"
    tampered.loc[test_idx, "job"] = "j0"
    p2, out2 = _fit(tampered, "target", random_state=5)
    assert list(out2["X_test"].index) == list(test_idx)
    assert _fitted_statistics(p1) == _fitted_statistics(p2)


def test_permuting_test_labels_leaves_target_encoding_unchanged(california_df):
    """B-2: target encoding is fitted on training labels only. Regression uses a random split that does
    not depend on y, so permuting the test labels keeps the partition and must keep every statistic."""
    df = california_df.copy()
    df["region"] = pd.cut(df["Longitude"], 40, labels=[f"r{i}" for i in range(40)]).astype(str)  # 11-100 levels
    p1, out = _fit(df, "MedHouseVal", random_state=1)
    assert p1.encoder.encoders["region"]["type"] == "target"
    permuted = df.copy()
    idx = out["X_test"].index
    permuted.loc[idx, "MedHouseVal"] = np.random.default_rng(0).permutation(df.loc[idx, "MedHouseVal"].values)
    p2, out2 = _fit(permuted, "MedHouseVal", random_state=1)
    assert list(out2["X_test"].index) == list(idx)
    assert _fitted_statistics(p1) == _fitted_statistics(p2)
    np.testing.assert_allclose(out["X_train"].to_numpy(), out2["X_train"].to_numpy())


def test_every_fitted_step_is_labelled_train(mixed_df):
    p, _ = _fit(mixed_df, "target")
    fitted = {e["step"]: e["fitted_on"] for e in p.full_log}
    for step in ("Outlier Handling", "Imputation", "Encoding", "Constant Filter", "Scaling"):
        assert fitted.get(step) == "train", step
    assert fitted["Deduplication"] == "all rows"


def test_zero_inflated_columns_are_not_capped_to_a_constant(mixed_df):
    """B-3: 'gain' is 90% zeros (IQR = 0) - it must be kept, not capped, and the reason logged."""
    p, out = _fit(mixed_df, "target")
    assert "gain" in p.feature_names_
    assert "gain" not in p.transformer.outlier_caps
    assert "IQR is 0" in p.transformer.capping_skipped["gain"]
    assert out["X_train"]["gain"].nunique() > 2


def test_adult_keeps_capital_columns(adult_df):
    p, out = _fit(adult_df, "class")
    for col in ("capital-gain", "capital-loss"):
        assert col in p.feature_names_ and col in p.transformer.capping_skipped
        assert out["X_train"][col].nunique() > 2


@pytest.mark.slow
def test_adult_xgboost_accuracy_with_capital_columns():
    """B-3 acceptance: with the columns kept, XGBoost on full Adult reaches >= 0.87 accuracy
    (mean over the ablation's 5 split seeds; measured 0.8715 +- 0.0033)."""
    from app_backend.model_trainer import ModelTrainer

    df = full_dataset("adult")
    accs = []
    for seed in (0, 1, 2, 3, 4):
        p, out = _fit(df, "class", random_state=seed)
        tr = ModelTrainer(df, "class", p.task_type, random_state=seed)
        tr.set_preprocessed_data(out["X_train"], out["X_test"], out["y_train"], out["y_test"])
        accs.append(float(tr.run_selected_models(["XGBoost"]).iloc[0]["Accuracy"]))
    assert np.mean(accs) >= 0.87, accs


def test_string_labels_are_encoded_and_decoded(adult_df):
    """B-7: '>50K'/'<=50K' labels are label-encoded; the encoder is stored and inverts."""
    p, out = _fit(adult_df, "class")
    assert p.classes_ == ["<=50K", ">50K"]
    assert set(np.unique(out["y_train"])) == {0, 1}
    assert list(p.decode_target([0, 1, 1])) == ["<=50K", ">50K", ">50K"]


def test_pickle_excludes_data_and_round_trips(credit_like_df, tmp_path):
    """B-8: the pickled pipeline holds no training data and transforms identically after loading."""
    p, out = _fit(credit_like_df, "Class")
    path = tmp_path / "p.pkl"
    p.save(str(path))
    assert path.stat().st_size < 200_000
    q = AutoPreprocessor.load(str(path))
    assert q.X_train is None and q.X_test is None and p.X_train is not None
    raw = credit_like_df.loc[out["X_test"].index].drop(columns=["Class"])
    np.testing.assert_allclose(q.transform(raw).to_numpy(), out["X_test"].to_numpy(), rtol=1e-9, atol=1e-9)
    assert pickle.loads(pickle.dumps(p)).feature_names_ == p.feature_names_


def test_deduplication_is_counted(mixed_df):
    df = pd.concat([mixed_df, mixed_df.head(25)], ignore_index=True)
    p, _ = _fit(df, "target")
    assert p.get_report()["summary"]["duplicates_removed"] == 25


def test_schema_handling(mixed_df):
    p, out = _fit(mixed_df, "target")
    raw = mixed_df.loc[out["X_test"].index].drop(columns=["target"]).head(20)
    base = p.transform(raw)
    # reordered columns -> identical output
    pd.testing.assert_frame_equal(p.transform(raw[raw.columns[::-1]]), base)
    # extra column ignored with a warning
    X, warn = p.transform(raw.assign(extra_col=1), return_warnings=True)
    pd.testing.assert_frame_equal(X, base)
    assert any("extra_col" in w for w in warn)
    # unseen category -> all-zero one-hot row + warning
    X, warn = p.transform(raw.assign(city="Atlantis"), return_warnings=True)
    assert (X[[c for c in X.columns if c.startswith("city_")]] == 0).all().all()
    assert any(w.startswith("city") for w in warn)
    # missing required column -> SchemaError naming it
    with pytest.raises(SchemaError) as err:
        p.transform(raw.drop(columns=["age"]))
    assert err.value.missing == ["age"]
    # non-numeric value in a numeric column -> SchemaError
    with pytest.raises(SchemaError) as err:
        p.transform(raw.assign(age="old"))
    assert "age" in err.value.invalid


def test_parity_check_report(mixed_df):
    p, out = _fit(mixed_df, "target")
    raw = mixed_df.loc[out["X_test"].index].drop(columns=["target"])
    report = p.parity_check(raw.head(40), out["X_test"])
    assert report["passed"].all() and len(report) == len(p.feature_names_)


def test_smote_on_train_only_with_outcome(credit_like_df):
    """B-14: SMOTE changes only the training rows and records its outcome."""
    p, out = _fit(credit_like_df, "Class", apply_smote=True)
    outcome = p.balancer.smote_outcome
    assert outcome["status"] == "applied", outcome
    assert outcome["rows_after"] > outcome["rows_before"] == int(0.8 * len(credit_like_df.drop_duplicates()))
    assert len(out["X_test"]) == p.n_test_
    p2, _ = _fit(credit_like_df.assign(Class=np.arange(len(credit_like_df)) % 2), "Class", apply_smote=True)
    assert p2.balancer.smote_outcome["status"] == "skipped"
    assert "balanced" in p2.balancer.smote_outcome["reason"]
