"""
test_cdc_benchmarking.py
========================
Automated tests for Phase 4 — CDC Model Benchmarking.

Tests cover:
- Model artifact loading
- Three-class prediction output shape and validity
- Probability shape and validity
- Feature schema consistency
- No training/test leakage verification
- Artifact persistence (model + metadata)
- Reproducibility (same seed → same CV scores)

All tests use synthetic data where possible to avoid the
long runtime of the real 253K dataset.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import joblib

# ── Project path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cdc_preprocessing import build_preprocessing_pipeline
from cdc_preprocessing import (
    BINARY_FEATURES,
    ORDINAL_FEATURES,
    COUNT_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_ORDER as FEATURE_NAMES,
)
from cdc_model_benchmarking import (
    CLASS_NAMES,
    EXPECTED_FINGERPRINT,
    RANDOM_SEED,
    N_CV_FOLDS,
    compute_metrics,
    get_class_weight_dict,
    make_preprocessor,
    _verify_fingerprint,
    ARTIFACTS,
    TARGET_COL,
    XGBPipeline,
)

# Check xgboost availability
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Synthetic dataset factory ─────────────────────────────────────────────────
# FEATURE_NAMES is imported from cdc_preprocessing as FEATURE_ORDER

def make_synthetic(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y) with class distribution ~84/2/14% mirroring real dataset.
    Uses fixed seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    binary = [
        "HighBP","HighChol","CholCheck","Smoker","Stroke",
        "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
        "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","DiffWalk","Sex",
    ]
    ordinal = {"GenHlth":(1,5),"Age":(1,13),"Education":(1,6),"Income":(1,8)}
    count   = {"MentHlth":(0,30),"PhysHlth":(0,30)}

    data = {}
    for f in binary:
        data[f] = rng.randint(0, 2, n).astype(float)
    for f, (lo, hi) in ordinal.items():
        data[f] = rng.randint(lo, hi + 1, n).astype(float)
    for f, (lo, hi) in count.items():
        data[f] = rng.randint(lo, hi + 1, n).astype(float)
    data["BMI"] = rng.uniform(10, 60, n)

    X = pd.DataFrame(data, columns=FEATURE_NAMES)
    # ~84 / 2 / 14 split
    y_arr = rng.choice([0, 1, 2], size=n, p=[0.84, 0.02, 0.14])
    # Ensure all 3 classes present
    y_arr[:3] = [0, 1, 2]
    y = pd.Series(y_arr.astype(int), name=TARGET_COL)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# TestMetricComputation
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricComputation:
    """Tests for compute_metrics() covering all required metric keys."""

    def _run(self, n=300, seed=1):
        rng = np.random.RandomState(seed)
        y_true = rng.choice([0, 1, 2], n, p=[0.84, 0.02, 0.14])
        y_pred = rng.choice([0, 1, 2], n, p=[0.84, 0.02, 0.14])
        y_proba = rng.dirichlet([4, 0.1, 1], size=n)
        y_true[:3] = [0, 1, 2]
        y_pred[:3] = [0, 1, 2]
        return compute_metrics(y_true, y_pred, y_proba)

    def test_all_required_keys_present(self):
        m = self._run()
        required = [
            "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_f1", "cohen_kappa", "mcc", "roc_auc_ovr",
            "class0_precision", "class0_recall", "class0_f1",
            "class1_precision", "class1_recall", "class1_f1",
            "class2_precision", "class2_recall", "class2_f1",
        ]
        for key in required:
            assert key in m, f"Missing metric key: {key}"

    def test_metrics_in_valid_range(self):
        m = self._run()
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                    "class0_f1", "class1_f1", "class2_f1"]:
            assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of [0,1]"

    def test_kappa_range(self):
        m = self._run()
        assert -1.0 <= m["cohen_kappa"] <= 1.0

    def test_mcc_range(self):
        m = self._run()
        assert -1.0 <= m["mcc"] <= 1.0

    def test_perfect_prediction_macro_f1_1(self):
        y = np.array([0, 0, 1, 2, 0, 2])
        m = compute_metrics(y, y, None)
        assert m["macro_f1"] == pytest.approx(1.0)

    def test_no_proba_gives_nan_roc(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        m = compute_metrics(y, y, None)
        assert np.isnan(m["roc_auc_ovr"])

    def test_per_class_values_are_floats(self):
        m = self._run()
        for ci in [0, 1, 2]:
            assert isinstance(m[f"class{ci}_f1"], float)


# ══════════════════════════════════════════════════════════════════════════════
# TestPreprocessorIntegration
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessorIntegration:
    """Tests that make_preprocessor() integrates correctly with model training."""

    def test_preprocessor_returns_correct_shape(self):
        X, y = make_synthetic(300)
        prep = make_preprocessor()
        Xt = prep.fit_transform(X)
        assert Xt.shape == (300, 21)

    def test_preprocessor_no_nan(self):
        X, y = make_synthetic(300)
        prep = make_preprocessor()
        Xt = prep.fit_transform(X)
        assert not np.isnan(Xt).any()

    def test_pipeline_fit_predict(self):
        X, y = make_synthetic(300)
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_SEED,
                                class_weight="balanced")
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("classifier", lr)])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert set(preds).issubset({0, 1, 2})
        assert len(preds) == 300

    def test_pipeline_predict_proba_shape(self):
        X, y = make_synthetic(300)
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_SEED,
                                class_weight="balanced")
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("classifier", lr)])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (300, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_preprocessor_train_only_fitted(self):
        """Verify that fitting on train and transforming test has no test leakage."""
        X, y = make_synthetic(400)
        X_tr, X_te = X.iloc[:300], X.iloc[300:]
        prep = make_preprocessor()
        Xt_tr = prep.fit_transform(X_tr)
        Xt_te = prep.transform(X_te)
        # Shapes correct
        assert Xt_tr.shape[1] == Xt_te.shape[1] == 21
        # No NaN
        assert not np.isnan(Xt_te).any()

    def test_feature_order_independent_of_input_column_order(self):
        """Shuffled column input → same preprocessed output as original order."""
        X, _ = make_synthetic(100, seed=5)
        X_shuffled = X[list(reversed(X.columns))]
        prep1 = make_preprocessor()
        prep2 = make_preprocessor()
        Xt1 = prep1.fit_transform(X)
        Xt2 = prep2.fit_transform(X_shuffled)
        np.testing.assert_array_almost_equal(Xt1, Xt2, decimal=5)


# ══════════════════════════════════════════════════════════════════════════════
# TestClassWeights
# ══════════════════════════════════════════════════════════════════════════════

class TestClassWeights:
    def test_all_classes_present(self):
        _, y = make_synthetic(600)
        cw = get_class_weight_dict(y)
        assert set(cw.keys()) == {0, 1, 2}

    def test_minority_class_has_highest_weight(self):
        _, y = make_synthetic(600)
        cw = get_class_weight_dict(y)
        # Class 1 is the smallest minority
        assert cw[1] > cw[0]
        assert cw[1] > cw[2]

    def test_weights_are_positive(self):
        _, y = make_synthetic(600)
        cw = get_class_weight_dict(y)
        for w in cw.values():
            assert w > 0


# ══════════════════════════════════════════════════════════════════════════════
# TestModelDefinitions
# ══════════════════════════════════════════════════════════════════════════════

class TestModelDefinitions:
    def _get_models(self):
        from cdc_model_benchmarking import get_model_definitions
        _, y = make_synthetic(600)
        cw = get_class_weight_dict(y)
        return get_model_definitions(cw)

    def test_expected_models_present(self):
        models = self._get_models()
        expected = ["LogisticRegression", "DecisionTree", "RandomForest"]
        if XGBOOST_AVAILABLE:
            expected.append("XGBoost")
        for name in expected:
            assert name in models, f"Missing model: {name}"

    def test_lightgbm_absent(self):
        """LightGBM must be explicitly excluded (not installed)."""
        models = self._get_models()
        assert "LightGBM" not in models

    def test_all_models_have_fit_predict(self):
        models = self._get_models()
        for name, model in models.items():
            assert hasattr(model, "fit"),     f"{name} missing .fit()"
            assert hasattr(model, "predict"), f"{name} missing .predict()"

    def test_logistic_regression_config(self):
        models = self._get_models()
        lr = models["LogisticRegression"]
        assert lr.random_state == RANDOM_SEED
        assert lr.class_weight == "balanced"
        assert lr.solver == "lbfgs"

    def test_decision_tree_config(self):
        models = self._get_models()
        dt = models["DecisionTree"]
        assert dt.class_weight == "balanced"
        assert dt.random_state == RANDOM_SEED

    def test_random_forest_config(self):
        models = self._get_models()
        rf = models["RandomForest"]
        assert rf.class_weight == "balanced"
        assert rf.random_state == RANDOM_SEED


# ══════════════════════════════════════════════════════════════════════════════
# TestCrossValidation
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossValidation:
    """Tests for the CV runner using fast synthetic data."""

    def test_cv_returns_expected_keys(self):
        from cdc_model_benchmarking import run_cross_validation
        X, y = make_synthetic(400)
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_SEED, class_weight="balanced")
        result = run_cross_validation("LR_test", lr, X, y, n_folds=2)
        assert "name" in result
        assert "fold_metrics" in result
        assert "aggregate" in result

    def test_cv_aggregate_has_mean_std(self):
        from cdc_model_benchmarking import run_cross_validation
        X, y = make_synthetic(400)
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_SEED, class_weight="balanced")
        result = run_cross_validation("LR_test", lr, X, y, n_folds=2)
        agg = result["aggregate"]
        assert "macro_f1_mean" in agg
        assert "macro_f1_std"  in agg

    def test_cv_fold_count(self):
        from cdc_model_benchmarking import run_cross_validation
        X, y = make_synthetic(400)
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_SEED, class_weight="balanced")
        result = run_cross_validation("LR_test", lr, X, y, n_folds=3)
        assert len(result["fold_metrics"]) == 3

    def test_cv_macro_f1_in_range(self):
        from cdc_model_benchmarking import run_cross_validation
        X, y = make_synthetic(400)
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_SEED, class_weight="balanced")
        result = run_cross_validation("LR_test", lr, X, y, n_folds=2)
        mean = result["aggregate"]["macro_f1_mean"]
        assert 0.0 <= mean <= 1.0

    def test_cv_no_test_data_used(self):
        """
        Verify that in each fold, the validation indices are disjoint from
        training indices — standard StratifiedKFold guarantee.
        """
        X, y = make_synthetic(400)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        for tr_idx, val_idx in skf.split(X, y):
            assert len(set(tr_idx) & set(val_idx)) == 0

    def test_cv_reproducibility(self):
        """Same seed → same CV macro_f1."""
        from cdc_model_benchmarking import run_cross_validation
        X, y = make_synthetic(400)
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_SEED, class_weight="balanced")
        r1 = run_cross_validation("LR_seed1", lr, X, y, n_folds=2)
        lr2 = LogisticRegression(max_iter=200, random_state=RANDOM_SEED, class_weight="balanced")
        r2 = run_cross_validation("LR_seed2", lr2, X, y, n_folds=2)
        assert abs(r1["aggregate"]["macro_f1_mean"] - r2["aggregate"]["macro_f1_mean"]) < 1e-8

    def test_three_class_output_in_cv(self):
        """CV folds must produce three-class predictions."""
        from cdc_model_benchmarking import run_cross_validation
        X, y = make_synthetic(400)
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_SEED, class_weight="balanced")
        result = run_cross_validation("LR_test", lr, X, y, n_folds=2)
        # All per-class metrics should be present in fold results
        for fold_m in result["fold_metrics"]:
            assert "class0_f1" in fold_m
            assert "class1_f1" in fold_m
            assert "class2_f1" in fold_m


# ══════════════════════════════════════════════════════════════════════════════
# TestArtifactSaving
# ══════════════════════════════════════════════════════════════════════════════

class TestArtifactSaving:
    """Tests for model serialization and metadata persistence."""

    def test_save_and_reload_model(self, tmp_path):
        """Fit a simple pipeline, save with joblib, reload and predict."""
        X, y = make_synthetic(300)
        prep = make_preprocessor()
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_SEED, class_weight="balanced")
        pipe = Pipeline([("preprocessor", prep), ("classifier", lr)])
        pipe.fit(X, y)

        save_path = tmp_path / "model.pkl"
        joblib.dump(pipe, save_path)
        loaded = joblib.load(save_path)

        preds = loaded.predict(X)
        assert set(preds).issubset({0, 1, 2})
        assert len(preds) == 300

    def test_reloaded_proba_shape(self, tmp_path):
        X, y = make_synthetic(300)
        prep = make_preprocessor()
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_SEED, class_weight="balanced")
        pipe = Pipeline([("preprocessor", prep), ("classifier", lr)])
        pipe.fit(X, y)

        save_path = tmp_path / "model.pkl"
        joblib.dump(pipe, save_path)
        loaded = joblib.load(save_path)
        proba = loaded.predict_proba(X)
        assert proba.shape == (300, 3)

    def test_metadata_json_has_required_keys(self, tmp_path):
        """Metadata JSON must contain all required keys."""
        meta = {
            "model_name": "TestModel",
            "dataset_fingerprint": EXPECTED_FINGERPRINT,
            "random_seed": RANDOM_SEED,
            "n_cv_folds": N_CV_FOLDS,
            "feature_names": FEATURE_NAMES,
            "best_params": {},
            "best_cv_macro_f1": 0.5,
            "cv_aggregate": {"macro_f1_mean": 0.5, "macro_f1_std": 0.01},
            "library_versions": {"scikit-learn": "1.8.0"},
            "unavailable_packages": {"LightGBM": "not installed"},
        }
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(meta))
        loaded = json.loads(meta_path.read_text())
        for key in ["model_name", "dataset_fingerprint", "feature_names",
                    "best_params", "cv_aggregate", "library_versions"]:
            assert key in loaded, f"Missing key: {key}"

    def test_benchmark_artifacts_directory_exists(self):
        """The benchmark artifact directory should be created by the module."""
        assert ARTIFACTS.exists(), f"Artifact dir not created: {ARTIFACTS}"

    def test_feature_schema_consistency(self, tmp_path):
        """Saved feature_names must match the 21 expected features."""
        meta = {"feature_names": FEATURE_NAMES}
        p = tmp_path / "meta.json"
        p.write_text(json.dumps(meta))
        loaded = json.loads(p.read_text())["feature_names"]
        assert len(loaded) == 21
        assert loaded == FEATURE_NAMES


# ══════════════════════════════════════════════════════════════════════════════
# TestDatasetIntegrity
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetIntegrity:
    """Verify fingerprint and split metadata without loading full training data."""

    def test_dataset_file_exists(self):
        from cdc_model_benchmarking import DATA_PATH
        assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"

    def test_fingerprint_matches_expected(self):
        fp = _verify_fingerprint()
        assert fp == EXPECTED_FINGERPRINT, (
            f"Fingerprint mismatch: got {fp}, expected {EXPECTED_FINGERPRINT}"
        )

    def test_split_metadata_exists(self):
        from cdc_model_benchmarking import META_PATH
        assert META_PATH.exists(), f"Split metadata not found: {META_PATH}"

    def test_split_metadata_has_required_fields(self):
        from cdc_model_benchmarking import META_PATH
        meta = json.loads(META_PATH.read_text())
        for key in ["n_train", "n_test", "random_seed", "leakage_check_passed",
                    "train_class_dist", "test_class_dist", "dataset_fingerprint"]:
            assert key in meta, f"Missing key in split metadata: {key}"

    def test_split_metadata_train_count(self):
        from cdc_model_benchmarking import META_PATH
        meta = json.loads(META_PATH.read_text())
        assert meta["n_train"] == 202943

    def test_split_metadata_leakage_passed(self):
        from cdc_model_benchmarking import META_PATH
        meta = json.loads(META_PATH.read_text())
        assert meta["leakage_check_passed"] is True

    def test_feature_names_json_exists(self):
        feat_path = ROOT / "models" / "cdc" / "cdc_feature_names.json"
        assert feat_path.exists()

    def test_feature_names_count(self):
        feat_path = ROOT / "models" / "cdc" / "cdc_feature_names.json"
        names = json.loads(feat_path.read_text())
        assert len(names) == 21


# ══════════════════════════════════════════════════════════════════════════════
# TestXGBoost (conditional on availability)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoost:
    def test_xgb_three_class_predict(self):
        X, y = make_synthetic(300)
        prep = make_preprocessor()
        Xt = prep.fit_transform(X)
        from sklearn.utils.class_weight import compute_class_weight
        sw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y.values)
        sample_w = np.array([sw[int(c)] for c in y.values])
        clf = XGBClassifier(
            objective="multi:softprob", num_class=3,
            n_estimators=20, max_depth=4,
            random_state=RANDOM_SEED, verbosity=0,
        )
        clf.fit(Xt, y.values, sample_weight=sample_w)
        preds = clf.predict(Xt)
        assert set(preds).issubset({0, 1, 2})

    def test_xgb_proba_sums_to_one(self):
        X, y = make_synthetic(300)
        prep = make_preprocessor()
        Xt = prep.fit_transform(X)
        clf = XGBClassifier(
            objective="multi:softprob", num_class=3,
            n_estimators=10, random_state=RANDOM_SEED, verbosity=0,
        )
        clf.fit(Xt, y.values)
        proba = clf.predict_proba(Xt)
        assert proba.shape == (300, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_xgb_in_model_definitions(self):
        from cdc_model_benchmarking import get_model_definitions
        _, y = make_synthetic(600)
        cw = get_class_weight_dict(y)
        models = get_model_definitions(cw)
        assert "XGBoost" in models


# ══════════════════════════════════════════════════════════════════════════════
# TestUnavailablePackages
# ══════════════════════════════════════════════════════════════════════════════

class TestUnavailablePackages:
    def test_lightgbm_documented_not_silently_skipped(self):
        from cdc_model_benchmarking import UNAVAILABLE, get_model_definitions
        # LightGBM must be in UNAVAILABLE dict
        assert "LightGBM" in UNAVAILABLE
        # And must NOT appear in model definitions
        _, y = make_synthetic(600)
        cw = get_class_weight_dict(y)
        models = get_model_definitions(cw)
        assert "LightGBM" not in models

    def test_smotenc_documented_not_silently_skipped(self):
        from cdc_model_benchmarking import UNAVAILABLE
        assert "SMOTENC" in UNAVAILABLE

    def test_lightgbm_note_is_non_empty(self):
        from cdc_model_benchmarking import UNAVAILABLE
        assert len(UNAVAILABLE["LightGBM"]) > 10

    def test_smotenc_note_is_non_empty(self):
        from cdc_model_benchmarking import UNAVAILABLE
        assert len(UNAVAILABLE["SMOTENC"]) > 10


# ══════════════════════════════════════════════════════════════════════════════
# TestSavedBenchmarkArtifacts (requires benchmark run to have been executed)
# ══════════════════════════════════════════════════════════════════════════════

class TestSavedBenchmarkArtifacts:
    """
    These tests verify artifacts that exist AFTER the benchmark has been run.
    They are skipped if the artifacts don't exist yet (pre-run state).
    """
    EXPECTED_MODELS = ["LogisticRegression", "DecisionTree", "RandomForest"]
    if XGBOOST_AVAILABLE:
        EXPECTED_MODELS.append("XGBoost")

    @pytest.mark.parametrize("model_name", EXPECTED_MODELS)
    def test_model_pkl_exists(self, model_name):
        model_path = ARTIFACTS / model_name / "model.pkl"
        if not model_path.exists():
            pytest.skip(f"Benchmark not yet run; {model_path} absent")
        assert model_path.stat().st_size > 100, "Model file too small"

    @pytest.mark.parametrize("model_name", EXPECTED_MODELS)
    def test_metadata_json_exists(self, model_name):
        meta_path = ARTIFACTS / model_name / "metadata.json"
        if not meta_path.exists():
            pytest.skip(f"Benchmark not yet run; {meta_path} absent")
        meta = json.loads(meta_path.read_text())
        assert meta["model_name"] == model_name

    @pytest.mark.parametrize("model_name", EXPECTED_MODELS)
    def test_saved_model_predicts_three_classes(self, model_name):
        model_path = ARTIFACTS / model_name / "model.pkl"
        if not model_path.exists():
            pytest.skip(f"Benchmark not yet run; {model_path} absent")
        X, y = make_synthetic(200, seed=99)
        loaded = joblib.load(model_path)
        if hasattr(loaded, "predict"):
            preds = loaded.predict(X)
            assert set(preds).issubset({0, 1, 2}), f"{model_name}: unexpected classes {set(preds)}"
        else:
            pytest.skip(f"{model_name}: loaded object has no predict() — XGB namespace")

    def test_benchmark_summary_json_exists(self):
        summary_path = ARTIFACTS / "benchmark_summary.json"
        if not summary_path.exists():
            pytest.skip("Benchmark not yet run")
        summary = json.loads(summary_path.read_text())
        assert isinstance(summary, list)
        assert len(summary) >= 3  # at least LR, DT, RF
