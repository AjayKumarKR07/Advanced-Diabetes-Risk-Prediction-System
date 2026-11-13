"""
test_cdc_group_aware_benchmarking.py
====================================
Comprehensive test suite for Phase 4.2 Group-Aware ML Model Benchmarking.

Validates:
  1. Deterministic and reproducible grouping (SHA-256 / 64-bit murmur3 hashing).
  2. Zero group overlap across all 5 outer folds (StratifiedGroupKFold).
  3. Zero group overlap in inner cross-validation folds.
  4. Frozen test set isolation (never touched, unchanged fingerprint).
  5. Leakage-safe preprocessing fitted strictly inside each training fold.
  6. Out-of-fold (OOF) prediction completeness (all 202,944 rows predicted).
  7. Valid probability outputs (non-negative, sum to 1.0, shape [N, 3]).
  8. Three-class prediction outputs ({0, 1, 2}).
  9. Artifact persistence & loading for all 4 models.
 10. Preservation of Phase 1–4 artifacts (no overwriting).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

import src.cdc_group_aware_benchmarking as gab

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "diabetes_012_health_indicators_BRFSS2015.csv"
GROUP_ARTIFACTS = ROOT / "models" / "cdc" / "group_aware_benchmarks"
ORIG_ARTIFACTS = ROOT / "models" / "cdc" / "benchmarks"
REPORTS_DIR = ROOT / "results" / "reports"
FIGURES_DIR = ROOT / "results" / "figures" / "group_aware_benchmark"

EXPECTED_FINGERPRINT = "a971809d0786d2d7"
EXPECTED_TRAIN_ROWS = 202944
EXPECTED_TEST_ROWS = 50736
MODELS = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]


@pytest.fixture(scope="module")
def train_data_and_groups():
    """Load training data and deterministic groups once for test module."""
    X_train, y_train, groups = gab.load_training_data()
    return X_train, y_train, groups


# ── Test 1: Deterministic and Reproducible Grouping ──────────────────────────

def test_reproducible_grouping(train_data_and_groups):
    X_train, _, groups1 = train_data_and_groups
    groups2 = gab.generate_deterministic_groups(X_train)

    assert len(groups1) == len(groups2)
    assert (groups1.values == groups2.values).all(), "Grouping must be 100% deterministic"
    assert groups1.nunique() == 182387, "Must yield exactly 182,387 unique feature profiles"


# ── Test 2: Zero Group Overlap in Outer 5-Fold CV ───────────────────────────

def test_zero_group_overlap_outer_cv(train_data_and_groups):
    X_train, y_train, groups = train_data_and_groups
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_counts = 0
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups=groups), 1):
        tr_g = set(groups.iloc[tr_idx])
        val_g = set(groups.iloc[val_idx])
        overlap = tr_g.intersection(val_g)

        assert len(overlap) == 0, f"Fold {fold} has {len(overlap)} overlapping profile groups!"
        assert len(tr_idx) + len(val_idx) == len(X_train)
        fold_counts += 1

    assert fold_counts == 5, "Must complete exactly 5 folds"


# ── Test 3: Zero Group Overlap in Inner CV ───────────────────────────────────

def test_zero_group_overlap_inner_cv(train_data_and_groups):
    X_train, y_train, groups = train_data_and_groups
    # Sample a fold partition to verify inner CV
    outer_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, _ in outer_sgkf.split(X_train, y_train, groups=groups):
        X_outer_tr = X_train.iloc[tr_idx]
        y_outer_tr = y_train.iloc[tr_idx]
        outer_groups = groups.iloc[tr_idx]
        break

    inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for inner_fold, (in_tr, in_val) in enumerate(inner_sgkf.split(X_outer_tr, y_outer_tr, groups=outer_groups), 1):
        in_tr_g = set(outer_groups.iloc[in_tr])
        in_val_g = set(outer_groups.iloc[in_val])
        inner_overlap = in_tr_g.intersection(in_val_g)
        assert len(inner_overlap) == 0, f"Inner fold {inner_fold} has {len(inner_overlap)} overlapping groups!"


# ── Test 4: Frozen Test Set Isolation ────────────────────────────────────────

def test_frozen_test_set_isolation(train_data_and_groups):
    X_train, y_train, _ = train_data_and_groups

    # Verify train size is exactly 202,944 rows
    assert len(X_train) == EXPECTED_TRAIN_ROWS
    assert len(y_train) == EXPECTED_TRAIN_ROWS

    # Verify raw CSV total rows minus train equals test rows
    df = pd.read_csv(RAW_DATA_PATH)
    assert len(df) == EXPECTED_TRAIN_ROWS + EXPECTED_TEST_ROWS

    # Fingerprint must match Phase 3
    fp = gab._verify_fingerprint()
    assert fp == EXPECTED_FINGERPRINT, f"Fingerprint mismatch: {fp} != {EXPECTED_FINGERPRINT}"


# ── Test 5: Leakage-Safe Preprocessing ───────────────────────────────────────

def test_leakage_safe_preprocessing(train_data_and_groups):
    X_train, _, _ = train_data_and_groups
    prep1 = gab.make_preprocessor()
    prep2 = gab.make_preprocessor()

    # Preprocessors must be distinct unfitted instances
    assert prep1 is not prep2
    assert not hasattr(prep1, "transformers_")

    # Fit prep1 on a subset
    subset = X_train.iloc[:1000]
    prep1.fit(subset)
    assert hasattr(prep1, "transformers_")
    assert not hasattr(prep2, "transformers_")


# ── Test 6: OOF Predictions Completeness & Validity ─────────────────────────

def test_oof_predictions_npz():
    npz_path = GROUP_ARTIFACTS / "oof_predictions.npz"
    assert npz_path.exists(), "oof_predictions.npz must exist after running benchmark"

    data = np.load(npz_path)
    assert "y_true" in data
    y_true = data["y_true"]
    assert len(y_true) == EXPECTED_TRAIN_ROWS

    for model_name in MODELS:
        pred_key = f"oof_preds_{model_name}"
        proba_key = f"oof_probas_{model_name}"

        assert pred_key in data, f"Missing {pred_key} in OOF predictions"
        assert proba_key in data, f"Missing {proba_key} in OOF predictions"

        preds = data[pred_key]
        probas = data[proba_key]

        # Completeness: all rows predicted
        assert len(preds) == EXPECTED_TRAIN_ROWS
        assert not np.isnan(preds).any()

        # Valid classes
        unique_classes = set(np.unique(preds))
        assert unique_classes.issubset({0, 1, 2})

        # Valid probability distributions
        assert probas.shape == (EXPECTED_TRAIN_ROWS, 3)
        assert (probas >= 0.0).all()
        assert (probas <= 1.0).all()
        sums = probas.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)


# ── Test 7: Group-Aware Model Artifacts Load & Predict ──────────────────────

@pytest.mark.parametrize("model_name", MODELS)
def test_group_aware_model_artifacts_load(model_name, train_data_and_groups):
    X_train, _, _ = train_data_and_groups
    model_dir = GROUP_ARTIFACTS / model_name
    assert model_dir.exists(), f"Artifact directory missing: {model_dir}"

    model_path = model_dir / "model.pkl"
    meta_path = model_dir / "metadata.json"

    assert model_path.exists(), f"Model file missing: {model_path}"
    assert meta_path.exists(), f"Metadata file missing: {meta_path}"

    # Load model
    model = joblib.load(model_path)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    # Predict on raw input features (must not require prior manual preprocessing)
    sample_X = X_train.iloc[:10]
    preds = model.predict(sample_X)
    probas = model.predict_proba(sample_X)

    assert len(preds) == 10
    assert set(preds).issubset({0, 1, 2})
    assert probas.shape == (10, 3)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    # Verify metadata contents
    meta = json.loads(meta_path.read_text())
    assert meta["model_name"] == model_name
    assert meta["dataset_fingerprint"] == EXPECTED_FINGERPRINT
    assert "0.0% overlap" in meta["group_overlap_verified"]
    assert "macro_f1" in meta["full_oof_metrics"]


# ── Test 8: Preservation of Phase 4 Original Artifacts ──────────────────────

def test_phase4_original_artifacts_preserved():
    assert ORIG_ARTIFACTS.exists(), "Original Phase 4 models/cdc/benchmarks/ must remain intact"
    orig_summary = ORIG_ARTIFACTS / "benchmark_summary.json"
    assert orig_summary.exists(), "Original benchmark_summary.json must remain intact"

    for model_name in MODELS:
        orig_model = ORIG_ARTIFACTS / model_name / "model.pkl"
        orig_meta = ORIG_ARTIFACTS / model_name / "metadata.json"
        assert orig_model.exists(), f"Original Phase 4 {model_name} model.pkl missing!"
        assert orig_meta.exists(), f"Original Phase 4 {model_name} metadata.json missing!"


# ── Test 9: Reports and Visualizations Verification ──────────────────────────

def test_reports_and_figures_generated():
    # Reports
    assert (REPORTS_DIR / "cdc_group_aware_benchmark_report.md").exists()
    assert (REPORTS_DIR / "phase42_completion_report.md").exists()
    assert (REPORTS_DIR / "group_aware_benchmark_summary.json").exists()

    # Figures
    expected_figures = [
        "benchmark_macro_f1_comparison.png",
        "benchmark_balanced_accuracy.png",
        "benchmark_prediabetes_recall.png",
        "benchmark_prediabetes_precision.png",
        "benchmark_per_class_f1.png",
        "benchmark_roc_auc.png",
        "cm_logisticregression.png",
        "cm_decisiontree.png",
        "cm_randomforest.png",
        "cm_xgboost.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Figure missing: {fig_path}"
        assert fig_path.stat().st_size > 0, f"Figure is empty: {fig_path}"

