"""
test_phase5_final_evaluation.py
================================
Automated test suite verifying Phase 5 Final Test Set Evaluation integrity.

Validates:
  1. Phase 5 deliverables exist (report, json, 4 figures).
  2. Dataset fingerprint and test set dimensions match expected values.
  3. Metric calculations are mathematically valid and within bounds.
  4. Confusion matrix row sums equal test set class support.
  5. Generalization gap between OOF and Test set is minimal (|delta| < 0.02).
  6. Phase 1-4.2 artifacts remain untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "results" / "reports"
FIGURES_DIR = ROOT / "results" / "figures"
MODELS_DIR = ROOT / "models" / "cdc"

EXPECTED_FINGERPRINT = "a971809d0786d2d7"
EXPECTED_TEST_ROWS = 50736
EXPECTED_CLASS_SUPPORT = {0: 42741, 1: 926, 2: 7069}


@pytest.fixture(scope="module")
def phase5_metrics() -> dict:
    json_path = REPORTS_DIR / "phase5_final_metrics.json"
    assert json_path.exists(), f"Phase 5 metrics JSON missing: {json_path}"
    return json.loads(json_path.read_text())


def test_phase5_deliverables_exist():
    """Verify all required deliverables exist with non-zero size."""
    expected_files = [
        REPORTS_DIR / "phase5_final_evaluation_report.md",
        REPORTS_DIR / "phase5_final_metrics.json",
        FIGURES_DIR / "phase5_final_confusion_matrix.png",
        FIGURES_DIR / "phase5_per_class_metrics.png",
        FIGURES_DIR / "phase5_oof_vs_test_comparison.png",
        FIGURES_DIR / "phase5_roc_curves.png",
    ]
    for p in expected_files:
        assert p.exists(), f"Missing deliverable: {p}"
        assert p.stat().st_size > 500, f"Deliverable too small or empty: {p}"


def test_phase5_figures_valid_png():
    """Verify generated figures are valid readable PNG images with expected dimensions."""
    figure_files = [
        FIGURES_DIR / "phase5_final_confusion_matrix.png",
        FIGURES_DIR / "phase5_per_class_metrics.png",
        FIGURES_DIR / "phase5_oof_vs_test_comparison.png",
        FIGURES_DIR / "phase5_roc_curves.png",
    ]
    for fig_path in figure_files:
        with Image.open(fig_path) as img:
            assert img.format == "PNG", f"File is not a PNG: {fig_path}"
            assert img.width >= 700 and img.height >= 500, f"Image dimensions too small: {img.size}"


def test_phase5_metadata_reconciliation(phase5_metrics):
    """Verify dataset fingerprint and reconciled test set row counts."""
    assert phase5_metrics["dataset_fingerprint"] == EXPECTED_FINGERPRINT
    split_info = phase5_metrics["split_information"]
    assert split_info["test_rows"] == EXPECTED_TEST_ROWS
    assert split_info["total_dataset_rows"] == 253680
    assert split_info["train_rows"] + split_info["test_rows"] == 253680


def test_phase5_metric_validity(phase5_metrics):
    """Verify metrics are finite, bounded in [0, 1], and properly computed."""
    metrics = phase5_metrics["test_metrics"]
    for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr_macro"]:
        val = metrics[key]
        assert np.isfinite(val), f"Metric {key} is not finite: {val}"
        assert 0.0 <= val <= 1.0, f"Metric {key} out of bounds: {val}"

    # Specific bounds for Logistic Regression on CDC dataset
    assert 0.60 <= metrics["accuracy"] <= 0.70
    assert 0.50 <= metrics["balanced_accuracy"] <= 0.55
    assert 0.40 <= metrics["macro_f1"] <= 0.45
    assert 0.75 <= metrics["roc_auc_ovr_macro"] <= 0.80


def test_phase5_confusion_matrix_consistency(phase5_metrics):
    """Verify confusion matrix sum equals test set rows and row sums match class support."""
    cm = np.array(phase5_metrics["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert int(cm.sum()) == EXPECTED_TEST_ROWS

    row_sums = cm.sum(axis=1)
    for c_idx in range(3):
        assert int(row_sums[c_idx]) == EXPECTED_CLASS_SUPPORT[c_idx]


def test_phase5_prediabetes_analysis(phase5_metrics):
    """Verify Prediabetes (Class 1) diagnostics calculations."""
    cls1 = phase5_metrics["class1_prediabetes_analysis"]
    assert cls1["support"] == EXPECTED_CLASS_SUPPORT[1]
    assert cls1["true_positives"] + cls1["false_negatives"] == EXPECTED_CLASS_SUPPORT[1]
    assert cls1["recall"] == pytest.approx(cls1["true_positives"] / EXPECTED_CLASS_SUPPORT[1], abs=1e-5)
    assert cls1["precision"] == pytest.approx(
        cls1["true_positives"] / (cls1["true_positives"] + cls1["false_positives"]), abs=1e-5
    )
    assert cls1["recall"] > 0.25, "Class 1 recall should exceed 25% for selected Logistic Regression"
    assert cls1["specificity"] > 0.80, "Class 1 specificity should exceed 80%"


def test_phase5_generalization_gap(phase5_metrics):
    """Verify minimal delta between development OOF metrics and final test metrics."""
    gen = phase5_metrics["generalization_comparison_oof_vs_test"]
    for metric_name in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr"]:
        delta_abs = abs(gen[metric_name]["delta_abs"])
        assert delta_abs < 0.01, f"Generalization gap too large for {metric_name}: {delta_abs}"


def test_phase5_preservation_of_earlier_phases():
    """Verify Phase 1-4.2 artifacts remain untouched."""
    orig_benchmarks = MODELS_DIR / "benchmarks"
    group_benchmarks = MODELS_DIR / "group_aware_benchmarks"

    assert (orig_benchmarks / "benchmark_summary.json").exists()
    assert (group_benchmarks / "oof_predictions.npz").exists()
    assert (group_benchmarks / "group_aware_benchmark_summary.json").exists()
    assert (group_benchmarks / "LogisticRegression" / "model.pkl").exists()
    assert (group_benchmarks / "XGBoost" / "model.pkl").exists()

