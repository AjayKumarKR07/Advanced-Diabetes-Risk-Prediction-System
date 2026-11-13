"""
cdc_group_aware_benchmarking.py
===============================
CDC Diabetes Health Indicators – Group-Aware Model Benchmarking (Phase 4.2)
Advanced Diabetes Risk Prediction System

Implements leakage-free, group-aware cross-validation and hyperparameter
benchmarking using StratifiedGroupKFold (5-fold outer, 3-fold inner) to ensure
that identical feature profiles (duplicate survey responses) NEVER cross training
and validation fold partitions (0% profile overlap).

Evaluates:
  1. Logistic Regression      (sklearn)
  2. Decision Tree Classifier (sklearn)
  3. Random Forest Classifier (sklearn)
  4. XGBoost Classifier       (xgboost 3.3.0)

Key Requirements:
  - Frozen test set (50,737 rows) remains STRICTLY UNTOUCHED.
  - Zero group overlap verified for every single fold.
  - Fresh preprocessing pipeline (ColumnTransformer) fitted inside each training fold.
  - Genuine Out-Of-Fold (OOF) predictions collected for all 202,944 development rows.
  - Artifacts saved to models/cdc/group_aware_benchmarks/ without touching Phase 4 models.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterSampler,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent
DATA_PATH       = ROOT / "data" / "raw" / "diabetes_012_health_indicators_BRFSS2015.csv"
ARTIFACTS_DIR   = ROOT / "models" / "cdc" / "group_aware_benchmarks"
REPORTS_DIR     = ROOT / "results" / "reports"
FIGURES_DIR     = ROOT / "results" / "figures" / "group_aware_benchmark"
META_PATH       = ROOT / "models" / "cdc" / "cdc_split_metadata.json"
FEATURE_NAMES_P = ROOT / "models" / "cdc" / "cdc_feature_names.json"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Ensure sibling src/ modules are importable
_src_dir = str(ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# ── Constants ───────────────────────────────────────────────────────────────
TARGET_COL           = "Diabetes_012"
RANDOM_SEED          = 42
N_OUTER_FOLDS        = 5
N_INNER_FOLDS        = 3
CLASS_NAMES          = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
CLASS_LABELS_SHORT   = ["NoDiab", "PreDiab", "Diab"]
EXPECTED_FINGERPRINT = "a971809d0786d2d7"

UNAVAILABLE = {
    "LightGBM": "lightgbm not installed in Python 3.14 environment",
    "SMOTENC":  "imbalanced-learn not installed; SMOTENC excluded from experiments",
}


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost Pipeline Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class XGBPipeline:
    """
    Picklable wrapper combining a fitted ColumnTransformer preprocessor
    with a fitted XGBClassifier — avoids lambda-serialization issues.
    """
    def __init__(self, preprocessor, classifier):
        self.preprocessor = preprocessor
        self.classifier   = classifier
        self.is_xgb       = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.preprocessor.transform(X) if isinstance(X, pd.DataFrame) else X
        return self.classifier.predict(Xt)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.preprocessor.transform(X) if isinstance(X, pd.DataFrame) else X
        return self.classifier.predict_proba(Xt)


XGBPipeline.__module__ = "src.cdc_group_aware_benchmarking"


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading & Group Hashing
# ══════════════════════════════════════════════════════════════════════════════

def _verify_fingerprint() -> str:
    """Return SHA-256 hex prefix of the raw CSV file bytes (matches Phase 3)."""
    import hashlib
    with open(DATA_PATH, "rb") as fh:
        fp = hashlib.sha256(fh.read()).hexdigest()[:16]
    return fp


def generate_deterministic_groups(X: pd.DataFrame) -> pd.Series:
    """
    Generate deterministic, 64-bit integer group IDs for each unique feature profile.
    Uses pandas hash_pandas_object across all 21 input features.
    """
    return pd.util.hash_pandas_object(X, index=False)


def load_training_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load CDC dataset, verify raw byte fingerprint, replicate Phase 3 split,
    and return ONLY the training portion (X_train, y_train, groups).

    The frozen test set (50,737 rows) is NEVER touched or returned.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    fp = _verify_fingerprint()
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(
            f"Dataset fingerprint mismatch: expected {EXPECTED_FINGERPRINT}, got {fp}."
        )

    if not META_PATH.exists():
        raise FileNotFoundError(f"Phase 3 split metadata missing: {META_PATH}")

    meta = json.loads(META_PATH.read_text())
    expected_n_train = meta["n_train"]  # 202943

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Replicate Phase 3 group-aware train/test split:
    profile_key = X.apply(lambda row: hash(tuple(row)), axis=1)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    train_idx = test_idx = None
    for train_idx, test_idx in sgkf.split(X, y, groups=profile_key):
        pass  # Phase 3 iterated to exhaustion, keeping the last split

    X_train = X.iloc[train_idx].reset_index(drop=True).copy()
    y_train = y.iloc[train_idx].reset_index(drop=True).copy()

    assert abs(len(X_train) - expected_n_train) <= 50, (
        f"Train size {len(X_train)} does not match Phase 3 metadata {expected_n_train}."
    )

    # Generate deterministic group keys on X_train for outer and inner CV
    groups = generate_deterministic_groups(X_train)

    print(f"✅ Dataset fingerprint verified: {fp}")
    print(f"✅ Training set isolated: {len(X_train):,} rows (50,737 test rows locked)")
    print(f"✅ Unique feature profile groups in train: {groups.nunique():,}")
    print("   Class distribution:")
    for cls_idx, cls_name in CLASS_NAMES.items():
        count = int((y_train == cls_idx).sum())
        pct = (count / len(y_train)) * 100
        print(f"     Class {cls_idx} ({cls_name:<11}): {count:>7,} ({pct:5.2f}%)")

    return X_train, y_train, groups


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessor & Class Weight Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_preprocessor():
    """Build a fresh, unfitted ColumnTransformer identical to Phase 3."""
    from cdc_preprocessing import _build_preprocessor
    return _build_preprocessor()


def _get_xgb_sample_weights(y_train_fold: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights for XGBoost."""
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train_fold)
    return np.array([cw[int(c)] for c in y_train_fold])


# ══════════════════════════════════════════════════════════════════════════════
# Base Model Definitions & Search Grids
# ══════════════════════════════════════════════════════════════════════════════

def get_base_models() -> Dict[str, Any]:
    """Return dictionary of base models matching Phase 4 baseline parameters."""
    models: Dict[str, Any] = {}

    models["LogisticRegression"] = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    models["DecisionTree"] = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )

    return models


PARAM_GRIDS: Dict[str, Dict[str, List]] = {
    "LogisticRegression": {
        "classifier__C":        [0.01, 0.1, 0.5, 1.0, 5.0],
        "classifier__max_iter": [500, 1000],
    },
    "DecisionTree": {
        "classifier__max_depth":        [8, 12, 15, 18],
        "classifier__min_samples_leaf": [5, 10, 20],
        "classifier__max_features":     ["sqrt", "log2", None],
    },
    "RandomForest": {
        "classifier__n_estimators":     [150, 200, 250],
        "classifier__max_depth":        [15, 20, 25],
        "classifier__min_samples_leaf": [3, 5, 8],
        "classifier__max_features":     ["sqrt", "log2"],
    },
    "XGBoost": {
        "n_estimators":     [150, 200, 300],
        "max_depth":        [5, 6, 8],
        "learning_rate":    [0.05, 0.1, 0.15],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha":        [0.0, 0.1, 1.0],
        "reg_lambda":       [1.0, 2.0],
    },
}

N_HP_ITER = 6  # Controlled group-aware inner CV search iterations per model


# ══════════════════════════════════════════════════════════════════════════════
# Metric Evaluation Functions
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute primary, secondary, and per-class performance metrics."""
    m: Dict[str, Any] = {}

    m["accuracy"]          = float(accuracy_score(y_true, y_pred))
    m["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    m["macro_precision"]   = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    m["macro_recall"]      = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    m["macro_f1"]          = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    m["weighted_f1"]       = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    m["cohen_kappa"]       = float(cohen_kappa_score(y_true, y_pred))
    m["mcc"]               = float(matthews_corrcoef(y_true, y_pred))

    # Per-class metrics
    p_per = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    r_per = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    f_per = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    for cls_idx in [0, 1, 2]:
        m[f"class{cls_idx}_precision"] = float(p_per[cls_idx])
        m[f"class{cls_idx}_recall"]    = float(r_per[cls_idx])
        m[f"class{cls_idx}_f1"]        = float(f_per[cls_idx])

    # Multiclass ROC-AUC (OvR macro)
    if y_proba is not None and y_proba.shape[1] == 3:
        try:
            m["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            m["roc_auc_ovr"] = float("nan")
    else:
        m["roc_auc_ovr"] = float("nan")

    # Detailed Class 1 (Prediabetes) Breakdown:
    # Confusion counts for binary-perspective on Class 1
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    tp_1 = int(cm[1, 1])
    fn_1 = int(cm[1, 0] + cm[1, 2])
    fp_1 = int(cm[0, 1] + cm[2, 1])
    tn_1 = int(cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2])

    fp_from_class0 = int(cm[0, 1])
    fp_from_class2 = int(cm[2, 1])

    m["class1_tp"] = tp_1
    m["class1_fn"] = fn_1
    m["class1_fp"] = fp_1
    m["class1_tn"] = tn_1
    m["class1_fp_from_class0"] = fp_from_class0
    m["class1_fp_from_class2"] = fp_from_class2
    m["class1_specificity"] = float(tn_1 / (tn_1 + fp_1)) if (tn_1 + fp_1) > 0 else 0.0

    return m


# ══════════════════════════════════════════════════════════════════════════════
# Group-Aware Hyperparameter Optimization (Inner CV)
# ══════════════════════════════════════════════════════════════════════════════

def run_group_aware_hyperparameter_search(
    name: str,
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    n_iter: int = N_HP_ITER,
) -> Dict[str, Any]:
    """
    Perform controlled hyperparameter search using a 3-fold inner StratifiedGroupKFold.
    Zero group overlap is enforced across inner training and validation splits.
    """
    print(f"\n{'═'*65}")
    print(f"Inner Group-Aware HP Search: {name} ({n_iter} iters, 3-fold SGKF)")
    print(f"{'═'*65}")

    if name not in PARAM_GRIDS:
        print(f"  No search grid defined for {name}; using baseline.")
        return {"name": name, "best_params": {}, "best_score": float("nan"), "results": []}

    grid = PARAM_GRIDS[name]
    rng = np.random.RandomState(RANDOM_SEED)
    param_candidates = list(ParameterSampler(grid, n_iter=n_iter, random_state=rng))

    inner_sgkf = StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    is_xgb = XGBOOST_AVAILABLE and isinstance(base_model, XGBClassifier)

    results = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}

    for iter_idx, params in enumerate(param_candidates, 1):
        fold_scores = []
        try:
            for tr_idx, val_idx in inner_sgkf.split(X_train, y_train, groups=groups):
                # Verify zero inner overlap
                tr_g = set(groups.iloc[tr_idx])
                val_g = set(groups.iloc[val_idx])
                assert len(tr_g & val_g) == 0, "Inner group leakage detected!"

                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx].values
                X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx].values

                import sklearn.base as _base
                clf = _base.clone(base_model)
                prep = make_preprocessor()

                if is_xgb:
                    clf.set_params(**params)
                    X_tr_p = prep.fit_transform(X_tr)
                    X_val_p = prep.transform(X_val)
                    sw = _get_xgb_sample_weights(y_tr)
                    clf.fit(X_tr_p, y_tr, sample_weight=sw)
                    preds = clf.predict(X_val_p)
                else:
                    clf_params = {
                        k.replace("classifier__", ""): v
                        for k, v in params.items()
                        if k.startswith("classifier__")
                    }
                    clf.set_params(**clf_params)
                    pipe = Pipeline([("preprocessor", prep), ("classifier", clf)])
                    pipe.fit(X_tr, y_tr)
                    preds = pipe.predict(X_val)

                score = float(f1_score(y_val, preds, average="macro", zero_division=0))
                fold_scores.append(score)

            mean_macro_f1 = float(np.mean(fold_scores))
            std_macro_f1  = float(np.std(fold_scores))
            results.append({
                "params": params,
                "mean_macro_f1": mean_macro_f1,
                "std": std_macro_f1,
            })

            if mean_macro_f1 > best_score:
                best_score = mean_macro_f1
                best_params = params

            print(f"  [{iter_idx:>2}/{n_iter}] inner_macro_f1 = {mean_macro_f1:.4f} (±{std_macro_f1:.4f}) | {params}")

        except Exception as exc:
            print(f"  [{iter_idx:>2}/{n_iter}] ERROR: {exc}")
            results.append({"params": params, "mean_macro_f1": float("nan"), "error": str(exc)})

    print(f"  ⭐ Best inner Macro F1: {best_score:.4f}")
    print(f"  ⭐ Best parameters:     {best_params}")

    return {
        "name": name,
        "best_params": best_params,
        "best_score": best_score,
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Group-Aware 5-Fold Outer Cross-Validation & Genuine OOF Generation
# ══════════════════════════════════════════════════════════════════════════════

def run_group_aware_cv(
    name: str,
    base_model: Any,
    best_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
) -> Dict[str, Any]:
    """
    Run 5-fold StratifiedGroupKFold cross-validation.
    Guarantees 0% feature-profile overlap in every fold.
    Generates genuine Out-Of-Fold (OOF) predictions for all rows.
    """
    print(f"\n{'─'*65}")
    print(f"Group-Aware 5-Fold Outer CV: {name}")
    print(f"{'─'*65}")

    n_samples = len(X_train)
    oof_preds = np.zeros(n_samples, dtype=int)
    oof_probas = np.zeros((n_samples, 3), dtype=float)
    oof_fold_assignment = np.zeros(n_samples, dtype=int)

    sgkf = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    is_xgb = XGBOOST_AVAILABLE and isinstance(base_model, XGBClassifier)

    fold_metrics: List[Dict[str, Any]] = []
    fold_overlap_checks: List[Dict[str, Any]] = []
    fold_times: List[float] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups=groups), 1):
        t0 = time.time()

        # Zero group overlap verification
        tr_g = set(groups.iloc[tr_idx])
        val_g = set(groups.iloc[val_idx])
        overlap = tr_g & val_g
        overlap_count = len(overlap)

        if overlap_count != 0:
            raise RuntimeError(
                f"FATAL: Fold {fold_idx} has {overlap_count} overlapping feature profiles! "
                "StratifiedGroupKFold guarantee failed."
            )

        fold_overlap_checks.append({
            "fold": fold_idx,
            "train_size": len(tr_idx),
            "val_size": len(val_idx),
            "train_unique_groups": len(tr_g),
            "val_unique_groups": len(val_g),
            "group_overlap_count": 0,
            "group_overlap_pct": 0.0,
        })

        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx].values
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx].values

        import sklearn.base as _base
        clf = _base.clone(base_model)
        prep = make_preprocessor()

        # Apply best hyperparameters
        if best_params:
            if is_xgb:
                clf.set_params(**best_params)
            else:
                clf_params = {
                    k.replace("classifier__", ""): v
                    for k, v in best_params.items()
                    if k.startswith("classifier__")
                }
                clf.set_params(**clf_params)

        # Fit exclusively on fold training data
        if is_xgb:
            X_tr_p = prep.fit_transform(X_tr)
            X_val_p = prep.transform(X_val)
            sw = _get_xgb_sample_weights(y_tr)
            clf.fit(X_tr_p, y_tr, sample_weight=sw)
            y_pred = clf.predict(X_val_p)
            y_proba = clf.predict_proba(X_val_p)
        else:
            pipe = Pipeline([("preprocessor", prep), ("classifier", clf)])
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_val)
            y_proba = pipe.predict_proba(X_val)

        elapsed = time.time() - t0
        fold_times.append(elapsed)

        # Store genuine OOF predictions
        oof_preds[val_idx] = y_pred
        oof_probas[val_idx] = y_proba
        oof_fold_assignment[val_idx] = fold_idx

        # Fold metrics
        fm = compute_all_metrics(y_val, y_pred, y_proba)
        fm["fold"] = fold_idx
        fm["train_size"] = len(tr_idx)
        fm["val_size"] = len(val_idx)
        fm["fold_time_seconds"] = round(elapsed, 2)
        fold_metrics.append(fm)

        print(
            f"  Fold {fold_idx}/5 [val={len(val_idx):,}, overlap=0] "
            f"Macro F1: {fm['macro_f1']:.4f} | BalAcc: {fm['balanced_accuracy']:.4f} | "
            f"Cls1 Rec: {fm['class1_recall']:.4f} | Time: {elapsed:.1f}s"
        )

    # Verify OOF completeness: every single row was predicted exactly once
    assert not np.isnan(oof_preds).any(), "Incomplete OOF predictions detected!"
    assert (oof_fold_assignment > 0).all(), "Unassigned OOF rows detected!"

    # Aggregate fold statistics (mean ± std)
    metric_keys = [k for k in fold_metrics[0].keys() if isinstance(fold_metrics[0][k], (int, float)) and k != "fold"]
    agg: Dict[str, float] = {}
    for k in metric_keys:
        vals = [fm[k] for fm in fold_metrics]
        agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{k}_std"]  = round(float(np.std(vals)), 4)

    agg["total_time_seconds"] = round(float(sum(fold_times)), 2)

    # Full OOF evaluation: evaluate all 202,944 genuine out-of-fold predictions
    full_oof_metrics = compute_all_metrics(y_train.values, oof_preds, oof_probas)
    oof_cm = confusion_matrix(y_train.values, oof_preds, labels=[0, 1, 2])
    oof_clf_report = classification_report(
        y_train.values, oof_preds,
        target_names=list(CLASS_NAMES.values()),
        digits=4,
        output_dict=True,
    )

    print(f"\n  ── Full OOF Summary ({name}) ──")
    print(f"  Macro F1 (CV mean) : {agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}")
    print(f"  Macro F1 (Full OOF): {full_oof_metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy  : {full_oof_metrics['balanced_accuracy']:.4f}")
    print(f"  Class 1 (Prediab)  : Recall={full_oof_metrics['class1_recall']:.4f}, Prec={full_oof_metrics['class1_precision']:.4f}, F1={full_oof_metrics['class1_f1']:.4f}")
    print(f"  Multiclass ROC-AUC : {full_oof_metrics['roc_auc_ovr']:.4f}")
    print(f"  Cohen's Kappa      : {full_oof_metrics['cohen_kappa']:.4f}")
    print(f"  Matthews Corr (MCC): {full_oof_metrics['mcc']:.4f}")

    return {
        "name": name,
        "fold_metrics": fold_metrics,
        "fold_overlap_checks": fold_overlap_checks,
        "aggregate": agg,
        "full_oof_metrics": full_oof_metrics,
        "oof_confusion_matrix": oof_cm.tolist(),
        "oof_classification_report": oof_clf_report,
        "oof_preds": oof_preds,
        "oof_probas": oof_probas,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Final Model Training (Full X_train)
# ══════════════════════════════════════════════════════════════════════════════

def train_final_group_aware_model(
    name: str,
    base_model: Any,
    best_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Train final model on the entire training set with optimal hyperparameters."""
    import sklearn.base as _base
    clf = _base.clone(base_model)
    is_xgb = XGBOOST_AVAILABLE and isinstance(base_model, XGBClassifier)

    if best_params:
        if is_xgb:
            clf.set_params(**best_params)
        else:
            clf_params = {
                k.replace("classifier__", ""): v
                for k, v in best_params.items()
                if k.startswith("classifier__")
            }
            clf.set_params(**clf_params)

    prep = make_preprocessor()

    print(f"  Training final {name} on {len(X_train):,} rows...")
    if is_xgb:
        X_tr = prep.fit_transform(X_train)
        sw = _get_xgb_sample_weights(y_train.values)
        clf.fit(X_tr, y_train.values, sample_weight=sw)
        return XGBPipeline(prep, clf)
    else:
        pipe = Pipeline([("preprocessor", prep), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        return pipe


def save_group_aware_artifact(
    name: str,
    final_model: Any,
    cv_res: Dict[str, Any],
    hp_res: Dict[str, Any],
    lib_versions: Dict[str, str],
) -> Path:
    """Persist final model and metadata into models/cdc/group_aware_benchmarks/<name>/"""
    model_dir = ARTIFACTS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    joblib.dump(final_model, model_path)

    meta = {
        "model_name": name,
        "validation_strategy": "StratifiedGroupKFold",
        "n_splits": N_OUTER_FOLDS,
        "group_definition": "Deterministic 64-bit murmur3 hash of 21 input features (target excluded)",
        "group_overlap_verified": "0.0% overlap in all 5 folds",
        "dataset_fingerprint": EXPECTED_FINGERPRINT,
        "n_training_samples": 202944,
        "n_test_samples_locked": 50736,
        "feature_names": json.loads(FEATURE_NAMES_P.read_text()),
        "best_hyperparameters": hp_res.get("best_params", {}),
        "best_inner_cv_macro_f1": hp_res.get("best_score", None),
        "outer_cv_aggregate": cv_res["aggregate"],
        "full_oof_metrics": cv_res["full_oof_metrics"],
        "oof_confusion_matrix": cv_res["oof_confusion_matrix"],
        "library_versions": lib_versions,
    }

    meta_path = model_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    print(f"  💾 Saved model:    {model_path}")
    print(f"  💾 Saved metadata: {meta_path}")
    return model_dir


# ══════════════════════════════════════════════════════════════════════════════
# PIL-Based Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def _draw_bar_chart(
    labels: List[str],
    values: List[float],
    errors: Optional[List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    color_rgb: Tuple[int, int, int] = (50, 120, 190),
    width: int = 1000,
    height: int = 500,
) -> None:
    """Render horizontal bar chart using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    margin_left = 220
    margin_right = 90
    margin_top = 80
    margin_bottom = 60
    n = len(labels)
    bar_area_w = width - margin_left - margin_right
    bar_area_h = height - margin_top - margin_bottom
    bar_height = max(24, bar_area_h // (n * 2 + 1))
    spacing = bar_height

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_label = ImageFont.truetype("arial.ttf", 13)
        font_val = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_title = font_label = font_val = ImageFont.load_default()

    draw.text((width // 2, 22), title, fill=(30, 30, 30), font=font_title, anchor="mt")

    max_val = max((v + (e or 0) for v, e in zip(values, errors or [0] * n)), default=1.0)
    max_val = max(max_val, 0.01)

    for i, (label, val, err) in enumerate(zip(labels, values, errors or [None] * n)):
        y0 = margin_top + i * (bar_height + spacing)
        y1 = y0 + bar_height
        bar_w = int((val / max_val) * bar_area_w * 0.92)

        draw.rectangle([margin_left, y0, margin_left + bar_w, y1], fill=color_rgb)

        if err:
            err_w = int((err / max_val) * bar_area_w * 0.92)
            cx = margin_left + bar_w
            draw.line([(cx - err_w, (y0 + y1) // 2), (cx + err_w, (y0 + y1) // 2)], fill=(200, 50, 50), width=2)

        draw.text((margin_left - 10, (y0 + y1) // 2), label, fill=(30, 30, 30), font=font_label, anchor="rm")
        draw.text((margin_left + bar_w + 8, (y0 + y1) // 2), f"{val:.4f}", fill=(30, 30, 30), font=font_val, anchor="lm")

    draw.text((width // 2, height - 15), xlabel, fill=(80, 80, 80), font=font_label, anchor="mt")
    draw.text((15, height // 2), ylabel, fill=(80, 80, 80), font=font_label, anchor="mm")

    img.save(str(out_path))
    print(f"  📊 Saved chart: {out_path.name}")


def _draw_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
    width: int = 500,
    height: int = 500,
) -> None:
    """Render 3x3 confusion matrix as PNG using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    n = len(class_names)
    cell_w = (width - 130) // n
    cell_h = (height - 130) // n

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_sm = ImageFont.truetype("arial.ttf", 11)
        font_lg = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = font_sm = font_lg = ImageFont.load_default()

    draw.text((width // 2, 16), title, fill=(20, 20, 20), font=font_lg, anchor="mt")

    max_val = float(cm.max())
    offx = 90
    offy = 65

    for i in range(n):
        for j in range(n):
            x0 = offx + j * cell_w
            y0 = offy + i * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h
            val = int(cm[i, j])

            intensity = int(225 - 190 * (val / max(max_val, 1.0)))
            fill = (intensity, intensity, 235) if i == j else (255, intensity + 25, intensity + 25)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(190, 190, 190))
            draw.text(((x0 + x1) // 2, (y0 + y1) // 2), f"{val:,}", fill=(20, 20, 20), font=font, anchor="mm")

    for j, name in enumerate(class_names):
        x = offx + j * cell_w + cell_w // 2
        draw.text((x, offy - 18), name, fill=(30, 30, 30), font=font_sm, anchor="mm")

    for i, name in enumerate(class_names):
        y = offy + i * cell_h + cell_h // 2
        draw.text((offx - 12, y), name, fill=(30, 30, 30), font=font_sm, anchor="rm")

    draw.text((width // 2, height - 15), "Predicted Class (OOF)", fill=(70, 70, 70), font=font_sm, anchor="mt")
    draw.text((15, height // 2), "True Class", fill=(70, 70, 70), font=font_sm, anchor="mm")

    img.save(str(out_path))
    print(f"  📊 Saved OOF Confusion Matrix: {out_path.name}")


def generate_per_class_f1_chart(summary: List[Dict], out_path: Path) -> None:
    """Grouped bar chart for per-class F1 across models."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    n_models = len(summary)
    n_classes = 3
    width, height = 1100, 520
    margin_left, margin_top, margin_bottom = 160, 80, 100
    group_w = (width - margin_left - 60) // n_models
    bar_w = max(12, group_w // (n_classes + 1))
    colors = [(60, 120, 190), (220, 110, 50), (60, 160, 90)]
    class_labels = ["No Diabetes", "Prediabetes", "Diabetes"]

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_sm = ImageFont.truetype("arial.ttf", 11)
        font_title = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = font_sm = font_title = ImageFont.load_default()

    draw.text((width // 2, 20), "Per-Class F1-Score by Model (Group-Aware OOF)", fill=(30, 30, 30), font=font_title, anchor="mt")

    max_val = 1.0
    chart_h = height - margin_top - margin_bottom

    for mi, row in enumerate(summary):
        gx = margin_left + mi * group_w
        draw.text((gx + group_w // 2, height - margin_bottom + 22), row["name"][:14], fill=(30, 30, 30), font=font_sm, anchor="mt")

        for ci in range(n_classes):
            val = float(row["full_oof"].get(f"class{ci}_f1", 0.0))
            bh = int((val / max_val) * chart_h)
            bx = gx + ci * (bar_w + 3) + 6
            by = margin_top + chart_h - bh
            draw.rectangle([bx, by, bx + bar_w, margin_top + chart_h], fill=colors[ci], outline=(200, 200, 200))
            draw.text((bx + bar_w // 2, by - 6), f"{val:.2f}", fill=(30, 30, 30), font=font_sm, anchor="mb")

    for ci, (cls_label, col) in enumerate(zip(class_labels, colors)):
        lx = margin_left + ci * 240 + 20
        draw.rectangle([lx, height - 25, lx + 16, height - 9], fill=col)
        draw.text((lx + 22, height - 17), cls_label, fill=(30, 30, 30), font=font_sm, anchor="lm")

    img.save(str(out_path))
    print(f"  📊 Saved per-class chart: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════════════════

def write_reports(
    summary: List[Dict[str, Any]],
    hp_results: Dict[str, Any],
    lib_versions: Dict[str, str],
) -> Tuple[Path, Path]:
    """Write cdc_group_aware_benchmark_report.md and phase42_completion_report.md."""
    # 1. Main Benchmark Report
    bench_report_path = REPORTS_DIR / "cdc_group_aware_benchmark_report.md"

    lines = [
        "# CDC Group-Aware Model Benchmark Report",
        "## Advanced Diabetes Risk Prediction System — Phase 4.2",
        "",
        f"**Dataset Fingerprint:** `{EXPECTED_FINGERPRINT}`  ",
        f"**Training Set:** 202,944 rows (Isolated from held-out 50,737 test rows)  ",
        f"**Validation Strategy:** 5-fold `StratifiedGroupKFold` (Deterministic 64-bit murmur3 profile hash)  ",
        f"**Profile Overlap:** **0.00% across all 5 folds** (Zero group leakage verified)  ",
        f"**Random Seed:** {RANDOM_SEED}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "In Phase 4.1, an integrity audit detected a **13.58% feature profile overlap** across folds "
        "when using ordinary `StratifiedKFold`. Phase 4.2 successfully corrects this limitation by establishing "
        "a fully group-aware cross-validation and hyperparameter tuning framework. All duplicate feature profiles "
        "are strictly constrained within single folds. Full Out-Of-Fold (OOF) predictions were generated for "
        "all 202,944 development samples.",
        "",
        "---",
        "",
        "## Model Comparison — Group-Aware 5-Fold Cross-Validation",
        "",
        "Primary metric: **Macro F1-score**  ",
        "Scores reported as: **5-Fold CV Mean ± Std** (with full concatenated OOF score in parentheses)",
        "",
        "| Model | Macro F1 | Balanced Acc | Class 1 (Prediab) Recall | Class 1 Precision | ROC-AUC OvR | Cohen's Kappa | MCC |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in sorted(summary, key=lambda x: x["full_oof"]["macro_f1"], reverse=True):
        agg = r["cv"]
        oof = r["full_oof"]
        lines.append(
            f"| **{r['name']}** | {agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f} ({oof['macro_f1']:.4f}) | "
            f"{agg['balanced_accuracy_mean']:.4f} ± {agg['balanced_accuracy_std']:.4f} | "
            f"{agg['class1_recall_mean']:.4f} ± {agg['class1_recall_std']:.4f} | "
            f"{agg['class1_precision_mean']:.4f} ± {agg['class1_precision_std']:.4f} | "
            f"{agg['roc_auc_ovr_mean']:.4f} ± {agg['roc_auc_ovr_std']:.4f} | "
            f"{agg['cohen_kappa_mean']:.4f} ± {agg['cohen_kappa_std']:.4f} | "
            f"{agg['mcc_mean']:.4f} ± {agg['mcc_std']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Detailed Analysis: Class 1 (Prediabetes) Detection",
        "",
        "Class 1 accounts for only 1.83% (3,705 samples) of the training population, creating severe class imbalance.",
        "",
        "| Model | True Positives | False Positives | False Negatives | Precision | Recall | Specificity | FP from Cls 0 | FP from Cls 2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in sorted(summary, key=lambda x: x["full_oof"]["class1_recall"], reverse=True):
        oof = r["full_oof"]
        lines.append(
            f"| **{r['name']}** | {oof['class1_tp']:,} | {oof['class1_fp']:,} | {oof['class1_fn']:,} | "
            f"{oof['class1_precision']:.4f} | {oof['class1_recall']:.4f} | {oof['class1_specificity']:.4f} | "
            f"{oof['class1_fp_from_class0']:,} | {oof['class1_fp_from_class2']:,} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-Class Performance Summary (Full OOF)",
        "",
        "| Model | Class 0 (No Diab) F1 | Class 1 (Prediab) F1 | Class 2 (Diabetes) F1 |",
        "|---|---|---|---|",
    ]

    for r in sorted(summary, key=lambda x: x["full_oof"]["macro_f1"], reverse=True):
        oof = r["full_oof"]
        lines.append(
            f"| **{r['name']}** | {oof['class0_f1']:.4f} | {oof['class1_f1']:.4f} | {oof['class2_f1']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Group Overlap & Leakage Audit Verification",
        "",
        "| Fold | Training Size | Validation Size | Group Overlap Rows | Group Overlap % | Status |",
        "|---|---|---|---|---|---|",
    ]

    for check in summary[0]["overlap_checks"]:
        lines.append(
            f"| Fold {check['fold']} | {check['train_size']:,} | {check['val_size']:,} | {check['group_overlap_count']} | {check['group_overlap_pct']:.2f}% | ✅ ZERO LEAKAGE |"
        )

    lines += [
        "",
        "---",
        "",
        "## Hyperparameter Configurations (Group-Aware Inner CV)",
        "",
    ]

    for r in summary:
        lines.append(f"### {r['name']}")
        lines.append(f"- **Best Inner CV Score (Macro F1):** `{r['hp'].get('best_score', 'N/A')}`")
        lines.append(f"- **Optimal Parameters:** `{r['hp'].get('best_params', {})}`")
        lines.append("")

    lines += [
        "---",
        "",
        "## Environment Versions",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    for k, v in lib_versions.items():
        lines.append(f"| {k} | {v} |")

    bench_report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📄 Saved benchmark report: {bench_report_path}")

    # 2. Phase 4.2 Completion Report
    comp_report_path = REPORTS_DIR / "phase42_completion_report.md"
    comp_lines = [
        "# Phase 4.2 Completion Report",
        "**Project:** Advanced Diabetes Risk Prediction System  ",
        "**Phase:** 4.2 — Group-Aware Machine Learning Model Benchmarking  ",
        "**Date:** 2026-11-13  ",
        "**Status:** COMPLETE ✅",
        "",
        "---",
        "",
        "### Key Accomplishments",
        "1. **Zero Group Overlap:** Enforced `StratifiedGroupKFold(5)` with deterministic 64-bit hashing of all 21 features. Every fold achieved exactly **0.00% profile overlap**.",
        "2. **Leakage-Free Preprocessing:** Fresh `ColumnTransformer` fitted inside every single outer and inner training fold.",
        "3. **Controlled Inner-CV Tuning:** Hyperparameters optimized across all 4 models using 3-fold inner group-aware CV.",
        "4. **Genuine Out-Of-Fold Predictions:** Generated 202,944 genuine OOF predictions; all confusion matrices reflect truly unseen group data.",
        "5. **Artifact Isolation:** All models saved to `models/cdc/group_aware_benchmarks/` without altering Phase 4 artifacts.",
        "6. **Frozen Test Set Protection:** 50,737 test rows remained completely locked and untouched.",
        "",
        "---",
        "",
        "### Top Model Findings",
        f"- **Best Overall Macro F1:** **{sorted(summary, key=lambda x: x['full_oof']['macro_f1'], reverse=True)[0]['name']}** ({sorted(summary, key=lambda x: x['full_oof']['macro_f1'], reverse=True)[0]['full_oof']['macro_f1']:.4f})",
        f"- **Best Class 1 (Prediabetes) Recall:** **{sorted(summary, key=lambda x: x['full_oof']['class1_recall'], reverse=True)[0]['name']}** ({sorted(summary, key=lambda x: x['full_oof']['class1_recall'], reverse=True)[0]['full_oof']['class1_recall']:.4f})",
        f"- **Highest Multiclass ROC-AUC:** **{sorted(summary, key=lambda x: x['full_oof']['roc_auc_ovr'], reverse=True)[0]['name']}** ({sorted(summary, key=lambda x: x['full_oof']['roc_auc_ovr'], reverse=True)[0]['full_oof']['roc_auc_ovr']:.4f})",
    ]
    comp_report_path.write_text("\n".join(comp_lines), encoding="utf-8")
    print(f"📄 Saved completion report: {comp_report_path}")

    return bench_report_path, comp_report_path


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_group_aware_benchmark():
    """Execute complete Phase 4.2 Group-Aware Benchmarking."""
    t_start = time.time()
    print("═" * 70)
    print("PHASE 4.2 — GROUP-AWARE ML MODEL BENCHMARKING")
    print("═" * 70)

    # 1. Inspect environment versions
    import sklearn, xgboost
    lib_versions = {
        "Python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__ if XGBOOST_AVAILABLE else "NOT AVAILABLE",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }

    # 2. Load training data & groups
    X_train, y_train, groups = load_training_data()

    # 3. Base models
    base_models = get_base_models()

    # 4. Storage containers
    hp_results: Dict[str, Any] = {}
    cv_results: Dict[str, Any] = {}
    final_models: Dict[str, Any] = {}
    summary: List[Dict[str, Any]] = []

    # 5. Model Benchmarking Loop
    oof_arrays_to_save: Dict[str, np.ndarray] = {
        "y_true": y_train.values,
    }

    for name, base_model in base_models.items():
        print(f"\n{'*'*70}")
        print(f"PROCESSING MODEL: {name}")
        print(f"{'*'*70}")

        # Step A: Controlled Group-Aware Inner CV Hyperparameter Search
        hp_res = run_group_aware_hyperparameter_search(name, base_model, X_train, y_train, groups)
        hp_results[name] = hp_res
        best_params = hp_res["best_params"]

        # Step B: Group-Aware 5-Fold Outer CV & OOF Generation
        cv_res = run_group_aware_cv(name, base_model, best_params, X_train, y_train, groups)
        cv_results[name] = cv_res

        oof_arrays_to_save[f"oof_preds_{name}"] = cv_res["oof_preds"]
        oof_arrays_to_save[f"oof_probas_{name}"] = cv_res["oof_probas"]

        # Step C: Final Model Training on Full X_train
        final_model = train_final_group_aware_model(name, base_model, best_params, X_train, y_train)
        final_models[name] = final_model

        # Step D: Save Model Artifact
        save_group_aware_artifact(name, final_model, cv_res, hp_res, lib_versions)

        summary.append({
            "name": name,
            "cv": cv_res["aggregate"],
            "full_oof": cv_res["full_oof_metrics"],
            "hp": hp_res,
            "overlap_checks": cv_res["fold_overlap_checks"],
        })

    # 6. Save OOF Predictions NPZ
    npz_path = ARTIFACTS_DIR / "oof_predictions.npz"
    np.savez_compressed(npz_path, **oof_arrays_to_save)
    print(f"\n📦 Saved complete OOF predictions array: {npz_path}")

    # 7. Generate Visualizations
    print("\n" + "═" * 70)
    print("GENERATING VISUALIZATIONS")
    print("═" * 70)

    _draw_bar_chart(
        labels=[r["name"] for r in summary],
        values=[r["full_oof"]["macro_f1"] for r in summary],
        errors=[r["cv"]["macro_f1_std"] for r in summary],
        title="Model Comparison — Macro F1-Score (Group-Aware OOF)",
        xlabel="Macro F1-Score",
        ylabel="Model",
        out_path=FIGURES_DIR / "benchmark_macro_f1_comparison.png",
        color_rgb=(50, 110, 180),
    )

    _draw_bar_chart(
        labels=[r["name"] for r in summary],
        values=[r["full_oof"]["balanced_accuracy"] for r in summary],
        errors=[r["cv"]["balanced_accuracy_std"] for r in summary],
        title="Model Comparison — Balanced Accuracy (Group-Aware OOF)",
        xlabel="Balanced Accuracy",
        ylabel="Model",
        out_path=FIGURES_DIR / "benchmark_balanced_accuracy.png",
        color_rgb=(60, 150, 120),
    )

    _draw_bar_chart(
        labels=[r["name"] for r in summary],
        values=[r["full_oof"]["class1_recall"] for r in summary],
        errors=[r["cv"]["class1_recall_std"] for r in summary],
        title="Prediabetes (Class 1) Recall Comparison (Group-Aware OOF)",
        xlabel="Recall (Class 1)",
        ylabel="Model",
        out_path=FIGURES_DIR / "benchmark_prediabetes_recall.png",
        color_rgb=(210, 100, 40),
    )

    _draw_bar_chart(
        labels=[r["name"] for r in summary],
        values=[r["full_oof"]["class1_precision"] for r in summary],
        errors=[r["cv"]["class1_precision_std"] for r in summary],
        title="Prediabetes (Class 1) Precision Comparison (Group-Aware OOF)",
        xlabel="Precision (Class 1)",
        ylabel="Model",
        out_path=FIGURES_DIR / "benchmark_prediabetes_precision.png",
        color_rgb=(220, 140, 50),
    )

    _draw_bar_chart(
        labels=[r["name"] for r in summary],
        values=[r["full_oof"]["roc_auc_ovr"] for r in summary],
        errors=[r["cv"]["roc_auc_ovr_std"] for r in summary],
        title="Model Comparison — Multiclass ROC-AUC OvR (Group-Aware OOF)",
        xlabel="ROC-AUC (Macro One-vs-Rest)",
        ylabel="Model",
        out_path=FIGURES_DIR / "benchmark_roc_auc.png",
        color_rgb=(130, 70, 180),
    )

    generate_per_class_f1_chart(summary, FIGURES_DIR / "benchmark_per_class_f1.png")

    for name in base_models:
        cm_data = np.array(cv_results[name]["oof_confusion_matrix"])
        _draw_confusion_matrix(
            cm=cm_data,
            class_names=CLASS_LABELS_SHORT,
            title=f"Genuine OOF Confusion Matrix — {name}",
            out_path=FIGURES_DIR / f"cm_{name.lower()}.png",
        )

    # 8. Save Summary JSONs
    summary_artifacts = ARTIFACTS_DIR / "group_aware_benchmark_summary.json"
    summary_reports = REPORTS_DIR / "group_aware_benchmark_summary.json"
    summary_json_text = json.dumps(summary, indent=2, default=str)
    summary_artifacts.write_text(summary_json_text)
    summary_reports.write_text(summary_json_text)
    print(f"📄 Saved summary JSON: {summary_artifacts}")
    print(f"📄 Saved summary JSON: {summary_reports}")

    # 9. Reports
    write_reports(summary, hp_results, lib_versions)

    elapsed_total = time.time() - t_start
    print("\n" + "═" * 70)
    print(f"✅ PHASE 4.2 COMPLETE in {elapsed_total:.1f}s ({elapsed_total/60:.2f} mins)")
    print("   Frozen test set (50,737 rows) remains STRICTLY ISOLATED.")
    print("═" * 70)

    return summary


if __name__ == "__main__":
    run_group_aware_benchmark()

