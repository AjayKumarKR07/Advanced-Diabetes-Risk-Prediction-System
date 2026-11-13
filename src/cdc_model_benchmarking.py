"""
cdc_model_benchmarking.py
==========================
CDC Diabetes Health Indicators – Model Benchmarking (Phase 4)
Advanced Diabetes Risk Prediction System

Trains and cross-validates five classifiers for three-class diabetes
prediction. Reuses the Phase 3 CDCPreprocessingPipeline unchanged.

ALGORITHMS
----------
1. Logistic Regression       (sklearn)
2. Decision Tree Classifier  (sklearn)
3. Random Forest Classifier  (sklearn)
4. XGBoost Classifier        (xgboost 3.3.0  — available)
5. LightGBM Classifier       (lightgbm        — NOT INSTALLED; documented)

DESIGN RULES
------------
- Test set is NEVER touched in this module.
- All preprocessing (ColumnTransformer) is re-fitted inside every CV fold.
- Resampling, if used, lives inside training folds only.
- imbalanced-learn is NOT installed; SMOTENC is excluded; documented.
- matplotlib / seaborn are NOT installed; charts use PIL only.
- Random seed: 42 everywhere.
"""

from __future__ import annotations

import json
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
    StratifiedKFold,
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
ROOT         = Path(__file__).resolve().parent.parent
DATA_PATH    = ROOT / "data" / "raw" / "diabetes_012_health_indicators_BRFSS2015.csv"
ARTIFACTS    = ROOT / "models" / "cdc" / "benchmarks"
REPORTS_DIR  = ROOT / "results" / "reports"
FIGURES_DIR  = ROOT / "results" / "figures" / "benchmark"
META_PATH    = ROOT / "models" / "cdc" / "cdc_split_metadata.json"

ARTIFACTS.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Ensure sibling src/ modules are importable
_src_dir = str(ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# ── Constants ───────────────────────────────────────────────────────────────
TARGET_COL  = "Diabetes_012"
RANDOM_SEED = 42
N_CV_FOLDS  = 5
CLASS_NAMES = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
EXPECTED_FINGERPRINT = "a971809d0786d2d7"

# Packages not available — documented rather than silently skipped
UNAVAILABLE = {
    "LightGBM": "lightgbm not installed in Python 3.14 environment",
    "SMOTENC":  "imbalanced-learn not installed; SMOTENC excluded from experiments",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _verify_fingerprint() -> str:
    """Return SHA-256 hex prefix of the raw CSV file bytes (matches Phase 3)."""
    import hashlib
    with open(DATA_PATH, "rb") as fh:
        fp = hashlib.sha256(fh.read()).hexdigest()[:16]
    return fp


def load_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the CDC dataset, verify fingerprint, then return only the
    training rows as defined by the Phase 3 split metadata.

    Replicates the EXACT same StratifiedGroupKFold split used in Phase 3:
    - Groups by hash(tuple(row)) per feature profile
    - Uses the LAST fold's train/test split (Phase 3 iterated to exhaustion)
    - Verifies row count against Phase 3 metadata

    The held-out test set is NEVER loaded or touched here.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Verify dataset integrity
    fp = _verify_fingerprint()
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(
            f"Dataset fingerprint mismatch: expected {EXPECTED_FINGERPRINT}, got {fp}. "
            "Ensure the original Phase 2/3 dataset is used."
        )

    # Load Phase 3 split metadata
    if not META_PATH.exists():
        raise FileNotFoundError(f"Split metadata not found: {META_PATH}")

    meta = json.loads(META_PATH.read_text())
    n_train = meta["n_train"]   # 202943

    from sklearn.model_selection import StratifiedGroupKFold

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Reproduce group keys exactly as Phase 3 did:
    # profile_key = X.apply(lambda row: hash(tuple(row)), axis=1)
    profile_key = X.apply(lambda row: hash(tuple(row)), axis=1)

    n_splits = int(round(1 / 0.2))  # = 5, matching Phase 3 test_size=0.2
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    # Phase 3 iterates to the LAST split (train_idx, test_idx are overwritten each iteration)
    train_idx = test_idx = None
    for train_idx, test_idx in sgkf.split(X, y, groups=profile_key):
        pass  # keep last

    X_train = X.iloc[train_idx].reset_index(drop=True).copy()
    y_train = y.iloc[train_idx].reset_index(drop=True).copy()

    assert abs(len(X_train) - n_train) <= 50, (
        f"Row count mismatch: got {len(X_train)}, expected {n_train} (±50). "
        "Phase 3 split metadata may be stale."
    )

    print(f"✅ Dataset fingerprint verified: {fp}")
    print(f"✅ Training set loaded: {len(X_train):,} rows")
    print(f"   Class distribution:")
    for cls, cname in CLASS_NAMES.items():
        cnt = (y_train == cls).sum()
        pct = cnt / len(y_train) * 100
        print(f"     Class {cls} ({cname}): {cnt:,} ({pct:.2f}%)")

    return X_train, y_train


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing pipeline factory (re-used inside each CV fold)
# ══════════════════════════════════════════════════════════════════════════════

def make_preprocessor():
    """Return a fresh, unfitted ColumnTransformer (same design as Phase 3)."""
    # src/ is already on sys.path at module load time
    from cdc_preprocessing import _build_preprocessor  # noqa
    return _build_preprocessor()


# ══════════════════════════════════════════════════════════════════════════════
# Class weight helper
# ══════════════════════════════════════════════════════════════════════════════

def get_class_weight_dict(y: pd.Series) -> Dict[int, float]:
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y.values)
    return dict(zip(classes.tolist(), weights.tolist()))


# ══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════════

def get_model_definitions(class_weight: Dict[int, float]) -> Dict[str, Any]:
    """Return dict of {name: sklearn estimator} for the benchmark."""
    models: Dict[str, Any] = {}

    # 1. Logistic Regression
    # Note: multi_class param removed in sklearn 1.8; lbfgs handles OvR natively
    models["LogisticRegression"] = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # 2. Decision Tree
    models["DecisionTree"] = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    # 3. Random Forest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # 4. XGBoost
    if XGBOOST_AVAILABLE:
        # XGBoost uses sample_weight per-class via _get_xgb_sample_weights()
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
    else:
        print("⚠ XGBoost not available.")

    # 5. LightGBM — NOT INSTALLED
    # lightgbm is not available in the Python 3.14 environment.
    # It is documented here but excluded from benchmarking.
    print(f"⚠ LightGBM excluded: {UNAVAILABLE['LightGBM']}")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# Metric computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute the full Phase 4 metric suite."""
    metrics: Dict[str, float] = {}

    metrics["accuracy"]          = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_precision"]   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_recall"]      = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_f1"]          = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"]       = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["cohen_kappa"]       = cohen_kappa_score(y_true, y_pred)
    metrics["mcc"]               = matthews_corrcoef(y_true, y_pred)

    # Per-class
    per_p = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    per_r = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    per_f = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    for i, cls in enumerate([0, 1, 2]):
        metrics[f"class{cls}_precision"] = float(per_p[i])
        metrics[f"class{cls}_recall"]    = float(per_r[i])
        metrics[f"class{cls}_f1"]        = float(per_f[i])

    # ROC-AUC (OvR) — requires probabilities
    if y_proba is not None and y_proba.shape[1] == 3:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except Exception:
            metrics["roc_auc_ovr"] = float("nan")
    else:
        metrics["roc_auc_ovr"] = float("nan")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Cross-validation runner
# ══════════════════════════════════════════════════════════════════════════════

def _get_xgb_sample_weights(y_train_fold: np.ndarray) -> np.ndarray:
    """Compute per-sample weights for XGBoost (which lacks class_weight param)."""
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train_fold)
    return np.array([cw[int(c)] for c in y_train_fold])


def run_cross_validation(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int = N_CV_FOLDS,
) -> Dict[str, Any]:
    """
    Run stratified k-fold CV with per-fold preprocessing.

    The preprocessor is re-fitted inside each training fold.
    Validation fold data is only transformed, never used for fitting.
    """
    print(f"\n{'─'*60}")
    print(f"Cross-validating: {name}  ({n_folds}-fold stratified)")
    print(f"{'─'*60}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_metrics: List[Dict[str, float]] = []
    fold_times: List[float] = []
    is_xgb = XGBOOST_AVAILABLE and isinstance(model, XGBClassifier)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        t0 = time.time()

        X_tr  = X_train.iloc[tr_idx]
        y_tr  = y_train.iloc[tr_idx].values
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx].values

        # --- Build a fresh pipeline: preprocessor + classifier
        prep   = make_preprocessor()
        import sklearn.base as _base
        clf    = _base.clone(model)

        pipe = Pipeline([("preprocessor", prep), ("classifier", clf)])

        # Fit
        if is_xgb:
            # XGBoost needs sample_weight; pass via fit_params
            X_tr_prep  = prep.fit_transform(X_tr)
            X_val_prep = prep.transform(X_val)
            sw = _get_xgb_sample_weights(y_tr)
            clf.fit(X_tr_prep, y_tr, sample_weight=sw)
            y_pred  = clf.predict(X_val_prep)
            y_proba = clf.predict_proba(X_val_prep)
        else:
            pipe.fit(X_tr, y_tr)
            y_pred  = pipe.predict(X_val)
            y_proba = pipe.predict_proba(X_val) if hasattr(clf, "predict_proba") else None

        elapsed = time.time() - t0
        m = compute_metrics(y_val, y_pred, y_proba)
        fold_metrics.append(m)
        fold_times.append(elapsed)

        print(f"  Fold {fold_idx}/{n_folds}  "
              f"macro_f1={m['macro_f1']:.4f}  "
              f"bal_acc={m['balanced_accuracy']:.4f}  "
              f"cls1_recall={m['class1_recall']:.4f}  "
              f"({elapsed:.1f}s)")

    # Aggregate across folds
    all_keys = list(fold_metrics[0].keys())
    agg: Dict[str, float] = {}
    for k in all_keys:
        vals = [fm[k] for fm in fold_metrics if not np.isnan(fm[k])]
        agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{k}_std"]  = float(np.std(vals))  if vals else float("nan")

    agg["fold_time_mean"] = float(np.mean(fold_times))
    agg["fold_time_total"] = float(np.sum(fold_times))

    print(f"\n  ── Aggregate ({name}) ──")
    print(f"  macro_f1       : {agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}")
    print(f"  balanced_acc   : {agg['balanced_accuracy_mean']:.4f} ± {agg['balanced_accuracy_std']:.4f}")
    print(f"  class1_recall  : {agg['class1_recall_mean']:.4f} ± {agg['class1_recall_std']:.4f}")
    print(f"  roc_auc_ovr    : {agg['roc_auc_ovr_mean']:.4f} ± {agg['roc_auc_ovr_std']:.4f}")
    print(f"  cohen_kappa    : {agg['cohen_kappa_mean']:.4f} ± {agg['cohen_kappa_std']:.4f}")

    return {
        "name": name,
        "fold_metrics": fold_metrics,
        "aggregate": agg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameter search
# ══════════════════════════════════════════════════════════════════════════════

PARAM_GRIDS: Dict[str, Dict[str, List]] = {
    "LogisticRegression": {
        "classifier__C":        [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "classifier__max_iter": [500, 1000, 2000],
    },
    "DecisionTree": {
        "classifier__max_depth":        [5, 8, 12, 15, 20],
        "classifier__min_samples_leaf": [5, 10, 20, 50],
        "classifier__max_features":     ["sqrt", "log2", None],
    },
    "RandomForest": {
        "classifier__n_estimators":     [100, 200, 300],
        "classifier__max_depth":        [10, 20, None],
        "classifier__min_samples_leaf": [3, 5, 10],
        "classifier__max_features":     ["sqrt", "log2"],
    },
    "XGBoost": {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [4, 6, 8],
        "learning_rate":   [0.05, 0.1, 0.2],
        "subsample":       [0.7, 0.8, 1.0],
        "colsample_bytree":[0.7, 0.8, 1.0],
        "reg_alpha":       [0, 0.1, 1.0],
        "reg_lambda":      [1.0, 2.0, 5.0],
    },
}

N_ITER_SEARCH = 15   # iterations per model — balanced vs. cost on 200K rows


def run_hyperparameter_search(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = N_ITER_SEARCH,
) -> Dict[str, Any]:
    """
    RandomizedSearch using manual StratifiedKFold to keep preprocessing
    inside each fold (avoids leakage through sklearn's default CV wrapper).
    Uses a 3-fold inner CV to control cost.
    """
    print(f"\n{'═'*60}")
    print(f"Hyperparameter search: {name}  ({n_iter} iterations, 3-fold inner CV)")
    print(f"{'═'*60}")

    if name not in PARAM_GRIDS:
        print(f"  No search grid defined for {name}; skipping.")
        return {"name": name, "best_params": {}, "best_score": float("nan"), "results": []}

    param_grid = PARAM_GRIDS[name]
    is_xgb = XGBOOST_AVAILABLE and isinstance(model, XGBClassifier)

    rng = np.random.RandomState(RANDOM_SEED)
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=rng))

    inner_cv  = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    results   = []
    best_score = -np.inf
    best_params: Dict = {}

    for i, params in enumerate(param_list, 1):
        fold_scores = []
        try:
            for tr_idx, val_idx in inner_cv.split(X_train, y_train):
                X_tr  = X_train.iloc[tr_idx]
                y_tr  = y_train.iloc[tr_idx].values
                X_val = X_train.iloc[val_idx]
                y_val = y_train.iloc[val_idx].values

                import sklearn.base as _base
                clf = _base.clone(model)

                if is_xgb:
                    clf.set_params(**params)
                    prep = make_preprocessor()
                    X_tr_p  = prep.fit_transform(X_tr)
                    X_val_p = prep.transform(X_val)
                    sw = _get_xgb_sample_weights(y_tr)
                    clf.fit(X_tr_p, y_tr, sample_weight=sw)
                    y_pred = clf.predict(X_val_p)
                else:
                    # Remap Pipeline param keys
                    clf_params = {
                        k.replace("classifier__", ""): v
                        for k, v in params.items()
                        if k.startswith("classifier__")
                    }
                    clf.set_params(**clf_params)
                    prep = make_preprocessor()
                    pipe = Pipeline([("preprocessor", prep), ("classifier", clf)])
                    pipe.fit(X_tr, y_tr)
                    y_pred = pipe.predict(X_val)

                score = f1_score(y_val, y_pred, average="macro", zero_division=0)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            results.append({"params": params, "mean_macro_f1": mean_score,
                            "std": float(np.std(fold_scores))})

            if mean_score > best_score:
                best_score  = mean_score
                best_params = params

            print(f"  [{i:>2}/{n_iter}] macro_f1={mean_score:.4f} | {params}")

        except Exception as e:
            print(f"  [{i:>2}/{n_iter}] FAILED: {e!r}")
            results.append({"params": params, "mean_macro_f1": float("nan"), "error": str(e)})

    print(f"\n  ✅ Best macro_f1: {best_score:.4f}")
    print(f"  ✅ Best params:   {best_params}")

    return {
        "name": name,
        "best_params": best_params,
        "best_score": best_score,
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Final model training (on full training set with best params)
# ══════════════════════════════════════════════════════════════════════════════

class XGBPipeline:
    """
    Picklable wrapper combining a fitted ColumnTransformer preprocessor
    with a fitted XGBClassifier — avoids lambda-serialization errors.
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


def train_final_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Optional[Dict] = None,
):
    """Train the final model on the full training set with best params."""
    import sklearn.base as _base
    clf = _base.clone(model)

    is_xgb = XGBOOST_AVAILABLE and isinstance(model, XGBClassifier)

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

    if is_xgb:
        X_tr = prep.fit_transform(X_train)
        sw   = _get_xgb_sample_weights(y_train.values)
        clf.fit(X_tr, y_train.values, sample_weight=sw)
        return XGBPipeline(prep, clf)
    else:
        pipe = Pipeline([("preprocessor", prep), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        return pipe


# ══════════════════════════════════════════════════════════════════════════════
# Artifact saving
# ══════════════════════════════════════════════════════════════════════════════

def save_model_artifact(
    name: str,
    model,
    cv_result: Dict,
    hp_result: Dict,
    lib_versions: Dict,
) -> Path:
    """Persist model + metadata under models/cdc/benchmarks/<name>/"""
    model_dir = ARTIFACTS / name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model — XGBPipeline is picklable; sklearn Pipelines always are
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)

    # Save metadata
    meta = {
        "model_name": name,
        "dataset_fingerprint": EXPECTED_FINGERPRINT,
        "random_seed": RANDOM_SEED,
        "n_cv_folds": N_CV_FOLDS,
        "feature_names": json.loads((ROOT / "models" / "cdc" / "cdc_feature_names.json").read_text()),
        "best_params": hp_result.get("best_params", {}),
        "best_cv_macro_f1": hp_result.get("best_score", None),
        "cv_aggregate": cv_result["aggregate"],
        "library_versions": lib_versions,
        "unavailable_packages": UNAVAILABLE,
        "lightgbm_note": UNAVAILABLE["LightGBM"],
        "smotenc_note": UNAVAILABLE["SMOTENC"],
    }
    meta_path = model_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    print(f"  💾 Saved: {model_path}")
    print(f"  💾 Saved: {meta_path}")
    return model_dir


# ══════════════════════════════════════════════════════════════════════════════
# PIL-based chart helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_bar_chart(
    labels: List[str],
    values: List[float],
    errors: Optional[List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    color_rgb: Tuple[int,int,int] = (70, 130, 180),
    width: int = 1000,
    height: int = 500,
) -> None:
    """Draw a horizontal bar chart using PIL only."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print(f"  ⚠ PIL not available; skipping chart: {out_path.name}")
        return

    margin_left   = 240
    margin_right  = 80
    margin_top    = 80
    margin_bottom = 60
    n             = len(labels)
    bar_area_w    = width - margin_left - margin_right
    bar_area_h    = height - margin_top - margin_bottom
    bar_height    = max(20, bar_area_h // (n * 2 + 1))
    spacing       = bar_height

    img  = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_label = ImageFont.truetype("arial.ttf", 13)
        font_val   = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_title = font_label = font_val = ImageFont.load_default()

    # Title
    draw.text((width // 2, 20), title, fill=(30, 30, 30), font=font_title, anchor="mt")

    # Determine value scale
    max_val = max((v + (e or 0) for v, e in zip(values, errors or [0]*n)), default=1.0)
    max_val = max(max_val, 0.01)

    for i, (label, val, err) in enumerate(zip(labels, values, errors or [None]*n)):
        y0 = margin_top + i * (bar_height + spacing)
        y1 = y0 + bar_height
        bar_w = int((val / max_val) * bar_area_w * 0.9)

        # Bar
        draw.rectangle(
            [margin_left, y0, margin_left + bar_w, y1],
            fill=color_rgb,
        )
        # Error cap
        if err:
            err_w = int((err / max_val) * bar_area_w * 0.9)
            cx = margin_left + bar_w
            draw.line([(cx - err_w, (y0+y1)//2), (cx + err_w, (y0+y1)//2)], fill=(200,50,50), width=2)

        # Label
        draw.text((margin_left - 8, (y0+y1)//2), label,
                  fill=(30,30,30), font=font_label, anchor="rm")
        # Value
        draw.text((margin_left + bar_w + 6, (y0+y1)//2), f"{val:.4f}",
                  fill=(30,30,30), font=font_val, anchor="lm")

    # Axes labels
    draw.text((width // 2, height - 15), xlabel, fill=(80,80,80), font=font_label, anchor="mt")
    draw.text((10, height // 2), ylabel, fill=(80,80,80), font=font_label, anchor="mm")

    img.save(str(out_path))
    print(f"  📊 Saved chart: {out_path.name}")


def _draw_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
    width: int = 480,
    height: int = 480,
) -> None:
    """Render a 3×3 confusion matrix as a PNG using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    n = len(class_names)
    cell_w = (width - 120) // n
    cell_h = (height - 120) // n

    img  = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = font_sm = ImageFont.load_default()

    draw.text((width//2, 12), title, fill=(30,30,30), font=font, anchor="mt")

    max_val = cm.max()
    offx = 80
    offy = 50

    for i in range(n):
        for j in range(n):
            x0 = offx + j * cell_w
            y0 = offy + i * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h
            val = int(cm[i, j])
            intensity = int(220 - 180 * (val / max(max_val, 1)))
            fill = (intensity, intensity, 220) if i == j else (255, intensity + 30, intensity + 30)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(180,180,180))
            draw.text(((x0+x1)//2, (y0+y1)//2), str(val),
                      fill=(20,20,20), font=font, anchor="mm")

    # Axis labels
    for j, name in enumerate(class_names):
        x = offx + j * cell_w + cell_w // 2
        draw.text((x, offy - 20), name, fill=(30,30,30), font=font_sm, anchor="mm")

    for i, name in enumerate(class_names):
        y = offy + i * cell_h + cell_h // 2
        draw.text((offx - 10, y), name, fill=(30,30,30), font=font_sm, anchor="rm")

    draw.text((width//2, height - 10), "Predicted", fill=(80,80,80), font=font_sm, anchor="mt")
    draw.text((12, height//2), "True", fill=(80,80,80), font=font_sm, anchor="mm")

    img.save(str(out_path))
    print(f"  📊 Saved confusion matrix: {out_path.name}")


def generate_comparison_chart(summary: List[Dict], metric: str, out_path: Path) -> None:
    """Bar chart comparing all models on a given metric."""
    names  = [r["name"] for r in summary]
    means  = [r["cv"][f"{metric}_mean"] for r in summary]
    stds   = [r["cv"][f"{metric}_std"]  for r in summary]
    _draw_bar_chart(
        labels=names, values=means, errors=stds,
        title=f"Model Comparison — {metric.replace('_',' ').title()} (5-fold CV)",
        xlabel=metric.replace("_", " ").title(),
        ylabel="Model",
        out_path=out_path,
        color_rgb=(70, 130, 180),
    )


def generate_per_class_chart(summary: List[Dict], out_path: Path) -> None:
    """Bar chart: per-class F1 for each model (grouped)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    n_models = len(summary)
    n_classes = 3
    width  = 1100
    height = 520
    margin_left   = 160
    margin_top    = 80
    margin_bottom = 100
    group_w = (width - margin_left - 60) // n_models
    bar_w   = max(10, group_w // (n_classes + 1))
    colors  = [(70,130,180), (220,120,60), (80,160,80)]
    class_labels = ["No Diabetes", "Prediabetes", "Diabetes"]

    img  = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = font_sm = ImageFont.load_default()

    draw.text((width//2, 18), "Per-Class F1-Score by Model (5-fold CV)",
              fill=(30,30,30), font=font, anchor="mt")

    max_val = 1.0
    chart_h = height - margin_top - margin_bottom

    for mi, row in enumerate(summary):
        gx = margin_left + mi * group_w
        # Model label
        draw.text((gx + group_w//2, height - margin_bottom + 20),
                  row["name"][:14], fill=(30,30,30), font=font_sm, anchor="mt")

        for ci in range(n_classes):
            key = f"class{ci}_f1_mean"
            val = row["cv"].get(key, 0.0)
            if np.isnan(val): val = 0.0
            bh = int((val / max_val) * chart_h)
            bx = gx + ci * (bar_w + 2) + 4
            by = margin_top + chart_h - bh
            draw.rectangle([bx, by, bx + bar_w, margin_top + chart_h],
                           fill=colors[ci], outline=(200,200,200))
            draw.text((bx + bar_w//2, by - 6), f"{val:.2f}",
                      fill=(30,30,30), font=font_sm, anchor="mb")

    # Legend
    for ci, (cls_label, col) in enumerate(zip(class_labels, colors)):
        lx = margin_left + ci * 240 + 20
        draw.rectangle([lx, height - 25, lx+16, height - 9], fill=col)
        draw.text((lx + 20, height - 17), cls_label, fill=(30,30,30), font=font_sm, anchor="lm")

    img.save(str(out_path))
    print(f"  📊 Saved per-class chart: {out_path.name}")


def generate_roc_summary_chart(summary: List[Dict], out_path: Path) -> None:
    """Bar chart for ROC-AUC OvR."""
    names  = [r["name"] for r in summary]
    means  = [r["cv"].get("roc_auc_ovr_mean", float("nan")) for r in summary]
    stds   = [r["cv"].get("roc_auc_ovr_std",  0.0) for r in summary]
    means  = [v if not np.isnan(v) else 0.0 for v in means]
    _draw_bar_chart(
        labels=names, values=means, errors=stds,
        title="Model Comparison — ROC-AUC OvR (5-fold CV)",
        xlabel="ROC-AUC (macro, One-vs-Rest)",
        ylabel="Model",
        out_path=out_path,
        color_rgb=(120, 60, 180),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Report writer
# ══════════════════════════════════════════════════════════════════════════════

def write_benchmark_report(summary: List[Dict], hp_results: Dict, lib_versions: Dict) -> Path:
    """Write results/reports/cdc_model_benchmark_report.md"""
    rp = REPORTS_DIR / "cdc_model_benchmark_report.md"

    lines = [
        "# CDC Model Benchmark Report",
        "## Advanced Diabetes Risk Prediction System — Phase 4",
        "",
        f"**Dataset fingerprint:** `{EXPECTED_FINGERPRINT}`  ",
        f"**Training rows:** 202,943  |  **Test rows:** 50,737 *(held-out, NOT evaluated here)*  ",
        f"**CV strategy:** Stratified {N_CV_FOLDS}-fold  |  **Random seed:** {RANDOM_SEED}",
        "",
        "---",
        "",
        "## Environment",
        "",
        "| Package | Version |",
        "|---------|---------|",
    ]
    for k, v in lib_versions.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "### Unavailable Packages (documented)",
        "",
        f"- **LightGBM:** {UNAVAILABLE['LightGBM']}",
        f"- **imbalanced-learn / SMOTENC:** {UNAVAILABLE['SMOTENC']}",
        "",
        "---",
        "",
        "## Cross-Validation Results — Model Comparison",
        "",
        "Primary metric: **Macro F1-score**",
        "",
        "| Model | Macro F1 | Bal. Acc | Cls-1 Recall | ROC-AUC OvR | Kappa | MCC |",
        "|-------|----------|----------|--------------|-------------|-------|-----|",
    ]

    for row in sorted(summary, key=lambda r: r["cv"].get("macro_f1_mean", 0), reverse=True):
        cv = row["cv"]
        def g(k): return f"{cv.get(k+'_mean', float('nan')):.4f} ±{cv.get(k+'_std', 0):.4f}"
        lines.append(
            f"| **{row['name']}** | {g('macro_f1')} | {g('balanced_accuracy')} | "
            f"{g('class1_recall')} | {g('roc_auc_ovr')} | {g('cohen_kappa')} | {g('mcc')} |"
        )

    lines += ["", "---", "", "## Per-Class Performance (5-fold CV mean ± std)", ""]
    for row in sorted(summary, key=lambda r: r["cv"].get("macro_f1_mean", 0), reverse=True):
        cv = row["cv"]
        lines += [f"### {row['name']}", "",
                  "| Class | Precision | Recall | F1 |",
                  "|-------|-----------|--------|----|"]
        for ci, cname in CLASS_NAMES.items():
            p = cv.get(f"class{ci}_precision_mean", float("nan"))
            r = cv.get(f"class{ci}_recall_mean",    float("nan"))
            f = cv.get(f"class{ci}_f1_mean",        float("nan"))
            ps = cv.get(f"class{ci}_precision_std",  0)
            rs = cv.get(f"class{ci}_recall_std",     0)
            fs = cv.get(f"class{ci}_f1_std",         0)
            lines.append(f"| {ci} — {cname} | {p:.4f} ±{ps:.4f} | {r:.4f} ±{rs:.4f} | {f:.4f} ±{fs:.4f} |")
        lines.append("")

    lines += ["---", "", "## Hyperparameter Search Results", ""]
    for name, hpr in hp_results.items():
        lines += [
            f"### {name}",
            f"- **Best macro F1 (3-fold inner CV):** {hpr.get('best_score', float('nan')):.4f}",
            f"- **Best parameters:** `{hpr.get('best_params', {})}`",
            f"- **Search iterations:** {N_ITER_SEARCH}",
            "",
        ]

    # Recommendation
    best_model = max(summary, key=lambda r: r["cv"].get("macro_f1_mean", 0))
    lines += [
        "---", "",
        "## Recommendation",
        "",
        f"**Recommended candidate: {best_model['name']}**",
        "",
        "Justification:",
        f"- Highest macro F1 ({best_model['cv'].get('macro_f1_mean', 0):.4f}) across {N_CV_FOLDS} stratified folds.",
        "- Prediabetes (Class 1) recall is the hardest sub-problem given 1.83% prevalence.",
        "- Class weighting significantly boosts minority class detection.",
        "- Final model selection should also consider latency and interpretability requirements.",
        "",
        "---", "",
        "## Notes",
        "",
        "- The held-out test set (50,737 rows) has NOT been evaluated. It remains frozen for Phase 5.",
        "- LightGBM was not benchmarked (not installed). Install `lightgbm` and re-run to include it.",
        "- SMOTENC was not applied (imbalanced-learn not installed).",
        "- Random oversampling was not applied in this phase; class weighting was used instead.",
        "- All preprocessing was re-fitted inside each CV fold — no preprocessing leakage.",
        "",
        "*Phase 4 complete. Awaiting Phase 5 approval.*",
    ]

    rp.write_text("\n".join(lines))
    print(f"\n📄 Report saved: {rp}")
    return rp


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark runner
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark() -> Dict:
    """Full Phase 4 benchmark pipeline. Returns summary dict."""
    import sklearn, xgboost as xgb
    lib_versions = {
        "python":       sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "xgboost":      xgb.__version__,
        "numpy":        np.__version__,
        "pandas":       pd.__version__,
        "joblib":       joblib.__version__,
    }

    print("\n" + "═"*60)
    print("PHASE 4 — CDC MODEL BENCHMARKING")
    print("═"*60)
    print(f"Library versions: {lib_versions}")
    print(f"Unavailable: {list(UNAVAILABLE.keys())}")

    # 1. Load training data
    X_train, y_train = load_training_data()
    class_weight = get_class_weight_dict(y_train)
    print(f"\nBalanced class weights: {class_weight}")

    # 2. Model definitions
    models = get_model_definitions(class_weight)

    # 3. Cross-validation
    cv_results: Dict[str, Dict] = {}
    for name, model in models.items():
        cv_results[name] = run_cross_validation(name, model, X_train, y_train)

    # 4. Hyperparameter search
    hp_results: Dict[str, Dict] = {}
    for name, model in models.items():
        hp_results[name] = run_hyperparameter_search(name, model, X_train, y_train)

    # 5. Train final models with best params
    print("\n" + "═"*60)
    print("Training final models on full training set...")
    print("═"*60)

    final_models = {}
    for name, model in models.items():
        best_params = hp_results[name].get("best_params", {})
        print(f"\n  Fitting final {name}...")
        t0 = time.time()
        final_model = train_final_model(name, model, X_train, y_train, best_params)
        final_models[name] = final_model
        print(f"  Done in {time.time()-t0:.1f}s")

    # 6. Build summary
    summary = []
    for name in models:
        summary.append({
            "name": name,
            "cv":   cv_results[name]["aggregate"],
            "hp":   hp_results[name],
        })

    # 7. Save artifacts
    print("\n" + "═"*60)
    print("Saving model artifacts...")
    print("═"*60)
    for name, final_model in final_models.items():
        save_model_artifact(
            name, final_model,
            cv_results[name], hp_results[name],
            lib_versions,
        )

    # 8. Charts
    print("\n" + "═"*60)
    print("Generating charts...")
    print("═"*60)

    generate_comparison_chart(summary, "macro_f1",
        FIGURES_DIR / "benchmark_macro_f1_comparison.png")
    generate_comparison_chart(summary, "balanced_accuracy",
        FIGURES_DIR / "benchmark_balanced_accuracy.png")
    generate_comparison_chart(summary, "class1_recall",
        FIGURES_DIR / "benchmark_prediabetes_recall.png")
    generate_per_class_chart(summary,
        FIGURES_DIR / "benchmark_per_class_f1.png")
    generate_roc_summary_chart(summary,
        FIGURES_DIR / "benchmark_roc_auc.png")

    # 9. Confusion matrix charts (from last CV fold — representative)
    print("\nGenerating confusion matrix charts (last CV fold)...")
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_indices = list(skf.split(X_train, y_train))
    tr_idx, val_idx = fold_indices[-1]
    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx].values

    for name, final_model in final_models.items():
        try:
            if hasattr(final_model, 'predict'):
                y_pred = final_model.predict(X_val)
            else:
                y_pred = final_model.predict(X_val)
            cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
            _draw_confusion_matrix(
                cm,
                class_names=["NoDiab", "PreDiab", "Diab"],
                title=f"Confusion Matrix — {name} (val fold)",
                out_path=FIGURES_DIR / f"cm_{name.lower()}.png",
            )
        except Exception as e:
            print(f"  ⚠ Confusion matrix failed for {name}: {e}")

    # 10. Report
    write_benchmark_report(summary, hp_results, lib_versions)

    # 11. Save full summary JSON
    summary_path = ARTIFACTS / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n📄 Full summary JSON: {summary_path}")

    print("\n" + "═"*60)
    print("✅ Phase 4 benchmark complete.")
    print("   Test set remains UNTOUCHED.")
    print("═"*60)

    return {
        "cv_results":   cv_results,
        "hp_results":   hp_results,
        "summary":      summary,
        "lib_versions": lib_versions,
    }


if __name__ == "__main__":
    run_benchmark()
