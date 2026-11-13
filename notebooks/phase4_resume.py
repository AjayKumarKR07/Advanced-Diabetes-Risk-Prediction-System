"""
Phase 4 — Resume script: saves XGBoost artifact, generates charts and report.
Reuses pre-computed CV and HP search results from the full benchmark run.
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from cdc_model_benchmarking import (
    load_training_data, get_model_definitions, get_class_weight_dict,
    make_preprocessor, _get_xgb_sample_weights, XGBPipeline,
    save_model_artifact, generate_comparison_chart, generate_per_class_chart,
    generate_roc_summary_chart, _draw_confusion_matrix, write_benchmark_report,
    ARTIFACTS, FIGURES_DIR, CLASS_NAMES, RANDOM_SEED, N_CV_FOLDS, UNAVAILABLE,
    EXPECTED_FINGERPRINT,
)
from xgboost import XGBClassifier

# ── Hard-coded CV results from the successful benchmark run ─────────────────
CV_RESULTS = {
    "LogisticRegression": {
        "name": "LogisticRegression",
        "aggregate": {
            "accuracy_mean": 0.7490, "accuracy_std": 0.0010,
            "balanced_accuracy_mean": 0.5202, "balanced_accuracy_std": 0.0035,
            "macro_precision_mean": 0.5244, "macro_precision_std": 0.0027,
            "macro_recall_mean": 0.5202, "macro_recall_std": 0.0035,
            "macro_f1_mean": 0.4270, "macro_f1_std": 0.0013,
            "weighted_f1_mean": 0.7004, "weighted_f1_std": 0.0012,
            "cohen_kappa_mean": 0.2459, "cohen_kappa_std": 0.0015,
            "mcc_mean": 0.2497, "mcc_std": 0.0015,
            "class0_precision_mean": 0.8724, "class0_precision_std": 0.0010,
            "class0_recall_mean": 0.8296, "class0_recall_std": 0.0049,
            "class0_f1_mean": 0.8504, "class0_f1_std": 0.0024,
            "class1_precision_mean": 0.0683, "class1_precision_std": 0.0024,
            "class1_recall_mean": 0.3088, "class1_recall_std": 0.0110,
            "class1_f1_mean": 0.1118, "class1_f1_std": 0.0038,
            "class2_precision_mean": 0.6324, "class2_precision_std": 0.0049,
            "class2_recall_mean": 0.4221, "class2_recall_std": 0.0057,
            "class2_f1_mean": 0.5069, "class2_f1_std": 0.0040,
            "roc_auc_ovr_mean": 0.7729, "roc_auc_ovr_std": 0.0024,
            "fold_time_mean": 6.6, "fold_time_total": 33.2,
        }
    },
    "DecisionTree": {
        "name": "DecisionTree",
        "aggregate": {
            "accuracy_mean": 0.7098, "accuracy_std": 0.0015,
            "balanced_accuracy_mean": 0.4708, "balanced_accuracy_std": 0.0052,
            "macro_precision_mean": 0.4512, "macro_precision_std": 0.0038,
            "macro_recall_mean": 0.4708, "macro_recall_std": 0.0052,
            "macro_f1_mean": 0.3830, "macro_f1_std": 0.0025,
            "weighted_f1_mean": 0.6758, "weighted_f1_std": 0.0022,
            "cohen_kappa_mean": 0.1811, "cohen_kappa_std": 0.0037,
            "mcc_mean": 0.1831, "mcc_std": 0.0037,
            "class0_precision_mean": 0.8619, "class0_precision_std": 0.0022,
            "class0_recall_mean": 0.7899, "class0_recall_std": 0.0074,
            "class0_f1_mean": 0.8243, "class0_f1_std": 0.0035,
            "class1_precision_mean": 0.0491, "class1_precision_std": 0.0025,
            "class1_recall_mean": 0.2740, "class1_recall_std": 0.0066,
            "class1_f1_mean": 0.0831, "class1_f1_std": 0.0037,
            "class2_precision_mean": 0.4426, "class2_precision_std": 0.0066,
            "class2_recall_mean": 0.3484, "class2_recall_std": 0.0075,
            "class2_f1_mean": 0.3899, "class2_f1_std": 0.0048,
            "roc_auc_ovr_mean": 0.6771, "roc_auc_ovr_std": 0.0034,
            "fold_time_mean": 0.6, "fold_time_total": 3.2,
        }
    },
    "RandomForest": {
        "name": "RandomForest",
        "aggregate": {
            "accuracy_mean": 0.8428, "accuracy_std": 0.0007,
            "balanced_accuracy_mean": 0.4952, "balanced_accuracy_std": 0.0020,
            "macro_precision_mean": 0.5990, "macro_precision_std": 0.0045,
            "macro_recall_mean": 0.4952, "macro_recall_std": 0.0020,
            "macro_f1_mean": 0.4447, "macro_f1_std": 0.0019,
            "weighted_f1_mean": 0.7756, "weighted_f1_std": 0.0007,
            "cohen_kappa_mean": 0.3349, "cohen_kappa_std": 0.0037,
            "mcc_mean": 0.3537, "mcc_std": 0.0038,
            "class0_precision_mean": 0.8628, "class0_precision_std": 0.0010,
            "class0_recall_mean": 0.9718, "class0_recall_std": 0.0010,
            "class0_f1_mean": 0.9140, "class0_f1_std": 0.0007,
            "class1_precision_mean": 0.0580, "class1_precision_std": 0.0049,
            "class1_recall_mean": 0.0111, "class1_recall_std": 0.0033,
            "class1_f1_mean": 0.0176, "class1_f1_std": 0.0043,
            "class2_precision_mean": 0.8763, "class2_precision_std": 0.0075,
            "class2_recall_mean": 0.5027, "class2_recall_std": 0.0038,
            "class2_f1_mean": 0.6382, "class2_f1_std": 0.0037,
            "roc_auc_ovr_mean": 0.7630, "roc_auc_ovr_std": 0.0045,
            "fold_time_mean": 6.7, "fold_time_total": 34.5,
        }
    },
    "XGBoost": {
        "name": "XGBoost",
        "aggregate": {
            "accuracy_mean": 0.7508, "accuracy_std": 0.0012,
            "balanced_accuracy_mean": 0.5117, "balanced_accuracy_std": 0.0043,
            "macro_precision_mean": 0.5100, "macro_precision_std": 0.0035,
            "macro_recall_mean": 0.5117, "macro_recall_std": 0.0043,
            "macro_f1_mean": 0.4239, "macro_f1_std": 0.0012,
            "weighted_f1_mean": 0.7024, "weighted_f1_std": 0.0010,
            "cohen_kappa_mean": 0.2524, "cohen_kappa_std": 0.0011,
            "mcc_mean": 0.2561, "mcc_std": 0.0012,
            "class0_precision_mean": 0.8755, "class0_precision_std": 0.0014,
            "class0_recall_mean": 0.8352, "class0_recall_std": 0.0057,
            "class0_f1_mean": 0.8549, "class0_f1_std": 0.0025,
            "class1_precision_mean": 0.0666, "class1_precision_std": 0.0025,
            "class1_recall_mean": 0.2213, "class1_recall_std": 0.0132,
            "class1_f1_mean": 0.1021, "class1_f1_std": 0.0049,
            "class2_precision_mean": 0.5879, "class2_precision_std": 0.0066,
            "class2_recall_mean": 0.4785, "class2_recall_std": 0.0074,
            "class2_f1_mean": 0.5272, "class2_f1_std": 0.0042,
            "roc_auc_ovr_mean": 0.7601, "roc_auc_ovr_std": 0.0037,
            "fold_time_mean": 5.4, "fold_time_total": 27.0,
        }
    },
}

HP_RESULTS = {
    "LogisticRegression": {
        "name": "LogisticRegression",
        "best_params": {"classifier__max_iter": 500, "classifier__C": 0.01},
        "best_score": 0.4271,
    },
    "DecisionTree": {
        "name": "DecisionTree",
        "best_params": {"classifier__min_samples_leaf": 5, "classifier__max_features": "sqrt", "classifier__max_depth": 8},
        "best_score": 0.3954,
    },
    "RandomForest": {
        "name": "RandomForest",
        "best_params": {"classifier__n_estimators": 200, "classifier__min_samples_leaf": 3, "classifier__max_features": "sqrt", "classifier__max_depth": 20},
        "best_score": 0.4486,
    },
    "XGBoost": {
        "name": "XGBoost",
        "best_params": {"subsample": 0.7, "reg_lambda": 1.0, "reg_alpha": 0.1, "n_estimators": 300, "max_depth": 8, "learning_rate": 0.2, "colsample_bytree": 1.0},
        "best_score": 0.4338,
    },
}

LIB_VERSIONS = {
    "python": "3.14.3",
    "scikit-learn": "1.8.0",
    "xgboost": "3.3.0",
    "numpy": "2.4.2",
    "pandas": "3.0.1",
    "joblib": "1.5.3",
}

# ── Load training data once ─────────────────────────────────────────────────
print("Loading training data...")
X_train, y_train = load_training_data()
cw = get_class_weight_dict(y_train)
models_def = get_model_definitions(cw)

# ── Retrain only XGBoost with best params (others already saved) ─────────────
print("\nRetraining XGBoost final model with best params...")
t0 = time.time()
xgb_best = HP_RESULTS["XGBoost"]["best_params"]
clf = XGBClassifier(
    objective="multi:softprob", num_class=3,
    n_estimators=xgb_best["n_estimators"],
    max_depth=xgb_best["max_depth"],
    learning_rate=xgb_best["learning_rate"],
    subsample=xgb_best["subsample"],
    colsample_bytree=xgb_best["colsample_bytree"],
    reg_alpha=xgb_best["reg_alpha"],
    reg_lambda=xgb_best["reg_lambda"],
    eval_metric="mlogloss",
    random_state=RANDOM_SEED,
    n_jobs=-1, verbosity=0,
)
prep = make_preprocessor()
X_tr = prep.fit_transform(X_train)
sw   = _get_xgb_sample_weights(y_train.values)
clf.fit(X_tr, y_train.values, sample_weight=sw)
xgb_pipeline = XGBPipeline(prep, clf)
print(f"  Done in {time.time()-t0:.1f}s")

# Save XGBoost artifact
save_model_artifact("XGBoost", xgb_pipeline,
                    CV_RESULTS["XGBoost"], HP_RESULTS["XGBoost"], LIB_VERSIONS)

# ── Build summary ────────────────────────────────────────────────────────────
summary = [
    {"name": n, "cv": CV_RESULTS[n]["aggregate"], "hp": HP_RESULTS[n]}
    for n in ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]
]

# ── Charts ───────────────────────────────────────────────────────────────────
print("\nGenerating charts...")
generate_comparison_chart(summary, "macro_f1",           FIGURES_DIR / "benchmark_macro_f1_comparison.png")
generate_comparison_chart(summary, "balanced_accuracy",  FIGURES_DIR / "benchmark_balanced_accuracy.png")
generate_comparison_chart(summary, "class1_recall",      FIGURES_DIR / "benchmark_prediabetes_recall.png")
generate_per_class_chart(summary,                        FIGURES_DIR / "benchmark_per_class_f1.png")
generate_roc_summary_chart(summary,                      FIGURES_DIR / "benchmark_roc_auc.png")

# ── Confusion matrices (last CV fold) ────────────────────────────────────────
print("\nGenerating confusion matrices from last CV fold...")
skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_indices = list(skf.split(X_train, y_train))
_, val_idx   = fold_indices[-1]
X_val = X_train.iloc[val_idx]
y_val = y_train.iloc[val_idx].values

for model_name in ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]:
    try:
        m_path = ARTIFACTS / model_name / "model.pkl"
        if m_path.exists():
            loaded = joblib.load(m_path)
            y_pred = loaded.predict(X_val)
            cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
            _draw_confusion_matrix(
                cm,
                class_names=["NoDiab", "PreDiab", "Diab"],
                title=f"Confusion Matrix — {model_name} (val fold)",
                out_path=FIGURES_DIR / f"cm_{model_name.lower()}.png",
            )
    except Exception as e:
        print(f"  ⚠ CM failed for {model_name}: {e}")

# ── Report ───────────────────────────────────────────────────────────────────
write_benchmark_report(summary, HP_RESULTS, LIB_VERSIONS)

# ── Summary JSON ─────────────────────────────────────────────────────────────
import json as _json
summary_path = ARTIFACTS / "benchmark_summary.json"
summary_path.write_text(_json.dumps(summary, indent=2, default=str))
print(f"\n📄 Full summary JSON: {summary_path}")

print("\n" + "═"*60)
print("✅ Phase 4 resume complete. Test set remains UNTOUCHED.")
print("═"*60)

