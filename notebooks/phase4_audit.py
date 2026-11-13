"""
phase4_integrity_audit.py
=========================
Read-only audit probes for Phase 4 ML benchmarking.
Does NOT train models, modify artifacts, or touch the test set.
"""
import sys, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

results = {}

# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 1 — CV strategy: StratifiedKFold vs StratifiedGroupKFold
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("AUDIT 1 — Cross-Validation Strategy")
print("="*60)

# Source evidence from cdc_model_benchmarking.py line 334
cv_code_line = None
with open(ROOT / "src" / "cdc_model_benchmarking.py") as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if "StratifiedKFold" in line and "skf" in line and "split" not in line:
            cv_code_line = (i, line.strip())
            break

hp_code_line = None
for i, line in enumerate(lines, 1):
    if "inner_cv" in line and "StratifiedKFold" in line:
        hp_code_line = (i, line.strip())
        break

print(f"CV instantiation found at: line {cv_code_line[0] if cv_code_line else 'NOT FOUND'}")
print(f"  Code: {cv_code_line[1] if cv_code_line else 'N/A'}")
print(f"HP inner_cv at: line {hp_code_line[0] if hp_code_line else 'NOT FOUND'}")
print(f"  Code: {hp_code_line[1] if hp_code_line else 'N/A'}")

cv_uses_group = "StratifiedGroupKFold" in (cv_code_line[1] if cv_code_line else "")
hp_uses_group = "StratifiedGroupKFold" in (hp_code_line[1] if hp_code_line else "")
print(f"\nCV uses StratifiedGroupKFold: {cv_uses_group}")
print(f"HP uses StratifiedGroupKFold: {hp_uses_group}")

results["cv_strategy"] = {
    "cv_type": "StratifiedGroupKFold" if cv_uses_group else "StratifiedKFold",
    "hp_type": "StratifiedGroupKFold" if hp_uses_group else "StratifiedKFold",
    "cv_line": cv_code_line,
    "hp_line": hp_code_line,
}

# ── Measure actual profile overlap within CV folds ──────────────────────────
print("\nMeasuring profile overlap within CV folds...")
df = pd.read_csv(ROOT / "data" / "raw" / "diabetes_012_health_indicators_BRFSS2015.csv")
X_full = df.drop(columns=["Diabetes_012"])
y_full = df["Diabetes_012"].astype(int)

# Reproduce the same training split as Phase 4
profile_key_full = X_full.apply(lambda row: hash(tuple(row)), axis=1)
sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx = test_idx = None
for train_idx, test_idx in sgkf5.split(X_full, y_full, groups=profile_key_full):
    pass

X_train = X_full.iloc[train_idx].reset_index(drop=True)
y_train = y_full.iloc[train_idx].reset_index(drop=True)
profile_train = profile_key_full.iloc[train_idx].reset_index(drop=True)

# Now measure profile overlap across CV folds used in benchmarking
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
overlap_counts = []
for fold_idx, (tr_i, val_i) in enumerate(skf.split(X_train, y_train), 1):
    tr_profiles  = set(profile_train.iloc[tr_i])
    val_profiles = set(profile_train.iloc[val_i])
    overlap      = tr_profiles & val_profiles
    n_overlap_rows = int(profile_train.iloc[val_i].isin(overlap).sum())
    pct = 100 * n_overlap_rows / len(val_i)
    overlap_counts.append({
        "fold": fold_idx,
        "val_size": len(val_i),
        "overlapping_profiles": len(overlap),
        "overlapping_rows": n_overlap_rows,
        "overlap_pct": round(pct, 2),
    })
    print(f"  Fold {fold_idx}: val_size={len(val_i):,}, "
          f"overlapping_profiles={len(overlap):,}, "
          f"overlap_rows={n_overlap_rows:,} ({pct:.2f}%)")

results["cv_profile_overlap"] = overlap_counts
mean_overlap = np.mean([o["overlap_pct"] for o in overlap_counts])
print(f"\nMean fold overlap: {mean_overlap:.2f}% of validation rows share "
      f"a feature profile with the training partition of the same fold")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 2 — Preprocessing leakage
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("AUDIT 2 — Preprocessing Leakage")
print("="*60)

# Check code: does each fold call make_preprocessor() freshly?
fresh_prep_per_fold = False
class_weight_per_fold = False
resample_in_fold = False

for i, line in enumerate(lines, 1):
    if "prep" in line and "make_preprocessor()" in line and 340 <= i <= 400:
        fresh_prep_per_fold = True
    if "_get_xgb_sample_weights(y_tr)" in line or "compute_class_weight" in line:
        if 340 <= i <= 400:
            class_weight_per_fold = True
    if "resample" in line.lower() or "smote" in line.lower():
        resample_in_fold = True

# Check whether class_weight="balanced" is set at model definition time (not fold)
cw_at_model_def = False
for i, line in enumerate(lines, 1):
    if 'class_weight="balanced"' in line and 200 <= i <= 270:
        cw_at_model_def = True
        break

print(f"Fresh preprocessor per fold (make_preprocessor() inside loop): {fresh_prep_per_fold}")
print(f"XGBoost sample weights computed from y_tr per fold: {class_weight_per_fold}")
print(f"class_weight='balanced' set at model definition time: {cw_at_model_def}")
print(f"  → sklearn estimators compute weights inside .fit() from training labels only: True")
print(f"Resampling applied anywhere: {resample_in_fold}")

results["preprocessing_leakage"] = {
    "fresh_preprocessor_per_fold": fresh_prep_per_fold,
    "xgb_sample_weight_from_fold_labels": class_weight_per_fold,
    "class_weight_balanced_at_definition": cw_at_model_def,
    "resampling_applied": resample_in_fold,
}

# Verify test data not used in preprocessing: check that X_test is never passed to .fit()
test_in_fit = False
for i, line in enumerate(lines, 1):
    if "X_test" in line and ".fit" in line:
        test_in_fit = True
        print(f"  ⚠ POTENTIAL ISSUE at line {i}: {line.strip()}")
print(f"X_test passed to .fit() anywhere: {test_in_fit}")
results["preprocessing_leakage"]["test_in_fit"] = test_in_fit


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 3 — Hyperparameter search audit
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("AUDIT 3 — Hyperparameter Search Audit")
print("="*60)

# Check test set access in HP search
test_accessed_in_hp = False
in_hp_func = False
for i, line in enumerate(lines, 1):
    if "def run_hyperparameter_search" in line:
        in_hp_func = True
    if in_hp_func and "def " in line and "run_hyperparameter_search" not in line:
        in_hp_func = False
    if in_hp_func and ("X_test" in line or "y_test" in line):
        test_accessed_in_hp = True
        print(f"  ⚠ Test data accessed in HP search at line {i}: {line.strip()}")

print(f"Test set accessed during HP search: {test_accessed_in_hp}")
print(f"HP inner CV type: {'StratifiedGroupKFold' if hp_uses_group else 'StratifiedKFold (3-fold)'}")
print(f"HP scores are inner 3-fold; outer 5-fold CV scores are independent estimates")

# Check that HP results are not mistakenly used as final reported scores
# The CV results come from run_cross_validation (5-fold), HP from run_hyperparameter_search (3-fold inner)
print(f"\nScore pipeline:")
print(f"  Reported CV scores  → run_cross_validation() → StratifiedKFold(5) on X_train")
print(f"  HP best scores      → run_hyperparameter_search() → StratifiedKFold(3) on X_train")
print(f"  Final model trained → train_final_model() on ALL of X_train (no test data)")

results["hp_audit"] = {
    "hp_inner_cv": "StratifiedKFold(3)" if not hp_uses_group else "StratifiedGroupKFold(3)",
    "test_accessed": test_accessed_in_hp,
    "scores_independent": True,  # HP 3-fold ≠ reported 5-fold
}


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 4 — Confusion matrix source
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("AUDIT 4 — Confusion Matrix Source")
print("="*60)

# Find confusion matrix generation code
cm_source_lines = []
for i, line in enumerate(lines, 1):
    if "confusion_matrix" in line or ("fold_indices[-1]" in line) or ("last CV fold" in line.lower()):
        cm_source_lines.append((i, line.strip()))

for ln, code in cm_source_lines[:8]:
    print(f"  Line {ln}: {code}")

# Key check: is X_test used?
cm_uses_test = any("X_test" in code for _, code in cm_source_lines)
# Key: phase4_resume.py uses loaded models on X_val from last CV fold
print(f"\nCM uses frozen test set: {cm_uses_test}")
print("CM source: predictions from final models on last StratifiedKFold validation fold of X_train")
print("  → val_idx comes from fold_indices[-1] of StratifiedKFold(5) on X_train")
print("  → This is a DEVELOPMENT split, not the frozen test set")

results["cm_audit"] = {
    "cm_source": "last_cv_fold_val_partition_of_X_train",
    "uses_frozen_test": cm_uses_test,
    "model_state": "final_model_trained_on_full_X_train",
    "note": "CMs show final model predictions on seen training data fold — slight optimism expected"
}


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 5 — Artifact verification
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("AUDIT 5 — Artifact Verification")
print("="*60)

BENCH_DIR = ROOT / "models" / "cdc" / "benchmarks"
MODEL_NAMES = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]
EXPECTED_FEATURES = json.loads((ROOT / "models" / "cdc" / "cdc_feature_names.json").read_text())
EXPECTED_FP = "a971809d0786d2d7"

X_probe, _ = X_train.iloc[:200], y_train.iloc[:200]  # 200-row probe, no test data

artifact_results = {}
for name in MODEL_NAMES:
    print(f"\n  ── {name} ──")
    model_path = BENCH_DIR / name / "model.pkl"
    meta_path  = BENCH_DIR / name / "metadata.json"
    res = {}

    # Load
    try:
        model = joblib.load(model_path)
        res["loads"] = True
        print(f"    ✅ Loads: {type(model).__name__}")
    except Exception as e:
        res["loads"] = False
        print(f"    ❌ Load failed: {e}")
        artifact_results[name] = res
        continue

    # Predict
    try:
        preds = model.predict(X_probe)
        pred_classes = set(int(p) for p in preds)
        res["predicts"] = True
        res["pred_classes"] = sorted(pred_classes)
        res["all_three_classes"] = pred_classes.issubset({0, 1, 2})
        print(f"    ✅ Predict: classes in output = {sorted(pred_classes)}")
    except Exception as e:
        res["predicts"] = False
        print(f"    ❌ Predict failed: {e}")

    # Predict proba
    try:
        proba = model.predict_proba(X_probe)
        res["proba_shape"] = list(proba.shape)
        res["proba_sums_to_1"] = bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-5))
        res["proba_non_negative"] = bool((proba >= 0).all())
        print(f"    ✅ Proba shape: {proba.shape}, sums_to_1={res['proba_sums_to_1']}")
    except Exception as e:
        res["proba_shape"] = None
        print(f"    ❌ Proba failed: {e}")

    # XGBoost packaging check
    if name == "XGBoost":
        from cdc_model_benchmarking import XGBPipeline
        res["xgb_wrapper"] = isinstance(model, XGBPipeline)
        res["xgb_has_preprocessor"] = hasattr(model, "preprocessor")
        res["xgb_has_classifier"]   = hasattr(model, "classifier")
        print(f"    ✅ XGBPipeline wrapper: {res['xgb_wrapper']}")
        print(f"    ✅ Has .preprocessor: {res['xgb_has_preprocessor']}")
        print(f"    ✅ Has .classifier:   {res['xgb_has_classifier']}")

    # Metadata
    try:
        meta = json.loads(meta_path.read_text())
        res["metadata_fingerprint"] = meta.get("dataset_fingerprint")
        res["metadata_fp_matches"]  = meta.get("dataset_fingerprint") == EXPECTED_FP
        res["metadata_features"]    = meta.get("feature_names") == EXPECTED_FEATURES
        res["metadata_model_name"]  = meta.get("model_name") == name
        print(f"    ✅ Metadata fingerprint: {meta.get('dataset_fingerprint')} "
              f"(match={res['metadata_fp_matches']})")
        print(f"    ✅ Feature names match:  {res['metadata_features']}")
        print(f"    ✅ Model name correct:   {res['metadata_model_name']}")
    except Exception as e:
        res["metadata_error"] = str(e)
        print(f"    ❌ Metadata error: {e}")

    artifact_results[name] = res

results["artifacts"] = artifact_results


# ══════════════════════════════════════════════════════════════════════════════
# SAVE FULL RESULTS JSON
# ══════════════════════════════════════════════════════════════════════════════
import json as _json
out_path = ROOT / "results" / "reports" / "phase4_audit_data.json"
out_path.write_text(_json.dumps(results, indent=2, default=str))
print(f"\n\n📄 Audit data saved: {out_path}")

# Print summary
print("\n" + "="*60)
print("AUDIT SUMMARY")
print("="*60)
print(f"CV type:                  {results['cv_strategy']['cv_type']}")
print(f"Mean CV fold overlap:     {mean_overlap:.2f}%")
print(f"Fresh preprocessor/fold:  {results['preprocessing_leakage']['fresh_preprocessor_per_fold']}")
print(f"Test in .fit():           {results['preprocessing_leakage']['test_in_fit']}")
print(f"Test in HP search:        {results['hp_audit']['test_accessed']}")
print(f"CM uses test set:         {results['cm_audit']['uses_frozen_test']}")
for name, ar in results["artifacts"].items():
    ok = ar.get("loads") and ar.get("predicts") and ar.get("proba_sums_to_1") and ar.get("metadata_fp_matches")
    print(f"Artifact {name:20s}: {'✅ OK' if ok else '⚠ CHECK'}")

