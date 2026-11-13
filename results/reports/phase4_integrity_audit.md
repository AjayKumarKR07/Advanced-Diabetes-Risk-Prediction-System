# Phase 4.1 — Benchmark Integrity and Leakage Audit
**Project:** Advanced Diabetes Risk Prediction System  
**Dataset:** CDC Diabetes Health Indicators (BRFSS 2015) — 253,680 rows, 21 features, 3 classes  
**Audit scope:** Read-only verification of Phase 4 ML benchmarking (no retraining, no test-set evaluation)  
**Audit script:** `notebooks/phase4_audit.py`  
**Raw audit data:** `results/reports/phase4_audit_data.json`  
**Date:** 2026-11-13

---

## Audit 1 — Cross-Validation Strategy

| Property | Value |
|---|---|
| **Outer CV (reported scores)** | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — line 334 |
| **Inner CV (HP search)** | `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` — line 465 |
| **Group-aware (StratifiedGroupKFold)?** | ❌ No — ordinary StratifiedKFold used in both |

### Profile Overlap Within CV Folds (Measured)

| Fold | Val Size | Overlapping Profiles | Overlapping Rows | Overlap % |
|---|---|---|---|---|
| Fold 1 | 40,589 | 4,039 | 5,465 | **13.46%** |
| Fold 2 | 40,589 | 4,009 | 5,470 | **13.48%** |
| Fold 3 | 40,589 | 4,089 | 5,514 | **13.58%** |
| Fold 4 | 40,589 | 4,127 | 5,566 | **13.71%** |
| Fold 5 | 40,588 | 4,121 | 5,542 | **13.65%** |
| **Mean** | — | — | — | **13.58%** |

**Interpretation:** ~13.6% of validation rows in each fold share an identical feature profile with at least one row in that fold's training partition. This occurs because the dataset has 23,899 exact duplicate rows and 12,228 unique feature profiles appearing more than once. With ordinary StratifiedKFold, duplicate profiles are freely distributed across fold boundaries.

**VERDICT: ⚠️ WARNING — Documented Limitation**  
- The CV strategy is ordinary StratifiedKFold, not group-aware.  
- 13.58% average profile overlap within each fold means CV scores may be very slightly optimistic (the model has seen identical rows during training).  
- **This is distinct from the train/test split:** Phase 3 used `StratifiedGroupKFold` to construct the held-out test set — the frozen 50,737-row test set has **zero** profile overlap with the training set.  
- Impact on Phase 5 is expected to be minor; true generalisation will be measured against the clean frozen test set.

---

## Audit 2 — Preprocessing Leakage

| Check | Result |
|---|---|
| `make_preprocessor()` called fresh inside each fold loop | ✅ True (line ~349) |
| ColumnTransformer `.fit()` uses only fold training rows | ✅ True |
| XGBoost `sample_weight` computed from `y_tr` per fold | ✅ True (line ~360) |
| `class_weight="balanced"` → sklearn computes from fold `.fit()` labels only | ✅ True |
| Resampling (SMOTE/oversampling) applied anywhere | ❌ Not applied — imbalanced-learn not installed; class weighting used instead |
| `X_test` (frozen 50,737 rows) passed to any `.fit()` call | ✅ False — never |

**VERDICT: ✅ PASS — No preprocessing leakage detected.**  
Each cross-validation fold independently fits its own ColumnTransformer on training rows only. No global preprocessing state is reused across folds.

---

## Audit 3 — Hyperparameter Search

| Check | Result |
|---|---|
| HP inner CV type | `StratifiedKFold(n_splits=3)` — not group-aware (same limitation as Audit 1) |
| Frozen test set accessed during HP search | ✅ False — never |
| HP best scores vs. reported CV scores | Correctly separated (inner 3-fold ≠ outer 5-fold) |

**Score pipeline:**
```
Reported CV scores  → run_cross_validation()      → StratifiedKFold(5) on X_train only
HP best scores      → run_hyperparameter_search()  → StratifiedKFold(3) on X_train only
Final model trained → train_final_model()           → all of X_train (no test data)
```

**VERDICT: ⚠️ WARNING (minor) — HP search shares the StratifiedKFold limitation from Audit 1.**  
The frozen test set was never accessed. HP scores and CV scores are correctly separated. The only concern is that HP inner-CV folds also admit profile overlap (~13%), meaning selected hyperparameters may be very slightly over-tuned to the duplicate-profile structure of the training set.

---

## Audit 4 — Confusion Matrix Source

| Check | Result |
|---|---|
| CM generation code | Line 1070: `tr_idx, val_idx = fold_indices[-1]` |
| CM predictions source | Final model (trained on **all** X_train) predicting on last-fold val partition |
| Frozen test set used for CMs | ✅ False — never |

**Code path:**
```python
# src/cdc_model_benchmarking.py, lines 1068–1081
skf = StratifiedKFold(n_splits=N_CV_FOLDS, ...)
fold_indices = list(skf.split(X_train, y_train))
tr_idx, val_idx = fold_indices[-1]          # last fold's validation partition
y_val  = y_train.iloc[val_idx]
y_pred = final_model.predict(X_train.iloc[val_idx])
cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
```

**VERDICT: ⚠️ WARNING — CMs are for reference only, not true out-of-fold predictions.**  
The final model was trained on **all** of X_train (including the validation partition used for the CM). Predictions on a seen partition introduce in-sample optimism. The CMs cannot be treated as independent evaluation. The frozen test set was not used.  
**Label in reports:** "Development confusion matrix (last-fold val partition)" — not a held-out evaluation.

---

## Audit 5 — Artifact Verification

All four model artifacts were loaded and probed on 200 synthetic samples drawn from the training feature space.

| Model | Loads | Predict (3 classes) | Proba shape | Proba valid | Fingerprint | Features | Name |
|---|---|---|---|---|---|---|---|
| LogisticRegression | ✅ `Pipeline` | ✅ [0,1,2] | (200,3) | ✅ | `a971809d0786d2d7` ✅ | 21 ✅ | ✅ |
| DecisionTree | ✅ `Pipeline` | ✅ [0,1,2] | (200,3) | ✅ | `a971809d0786d2d7` ✅ | 21 ✅ | ✅ |
| RandomForest | ✅ `Pipeline` | ✅ [0,1,2] | (200,3) | ✅ | `a971809d0786d2d7` ✅ | 21 ✅ | ✅ |
| XGBoost (`XGBPipeline`) | ✅ `.preprocessor` + `.classifier` | ✅ [0,1,2] | (200,3) | ✅ | `a971809d0786d2d7` ✅ | 21 ✅ | ✅ |

All artifacts:
- Accept the 21 original raw input features (no preprocessing assumed externally)
- Produce valid three-class probability distributions (non-negative, sum to 1.0)
- Carry the correct Phase 3 dataset fingerprint `a971809d0786d2d7`

**VERDICT: ✅ PASS — All 4 artifacts verified.**

---

## Overall Assessment

| Audit | Verdict | Summary |
|---|---|---|
| 1. CV Strategy | ⚠️ WARNING | StratifiedKFold (not group-aware); 13.58% profile overlap within folds |
| 2. Preprocessing Leakage | ✅ PASS | Fresh preprocessor per fold; no test-set contamination |
| 3. HP Search | ⚠️ WARNING | Same StratifiedKFold limitation; test set never accessed |
| 4. Confusion Matrices | ⚠️ WARNING | Development-partition CMs only; slight in-sample optimism expected |
| 5. Artifact Verification | ✅ PASS | All 4 models load, predict 3 classes, valid probabilities |

### Are Phase 4 Results Trustworthy for Phase 5 Model Selection?

**YES — with the following caveats:**

1. **CV scores may be marginally optimistic** (~13.6% of validation rows share profiles seen during training). This is a dataset-structure artefact (survey data with limited binary/ordinal feature space) rather than a methodological error.
2. **Confusion matrices are reference-only** — they reflect final-model predictions on partially-seen data, not true held-out evaluation.
3. **The frozen test set (50,737 rows) is clean and uncontaminated.** Phase 3 used `StratifiedGroupKFold` to ensure zero profile overlap between train and test. Phase 4 CV/HP search never accessed the test set.
4. **Phase 5 final evaluation against the frozen test set will be the authoritative ground truth.** CV scores serve only for model selection ordering.

### Recommended Actions for Phase 5

- Evaluate all 4 final models against the frozen test set (first and only test-set access)
- Report macro F1, balanced accuracy, per-class precision/recall/F1, and ROC-AUC
- Generate true held-out confusion matrices
- Consider upgrading CV to `StratifiedGroupKFold` in any future retraining runs to eliminate the 13.6% overlap
- Document the overlap limitation in the final technical report

---

*Audit completed: 2026-11-13 | Script: `notebooks/phase4_audit.py` | Data: `results/reports/phase4_audit_data.json`*
