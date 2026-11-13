# CDC Preprocessing Pipeline Report
## Advanced Diabetes Risk Prediction System — Phase 3

**Generated:** Phase 3 completion  
**Dataset:** `data/raw/diabetes_012_health_indicators_BRFSS2015.csv`  
**Dataset fingerprint (SHA-256 prefix):** `a971809d0786d2d7`  
**Python:** 3.14.3 · scikit-learn 1.8.0 · pandas 3.0.1 · numpy 2.4.2

---

## 1. Pipeline Architecture

```
CDCDataValidator
  └─ Validates schema, types, ranges, targets, duplicates, conflicts
       │
       ▼
CDCPreprocessingPipeline
  ├─ StratifiedGroupKFold split (groups = feature-profile hash)
  ├─ ColumnTransformer preprocessing (fit on train only)
  └─ Saves artifacts → models/cdc/
       │
       ▼
CDCFeatureEngineer  (optional — applied after split)
  ├─ 7 domain features (BMI category, health burden, lifestyle, etc.)
  └─ Class-weight / oversampling hooks (training folds only)
```

| Module | File | Role |
|--------|------|------|
| `CDCDataValidator` | `src/cdc_data_validation.py` | Pre-flight data quality checks |
| `CDCPreprocessingPipeline` | `src/cdc_preprocessing.py` | Leakage-free split + ColumnTransformer |
| `CDCFeatureEngineer` | `src/cdc_feature_engineering.py` | Domain features + imbalance helpers |

---

## 2. Feature Transformation Strategy

| Group | Features (count) | Transformation | Rationale |
|-------|-----------------|----------------|-----------|
| **Binary** | HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex (14) | **Passthrough** (no scaling) | Already 0/1; scaling harms tree-based models |
| **Ordinal** | GenHlth, Age, Education, Income (4) | **OrdinalEncoder** with explicit category lists | Preserves natural rank ordering |
| **Count-like** | MentHlth, PhysHlth (2) | **MinMaxScaler** [0, 1] | Bounded counts (0–30); normalization without distortion |
| **Continuous** | BMI (1) | **StandardScaler** (z-score) | Unbounded; zero-mean unit-variance for linear/distance models |

**ColumnTransformer output column order** (index positions):

| Index | Feature |
|-------|---------|
| 0–13 | Binary features (passthrough, original order) |
| 14–15 | MentHlth, PhysHlth (MinMaxScaler) |
| 16 | BMI (StandardScaler) |
| 17–20 | GenHlth, Age, Education, Income (OrdinalEncoder) |

> **Key guarantee:** The preprocessor is fitted exclusively on training data (`X_train`). `transform()` is then applied to both train and test sets. No test information leaks into fitting.

---

## 3. Duplicate Record Analysis

| Metric | Count |
|--------|-------|
| Total dataset rows | 253,680 |
| Exact duplicate rows | **23,899 (9.42%)** |
| Unique feature profiles appearing > 1 time | 12,228 |
| Feature profiles with conflicting targets | 1,834 |
| Rows involved in conflicting profiles | 6,120 |

**Decision: ALL records retained.**

Rationale:
- The BRFSS is a self-reported telephone survey. Independent respondents may legitimately share identical responses across all 21 features.
- Exact duplicates more likely represent survey coincidence than data-entry errors.
- Automatic removal would introduce selection bias and deflate the minority class (prediabetes).
- Conflicting profiles (same features, different target) represent **irreducible Bayes-error noise** inherent to self-reported health data — not a preprocessing defect.
- Deduplication, if desired, is configured to occur inside **training folds only**.

---

## 4. Data Split Strategy

| Parameter | Value |
|-----------|-------|
| Strategy | `StratifiedGroupKFold` (n_splits=5) |
| Group key | SHA-256 hash of all 21 input feature values (mod 2^63) |
| Unique feature profiles | 227,908 |
| Random seed | 42 |
| Train rows | **202,943** |
| Test rows | **50,737** |
| Test fraction | 20.0% |

**Class distribution (train / test):**

| Class | Label | Train count | Train % | Test count | Test % |
|-------|-------|-------------|---------|------------|--------|
| 0 | No Diabetes | 170,962 | 84.24% | 42,741 | 84.24% |
| 1 | Prediabetes | 3,705 | 1.83% | 926 | 1.83% |
| 2 | Diabetes | 28,276 | 13.93% | 7,070 | 13.93% |

Class proportions are **perfectly preserved** between train and test sets.

---

## 5. Data Leakage Prevention

### Split-level leakage prevention
- Identical feature profiles are grouped by hash and kept entirely in train **or** test — never split across both.
- **Leakage check result: PASSED** — zero feature profiles appear in both train and test sets.

### Preprocessing-level leakage prevention
- `ColumnTransformer` fitted on `X_train` only.
- `transform()` applied separately to `X_train` and `X_test`.
- No statistics (means, scales, ordinal encodings) are computed using test data.

### Resampling-level leakage prevention
- All imbalance-handling hooks in `CDCFeatureEngineer` are designed for use **inside cross-validation training folds only**.
- The held-out test set is **never resampled**.
- Test class distribution is kept at the original survey proportions.

### Feature engineering leakage prevention
- `CDCFeatureEngineer` is a **stateless** scikit-learn transformer (`fit()` is a no-op).
- All derived features use only row-level arithmetic — no cross-row statistics.
- Target column is excluded before any feature computation.

---

## 6. Class Imbalance Preparation

**Observed distribution:**

| Class | Count | Proportion |
|-------|-------|------------|
| 0 — No Diabetes | 213,703 | 84.24% |
| 1 — Prediabetes | 4,631 | **1.83% ← minority** |
| 2 — Diabetes | 35,346 | 13.93% |

**Computed balanced class weights (train set):**

| Class | Weight |
|-------|--------|
| 0 | 0.3957 |
| 1 | **18.2585** |
| 2 | 2.3924 |

**Three configurable strategies (implemented in `cdc_feature_engineering.py`):**

| Strategy | Function / Method | Notes |
|----------|-----------------|-------|
| **Class weighting** *(default)* | `get_class_weight_dict()` | No data modification; use `class_weight='balanced'` or the computed dict in any sklearn estimator |
| **Random oversampling** | `oversample_training_data()` | Replicates minority rows with replacement; only inside training folds |
| **SMOTE-NC** *(experimental)* | `smotenc_training_data()` | Synthetic interpolation respecting mixed binary/continuous features; requires `pip install imbalanced-learn` |

> **Recommendation:** Begin with class weighting (zero risk of overfitting from duplication). Evaluate SMOTE-NC in cross-validation only if class weighting proves insufficient for the prediabetes class. Never apply resampling to validation or test data.

---

## 7. Artifact Locations

| Artifact | Path | Description |
|----------|------|-------------|
| Preprocessor | `models/cdc/cdc_preprocessor.pkl` | Fitted `ColumnTransformer` (6,395 bytes) |
| Feature names | `models/cdc/cdc_feature_names.json` | Ordered list of 21 input features |
| Split metadata | `models/cdc/cdc_split_metadata.json` | Train/test sizes, class dists, seed, fingerprint |
| Validation JSON | `results/reports/cdc_validation_detail.json` | Full CDCDataValidator output |
| This report | `results/reports/cdc_preprocessing_report.md` | Phase 3 summary |
| EDA report | `results/reports/cdc_eda_report.md` | Phase 2 EDA findings |
| Data quality | `results/reports/cdc_data_quality_report.md` | Phase 2 quality audit |

**CDC artifacts are stored separately from production Pima artifacts** (`models/diabetes_model.pkl`, `models/diabetes_advanced_model.pkl`, `models/scaler.pkl`) and do not interfere with the existing Streamlit application.

---

## 8. Automated Test Results

**Test suite:** `tests/test_cdc_preprocessing.py`  
**Result: ✅ 64 / 64 passed** in 145.95 s (Python 3.14.3, pytest 9.0.2)

| Test group | Tests | Result |
|------------|-------|--------|
| TestDatasetLoading | 4 | ✅ All passed |
| TestDatasetValidation | 6 | ✅ All passed |
| TestTargetClassValidation | 6 | ✅ All passed |
| TestFeatureSchemaValidation | 8 | ✅ All passed |
| TestDuplicateProfileDetection | 4 | ✅ All passed |
| TestConflictingTargetDetection | 2 | ✅ All passed |
| TestLeakagePrevention | 6 | ✅ All passed |
| TestPreprocessingTransformations | 7 | ✅ All passed |
| TestFeatureOrderConsistency | 3 | ✅ All passed |
| TestReproducibility | 3 | ✅ All passed |
| TestFeatureEngineering | 11 | ✅ All passed |
| TestArtifactSaving | 2 | ✅ All passed |

---

## 9. Known Limitations

| Limitation | Detail | Mitigation |
|------------|--------|------------|
| **Conflicting profiles not resolvable** | 1,834 feature profiles map to multiple target classes — irreducible Bayes-error noise | Retained; model must learn probabilistic boundary |
| **StratifiedGroupKFold approximation** | Group-based split minimizes but cannot guarantee zero profile leakage for conflicting-label profiles | Leakage check passes on actual split; documented limitation |
| **imbalanced-learn not installed** | SMOTENC requires `pip install imbalanced-learn`; not in current Python 3.14 env | SMOTENC function is implemented but guarded with an `ImportError` check |
| **No feature selection applied** | All 21 features retained | Deliberate: feature selection is a Phase 4 / model-training concern |
| **Binary features not scaled** | Passthrough preserves 0/1 values | Appropriate for tree models; linear/distance models may benefit from scaling (configurable in Phase 4) |

---

## 10. Recommended Next Steps (Phase 4)

1. **Baseline models** — Logistic Regression, Decision Tree, Random Forest, LightGBM, XGBoost using `class_weight='balanced'`.
2. **Cross-validation** — Stratified K-Fold (k=5) on training data only; keep test set frozen.
3. **Evaluation metrics** — Prioritize macro-averaged F1, Cohen's Kappa, per-class recall (especially Class 1 — Prediabetes). Do not rely on accuracy alone due to class imbalance.
4. **Threshold tuning** — Post-training, calibrate per-class thresholds on a validation fold (not the test set).
5. **Feature importance** — Use SHAP values or permutation importance to identify low-signal features after training.
6. **SMOTE-NC experiment** — If prediabetes recall remains unacceptably low, install `imbalanced-learn` and test SMOTENC inside training folds.

---

*Phase 3 complete. No existing files modified. Awaiting Phase 4 approval.*

