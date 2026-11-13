# CDC Group-Aware Model Benchmark Report
## Advanced Diabetes Risk Prediction System — Phase 4.2

**Dataset Fingerprint:** `a971809d0786d2d7`  
**Training Set:** 202,944 rows (Isolated from held-out 50,737 test rows)  
**Validation Strategy:** 5-fold `StratifiedGroupKFold` (Deterministic 64-bit murmur3 profile hash)  
**Profile Overlap:** **0.00% across all 5 folds** (Zero group leakage verified)  
**Random Seed:** 42

---

## Executive Summary

In Phase 4.1, an integrity audit detected a **13.58% feature profile overlap** across folds when using ordinary `StratifiedKFold`. Phase 4.2 successfully corrects this limitation by establishing a fully group-aware cross-validation and hyperparameter tuning framework. All duplicate feature profiles are strictly constrained within single folds. Full Out-Of-Fold (OOF) predictions were generated for all 202,944 development samples.

---

## Model Comparison — Group-Aware 5-Fold Cross-Validation

Primary metric: **Macro F1-score**  
Scores reported as: **5-Fold CV Mean ± Std** (with full concatenated OOF score in parentheses)

| Model | Macro F1 | Balanced Acc | Class 1 (Prediab) Recall | Class 1 Precision | ROC-AUC OvR | Cohen's Kappa | MCC |
|---|---|---|---|---|---|---|---|
| **RandomForest** | 0.4481 ± 0.0014 (0.4481) | 0.4834 ± 0.0014 | 0.0049 ± 0.0020 | 0.0219 ± 0.0114 | 0.7572 ± 0.0014 | 0.3431 ± 0.0029 | 0.3541 ± 0.0029 |
| **XGBoost** | 0.4332 ± 0.0021 (0.4332) | 0.4974 ± 0.0035 | 0.0923 ± 0.0078 | 0.0269 ± 0.0024 | 0.7394 ± 0.0040 | 0.2828 ± 0.0033 | 0.3191 ± 0.0036 |
| **LogisticRegression** | 0.4271 ± 0.0011 (0.4271) | 0.5195 ± 0.0044 | 0.3061 ± 0.0154 | 0.0304 ± 0.0014 | 0.7732 ± 0.0028 | 0.2464 ± 0.0027 | 0.2954 ± 0.0029 |
| **DecisionTree** | 0.3872 ± 0.0032 (0.3872) | 0.4855 ± 0.0058 | 0.3196 ± 0.0202 | 0.0244 ± 0.0010 | 0.7108 ± 0.0048 | 0.1871 ± 0.0032 | 0.2428 ± 0.0032 |

---

## Detailed Analysis: Class 1 (Prediabetes) Detection

Class 1 accounts for only 1.83% (3,705 samples) of the training population, creating severe class imbalance.

| Model | True Positives | False Positives | False Negatives | Precision | Recall | Specificity | FP from Cls 0 | FP from Cls 2 |
|---|---|---|---|---|---|---|---|---|
| **DecisionTree** | 1,184 | 47,322 | 2,521 | 0.0244 | 0.3196 | 0.7625 | 39,173 | 8,149 |
| **LogisticRegression** | 1,134 | 36,154 | 2,571 | 0.0304 | 0.3061 | 0.8185 | 29,397 | 6,757 |
| **XGBoost** | 342 | 12,372 | 3,363 | 0.0269 | 0.0923 | 0.9379 | 10,090 | 2,282 |
| **RandomForest** | 18 | 856 | 3,687 | 0.0206 | 0.0049 | 0.9957 | 696 | 160 |

---

## Per-Class Performance Summary (Full OOF)

| Model | Class 0 (No Diab) F1 | Class 1 (Prediab) F1 | Class 2 (Diabetes) F1 |
|---|---|---|---|
| **RandomForest** | 0.8741 | 0.0079 | 0.4623 |
| **XGBoost** | 0.8161 | 0.0417 | 0.4418 |
| **LogisticRegression** | 0.7802 | 0.0553 | 0.4457 |
| **DecisionTree** | 0.7153 | 0.0454 | 0.4008 |

---

## Group Overlap & Leakage Audit Verification

| Fold | Training Size | Validation Size | Group Overlap Rows | Group Overlap % | Status |
|---|---|---|---|---|---|
| Fold 1 | 162,355 | 40,589 | 0 | 0.00% | ✅ ZERO LEAKAGE |
| Fold 2 | 162,355 | 40,589 | 0 | 0.00% | ✅ ZERO LEAKAGE |
| Fold 3 | 162,355 | 40,589 | 0 | 0.00% | ✅ ZERO LEAKAGE |
| Fold 4 | 162,355 | 40,589 | 0 | 0.00% | ✅ ZERO LEAKAGE |
| Fold 5 | 162,356 | 40,588 | 0 | 0.00% | ✅ ZERO LEAKAGE |

---

## Hyperparameter Configurations (Group-Aware Inner CV)

### LogisticRegression
- **Best Inner CV Score (Macro F1):** `0.42729003695743367`
- **Optimal Parameters:** `{'classifier__max_iter': 1000, 'classifier__C': 0.01}`

### DecisionTree
- **Best Inner CV Score (Macro F1):** `0.39021377938291985`
- **Optimal Parameters:** `{'classifier__min_samples_leaf': 10, 'classifier__max_features': 'log2', 'classifier__max_depth': 12}`

### RandomForest
- **Best Inner CV Score (Macro F1):** `0.44945287135875706`
- **Optimal Parameters:** `{'classifier__n_estimators': 200, 'classifier__min_samples_leaf': 3, 'classifier__max_features': 'sqrt', 'classifier__max_depth': 20}`

### XGBoost
- **Best Inner CV Score (Macro F1):** `0.43491008951420335`
- **Optimal Parameters:** `{'subsample': 0.8, 'reg_lambda': 2.0, 'reg_alpha': 1.0, 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'colsample_bytree': 1.0}`

---

## Environment Versions

| Package | Version |
|---|---|
| Python | 3.14.3 |
| scikit-learn | 1.8.0 |
| xgboost | 3.3.0 |
| numpy | 2.4.2 |
| pandas | 3.0.1 |
| joblib | 1.5.3 |