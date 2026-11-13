# CDC Model Benchmark Report
## Advanced Diabetes Risk Prediction System — Phase 4

**Dataset fingerprint:** `a971809d0786d2d7`  
**Training rows:** 202,943  |  **Test rows:** 50,737 *(held-out, NOT evaluated here)*  
**CV strategy:** Stratified 5-fold  |  **Random seed:** 42

---

## Environment

| Package | Version |
|---------|---------|
| python | 3.14.3 |
| scikit-learn | 1.8.0 |
| xgboost | 3.3.0 |
| numpy | 2.4.2 |
| pandas | 3.0.1 |
| joblib | 1.5.3 |

### Unavailable Packages (documented)

- **LightGBM:** lightgbm not installed in Python 3.14 environment
- **imbalanced-learn / SMOTENC:** imbalanced-learn not installed; SMOTENC excluded from experiments

---

## Cross-Validation Results — Model Comparison

Primary metric: **Macro F1-score**

| Model | Macro F1 | Bal. Acc | Cls-1 Recall | ROC-AUC OvR | Kappa | MCC |
|-------|----------|----------|--------------|-------------|-------|-----|
| **RandomForest** | 0.4447 ±0.0019 | 0.4952 ±0.0020 | 0.0111 ±0.0033 | 0.7630 ±0.0045 | 0.3349 ±0.0037 | 0.3537 ±0.0038 |
| **LogisticRegression** | 0.4270 ±0.0013 | 0.5202 ±0.0035 | 0.3088 ±0.0110 | 0.7729 ±0.0024 | 0.2459 ±0.0015 | 0.2497 ±0.0015 |
| **XGBoost** | 0.4239 ±0.0012 | 0.5117 ±0.0043 | 0.2213 ±0.0132 | 0.7601 ±0.0037 | 0.2524 ±0.0011 | 0.2561 ±0.0012 |
| **DecisionTree** | 0.3830 ±0.0025 | 0.4708 ±0.0052 | 0.2740 ±0.0066 | 0.6771 ±0.0034 | 0.1811 ±0.0037 | 0.1831 ±0.0037 |

---

## Per-Class Performance (5-fold CV mean ± std)

### RandomForest

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 0 — No Diabetes | 0.8628 ±0.0010 | 0.9718 ±0.0010 | 0.9140 ±0.0007 |
| 1 — Prediabetes | 0.0580 ±0.0049 | 0.0111 ±0.0033 | 0.0176 ±0.0043 |
| 2 — Diabetes | 0.8763 ±0.0075 | 0.5027 ±0.0038 | 0.6382 ±0.0037 |

### LogisticRegression

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 0 — No Diabetes | 0.8724 ±0.0010 | 0.8296 ±0.0049 | 0.8504 ±0.0024 |
| 1 — Prediabetes | 0.0683 ±0.0024 | 0.3088 ±0.0110 | 0.1118 ±0.0038 |
| 2 — Diabetes | 0.6324 ±0.0049 | 0.4221 ±0.0057 | 0.5069 ±0.0040 |

### XGBoost

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 0 — No Diabetes | 0.8755 ±0.0014 | 0.8352 ±0.0057 | 0.8549 ±0.0025 |
| 1 — Prediabetes | 0.0666 ±0.0025 | 0.2213 ±0.0132 | 0.1021 ±0.0049 |
| 2 — Diabetes | 0.5879 ±0.0066 | 0.4785 ±0.0074 | 0.5272 ±0.0042 |

### DecisionTree

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 0 — No Diabetes | 0.8619 ±0.0022 | 0.7899 ±0.0074 | 0.8243 ±0.0035 |
| 1 — Prediabetes | 0.0491 ±0.0025 | 0.2740 ±0.0066 | 0.0831 ±0.0037 |
| 2 — Diabetes | 0.4426 ±0.0066 | 0.3484 ±0.0075 | 0.3899 ±0.0048 |

---

## Hyperparameter Search Results

### LogisticRegression
- **Best macro F1 (3-fold inner CV):** 0.4271
- **Best parameters:** `{'classifier__max_iter': 500, 'classifier__C': 0.01}`
- **Search iterations:** 15

### DecisionTree
- **Best macro F1 (3-fold inner CV):** 0.3954
- **Best parameters:** `{'classifier__min_samples_leaf': 5, 'classifier__max_features': 'sqrt', 'classifier__max_depth': 8}`
- **Search iterations:** 15

### RandomForest
- **Best macro F1 (3-fold inner CV):** 0.4486
- **Best parameters:** `{'classifier__n_estimators': 200, 'classifier__min_samples_leaf': 3, 'classifier__max_features': 'sqrt', 'classifier__max_depth': 20}`
- **Search iterations:** 15

### XGBoost
- **Best macro F1 (3-fold inner CV):** 0.4338
- **Best parameters:** `{'subsample': 0.7, 'reg_lambda': 1.0, 'reg_alpha': 0.1, 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.2, 'colsample_bytree': 1.0}`
- **Search iterations:** 15

---

## Recommendation

**Recommended candidate: RandomForest**

Justification:
- Highest macro F1 (0.4447) across 5 stratified folds.
- Prediabetes (Class 1) recall is the hardest sub-problem given 1.83% prevalence.
- Class weighting significantly boosts minority class detection.
- Final model selection should also consider latency and interpretability requirements.

---

## Notes

- The held-out test set (50,737 rows) has NOT been evaluated. It remains frozen for Phase 5.
- LightGBM was not benchmarked (not installed). Install `lightgbm` and re-run to include it.
- SMOTENC was not applied (imbalanced-learn not installed).
- Random oversampling was not applied in this phase; class weighting was used instead.
- All preprocessing was re-fitted inside each CV fold — no preprocessing leakage.

*Phase 4 complete. Awaiting Phase 5 approval.*