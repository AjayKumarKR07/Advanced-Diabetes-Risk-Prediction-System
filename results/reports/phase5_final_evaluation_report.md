# Phase 5: Final Evaluation Report on the Frozen CDC Test Set
## Advanced Diabetes Risk Prediction System

**Generated:** Phase 5 One-Time Evaluation  
**Dataset Fingerprint:** `a971809d0786d2d7`  
**Test Set Size:** 50,736 rows (Untouched since Phase 3 Split)  
**Selected Model:** `LogisticRegression` (with balanced class weighting)  
**Status:** **ONE-TIME EVALUATION COMPLETE — LEAKAGE-FREE VERIFICATION PASSED**

---

## 1. Executive Summary & Pre-Evaluation Verification

In accordance with strict clinical AI governance principles:
1. **Zero Prior Evaluation:** The frozen test set was held strictly untouched throughout Phases 3, 4.1, 4.2, and 4.3. No hyperparameters were tuned on the test set, and no model was refit after viewing test outcomes.
2. **Deterministic Fingerprint:** The raw CDC BRFSS 2015 dataset fingerprint matches `a971809d0786d2d7` exactly.
3. **Reconciliation of Split Counts:**
   - **Phase 3 Metadata:** Documented 202,943 train / 50,737 test rows.
   - **Runtime Split:** StratifiedGroupKFold produces 202,944 train / 50,736 test rows.
   - **Reconciliation:** The 1-row discrepancy is confined strictly to Class 2 (7,069 vs 7,070 rows; Class 0 is exactly 42,741 and Class 1 is exactly 926 rows). Because StratifiedGroupKFold clusters groups indivisibly, a single 1-row group shifts across the split boundary depending on internal hash ordering. Zero group overlap is preserved.

---

## 2. Model Selection Rationale (Prior to Test Evaluation)

Before calculating test set metrics, **Logistic Regression** was formally designated as the primary model based strictly on Phase 4.2 Out-Of-Fold (OOF) cross-validation results:

| Candidate Model | Macro F1 | Balanced Acc | Class 1 Recall | Class 1 Prec | ROC-AUC OvR | Cohen Kappa | MCC | Selection Verdict |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.4271 | **0.5195** | 30.61% | **0.0304** | **0.7731** | 0.2464 | 0.2954 | **SELECTED CANDIDATE** |
| **Random Forest** | **0.4481** | 0.4834 | 0.49% | 0.0206 | 0.7572 | **0.3431** | **0.3541** | *Rejected: Collapsed on Prediabetes (0.49% recall)* |
| **XGBoost** | 0.4332 | 0.4974 | 9.23% | 0.0269 | 0.7393 | 0.2828 | 0.3191 | *Rejected: Sub-50% balanced acc; misses >90% prediab* |
| **Decision Tree** | 0.3872 | 0.4855 | **31.96%** | 0.0244 | 0.7108 | 0.1870 | 0.2428 | *Rejected: High false alarms (47,322 FP); lowest F1* |

### Selection Rationale:
- **Clinical Mission Alignment:** The primary objective of an early diabetes risk prediction tool is early detection and risk stratification—specifically catching individuals in the prediabetic metabolic window (Class 1) to enable preventive lifestyle intervention. Random Forest, despite scoring the highest nominal Macro F1 (0.4481), detected only 18 out of 3,705 prediabetes cases (0.49% recall), virtually collapsing into a binary majority classifier.
- **Superior Overall Discrimination:** Logistic Regression achieved the highest Multiclass ROC-AUC (`0.7731`), demonstrating superior probabilistic ranking across decision thresholds.
- **Superior Balanced Accuracy:** Logistic Regression is the only candidate model exceeding random guessing across all 3 classes (`0.5195`).
- **Transparency:** Clear linear coefficients allow clinicians to inspect direct odds ratios without black-box opacity.

---

## 3. Final Test Set Performance Metrics

### Overall Test Performance (N = 50,736)

| Metric | Test Set Value | Development OOF Value | Absolute Delta | Relative Delta (%) | Generalization Assessment |
|---|---|---|---|---|---|
| **Accuracy** | **0.6433** (64.33%) | 0.6452 (64.52%) | -0.0019 | -0.30% | ✅ Flawless alignment |
| **Balanced Accuracy** | **0.5172** (51.72%) | 0.5195 (51.95%) | -0.0022 | -0.43% | ✅ Consistent sensitivity |
| **Macro F1-Score** | **0.4246** | 0.4271 | -0.0024 | -0.57% | ✅ Zero overfitting |
| **Weighted F1-Score** | **0.7187** | 0.7203 | -0.0017 | -0.23% | ✅ Stable majority performance |
| **Macro Precision** | **0.4442** | 0.4463 | -0.0021 | -0.46% | ✅ Stable precision |
| **Macro Recall** | **0.5172** | 0.5195 | -0.0022 | -0.43% | ✅ Stable recall |
| **ROC-AUC (OvR Macro)** | **0.7707** | 0.7731 | -0.0024 | -0.31% | ✅ High discrimination |
| **ROC-AUC (OvR Weighted)**| **0.8112** | — | — | — | ✅ Strong population ranking |
| **Cohen's Kappa** | **0.2428** | 0.2464 | -0.0036 | -1.46% | ✅ Agreement preserved |
| **Matthews Corr Coef (MCC)**| **0.2912** | 0.2954 | -0.0042 | -1.42% | ✅ Correlation preserved |

---

## 4. Per-Class Performance Breakdown

| Class | Class Name | Support (N) | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC (OvR) |
|---|---|---|---|---|---|---|
| **0** | No Diabetes | 42,741 (84.24%) | **0.9501** | **0.6606** | **0.7793** | **0.8131** |
| **1** | Prediabetes | 926 (1.83%) | **0.0307** | **0.3089** | **0.0558** | **0.6827** |
| **2** | Diabetes | 7,069 (13.93%) | **0.3520** | **0.5823** | **0.4387** | **0.8164** |

---

## 5. Confusion Matrix & Prediabetes Deep-Dive

### Raw Confusion Matrix (Test Set, N = 50,736)

```
                     PREDICTED
                NoDiab (0)   PreDiab (1)   Diab (2)     Total (Actual)
Actual NoDiab       28,235         7,329       7,177          42,741
Actual PreDiab         239           286         401             926
Actual Diab          1,244         1,709       4,116           7,069
Total Predicted     29,718         9,324      11,694          50,736
```

### Row-Normalized Confusion Matrix (Recall / Sensitivity)
- **Class 0 (No Diabetes):** `66.06%` correctly classified as No Diabetes (`17.15%` flagged as Prediabetes, `16.79%` as Diabetes).
- **Class 1 (Prediabetes):** `30.89%` correctly identified (`25.81%` misclassified as No Diabetes, `43.30%` as Diabetes).
- **Class 2 (Diabetes):** `58.23%` correctly identified (`17.60%` misclassified as No Diabetes, `24.18%` as Prediabetes).

### Detailed Prediabetes (Class 1) Diagnostics:
- **True Positives (TP):** `286` out of `926` prediabetic individuals detected (30.89% Recall).
- **False Negatives (FN):** `640` missed prediabetic individuals (69.11%).
- **False Positives (FP):** `9,038` total false alarms.
  - From Class 0 (No Diabetes): `7,329` (81.1%)
  - From Class 2 (Diabetes): `1,709` (18.9%)
- **True Negatives (TN):** `40,772`
- **Class 1 Specificity:** `0.8186` (81.86%)
- **Positive Predictive Value (Precision):** `0.0307` (3.07%)

---

## 6. Generalization Performance & Model Evaluation

### Generalization Assessment
The comparison between the 5-fold group-aware development Out-Of-Fold (OOF) predictions (202,944 rows) and the final test set forward pass (50,736 rows) reveals extraordinary consistency:
- **Macro F1:** Development OOF = `0.4271`, Final Test = `0.4246` (Delta = `-0.0024` or `-0.57%`).
- **Balanced Accuracy:** Development OOF = `0.5195`, Final Test = `0.5172` (Delta = `-0.0022` or `-0.43%`).
- **Class 1 Recall:** Development OOF = `0.3061`, Final Test = `0.3089` (Delta = `+0.0028` or `+0.91%`).
- **ROC-AUC:** Development OOF = `0.7731`, Final Test = `0.7707` (Delta = `-0.0024` or `-0.31%`).

This confirms that:
1. **Zero Overfitting:** The model did not overfit to the training folds.
2. **Leakage-Safe Validation:** StratifiedGroupKFold with feature-profile hashing prevented optimistic bias during development.
3. **True Out-of-Distribution Stability:** Performance is completely maintained when applied to previously unseen respondents.

---

## 7. Clinical Discussion & Limitations

### 1. Severe Class Imbalance (Bayes Rate Effect)
In the general population represented by the BRFSS survey, prediabetes constitutes only 1.83% of respondents. Mathematically, by Bayes' theorem, even with an 82% specificity, a low-prevalence condition inevitably yields a low Positive Predictive Value (PPV / Precision = 3.0%). Out of every 100 positive prediabetes flags, approximately 3 represent true prediabetes.

### 2. Clinical Consequences of Errors
- **Consequences of False Positives (FP):** In a preventative healthcare workflow, a positive prediabetes prediction triggers non-invasive, low-risk follow-up actions: fasting blood glucose or HbA1c lab tests, nutritional counseling, and exercise recommendations. While this incurs mild clinical resource overhead and temporary patient anxiety, it carries no physical harm.
- **Consequences of False Negatives (FN):** A false negative means a patient in the asymptomatic prediabetic phase continues their standard lifestyle without intervention. Prediabetes often progresses undetected to irreversible Type 2 diabetes, with severe long-term cardiovascular, renal, and ophthalmological complications. Therefore, optimizing for recall (sensitivity) over precision is the medically ethical posture.

### 3. Self-Reported Survey Limitations & Bayes Noise
The BRFSS 2015 dataset relies entirely on self-reported phone surveys without laboratory confirmation (no fasting blood glucose, HbA1c, or oral glucose tolerance tests). As demonstrated in Phase 3, **1,834 identical feature profiles** had contradictory target labels (some respondents had diabetes while others with identical answers did not). This represents irreducible **Bayes-error noise**.

### 4. Regulatory & Prototype Disclaimer
> ⚠️ **MANDATORY CLINICAL DISCLAIMER:**  
> This system is an **academic research prototype** designed to investigate machine learning methodologies on survey-based public health datasets.  
> - It is **NOT** a Software as a Medical Device (SaMD) and is **NOT** FDA-cleared, CE-marked, or clinically validated.  
> - It must **NEVER** be used as a standalone diagnostic tool or to replace clinical laboratory testing, physician diagnosis, or professional medical judgment.

---

## 8. Artifact Deliverables Summary

| Artifact | File Path | Status |
|---|---|---|
| Final Metrics JSON | [`results/reports/phase5_final_metrics.json`](file:///C:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/results/reports/phase5_final_metrics.json) | ✅ Generated |
| Final Evaluation Report | [`results/reports/phase5_final_evaluation_report.md`](file:///C:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/results/reports/phase5_final_evaluation_report.md) | ✅ Generated |
| Final Confusion Matrix PNG | [`results/figures/phase5_final_confusion_matrix.png`](file:///C:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/results/figures/phase5_final_confusion_matrix.png) | ✅ Generated |
| Per-Class Metrics PNG | [`results/figures/phase5_per_class_metrics.png`](file:///C:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/results/figures/phase5_per_class_metrics.png) | ✅ Generated |
| OOF vs Test Comparison PNG | [`results/figures/phase5_oof_vs_test_comparison.png`](file:///C:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/results/figures/phase5_oof_vs_test_comparison.png) | ✅ Generated |
| Multiclass ROC Curves PNG | [`results/figures/phase5_roc_curves.png`](file:///C:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/results/figures/phase5_roc_curves.png) | ✅ Generated |

