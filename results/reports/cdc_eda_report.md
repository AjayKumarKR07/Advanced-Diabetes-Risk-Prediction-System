# CDC Diabetes Health Indicators – EDA Report
**Generated:** 2026-11-13 11:53  
**Dataset:** CDC BRFSS 2015 | `diabetes_012_health_indicators_BRFSS2015.csv`

---

## 1. Dataset Quality

The dataset contains **253,680 records** and **22 columns**. There are **no missing values**, making imputation unnecessary. All features are numeric (float64), consistent with the binary/ordinal/count nature of BRFSS survey items. No invalid out-of-range values were detected. The only quality concern is **23,899 duplicate rows (9.42%)**, which are likely legitimate given the survey's categorical structure.

## 2. Feature Characteristics

### 2.1 Feature Categorisation

| Category | Features |
|----------|----------|
| **Binary (0/1)** | `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex` |
| **Ordinal / Likert-scale** | `GenHlth`, `Age`, `Education`, `Income` |
| **Count-like (0–30)** | `MentHlth`, `PhysHlth` |
| **Continuous** | `BMI` |

### 2.2 Descriptive Statistics (selected features)

| Feature | Mean | Std | Min | 25% | Median | 75% | Max |
|---------|------|-----|-----|-----|--------|-----|-----|
| `BMI` | 28.38 | 6.61 | 12 | 24 | 27 | 31 | 98 |
| `MentHlth` | 3.18 | 7.41 | 0 | 0 | 0 | 2 | 30 |
| `PhysHlth` | 4.24 | 8.72 | 0 | 0 | 0 | 3 | 30 |
| `Age` | 8.03 | 3.05 | 1 | 6 | 8 | 10 | 13 |
| `GenHlth` | 2.51 | 1.07 | 1 | 2 | 2 | 3 | 5 |
| `Education` | 5.05 | 0.99 | 1 | 4 | 5 | 6 | 6 |
| `Income` | 6.05 | 2.07 | 1 | 5 | 7 | 8 | 8 |

### 2.3 Class-wise Feature Means

| Feature | No Diabetes (0) | Prediabetes (1) | Diabetes (2) |
|---------|----------------|-----------------|---------------|
| `HighBP` | 0.371 | 0.629 | 0.753 |
| `HighChol` | 0.379 | 0.621 | 0.670 |
| `CholCheck` | 0.957 | 0.987 | 0.993 |
| `BMI` | 27.742 | 30.724 | 31.944 |
| `Smoker` | 0.430 | 0.493 | 0.518 |
| `Stroke` | 0.032 | 0.057 | 0.092 |
| `HeartDiseaseorAttack` | 0.072 | 0.143 | 0.223 |
| `PhysActivity` | 0.779 | 0.678 | 0.630 |
| `Fruits` | 0.643 | 0.602 | 0.585 |
| `Veggies` | 0.821 | 0.769 | 0.756 |
| `HvyAlcoholConsump` | 0.062 | 0.045 | 0.024 |
| `AnyHealthcare` | 0.950 | 0.945 | 0.960 |
| `NoDocbcCost` | 0.080 | 0.129 | 0.106 |
| `GenHlth` | 2.372 | 2.976 | 3.291 |
| `MentHlth` | 2.944 | 4.530 | 4.462 |
| `PhysHlth` | 3.582 | 6.348 | 7.955 |
| `DiffWalk` | 0.132 | 0.278 | 0.371 |
| `Sex` | 0.434 | 0.438 | 0.479 |
| `Age` | 7.787 | 9.083 | 9.379 |
| `Education` | 5.107 | 4.785 | 4.745 |
| `Income` | 6.209 | 5.351 | 5.210 |

### 2.4 Top Features Correlated with Target

| Rank | Feature | Pearson r (with Diabetes_012) | Direction |
|------|---------|------------------------------|-----------|
| 1 | `GenHlth` | +0.3026 | Positive ↑ |
| 2 | `HighBP` | +0.2716 | Positive ↑ |
| 3 | `BMI` | +0.2244 | Positive ↑ |
| 4 | `DiffWalk` | +0.2242 | Positive ↑ |
| 5 | `HighChol` | +0.2091 | Positive ↑ |
| 6 | `Age` | +0.1850 | Positive ↑ |
| 7 | `HeartDiseaseorAttack` | +0.1803 | Positive ↑ |
| 8 | `PhysHlth` | +0.1763 | Positive ↑ |
| 9 | `Income` | -0.1715 | Negative ↓ |
| 10 | `Education` | -0.1305 | Negative ↓ |

### 2.5 Possibly Redundant Feature Pairs (Cramér's V > 0.3)

| Feature Pair | Cramér's V | Note |
|-------------|-----------|------|
| `HighBP` ↔ `HighChol` | 0.298 | Monitor |
| `HighBP` ↔ `DiffWalk` | 0.224 | Monitor |
| `HighBP` ↔ `HeartDiseaseorAttack` | 0.209 | Monitor |
| `HighChol` ↔ `HeartDiseaseorAttack` | 0.181 | Monitor |
| `HighBP` ↔ `Stroke` | 0.130 | Monitor |

## 3. Target Distribution

| Class | Label | Count | Proportion |
|-------|-------|-------|------------|
| 0 | No Diabetes | 213,703 | 84.24% |
| 1 | Prediabetes | 4,631 | 1.83% |
| 2 | Diabetes | 35,346 | 13.93% |

> The dataset is **severely imbalanced**. Class 0 (No Diabetes) dominates with ~84% of records. Class 1 (Prediabetes) is the minority class with only 4,631 samples (~1.8%), roughly **46× less** than the majority class. Class 2 (Diabetes) is moderately represented at ~14%.

## 4. Duplicate-Record Findings

- **23,899 duplicate rows** (9.42% of total).
- **35,086 rows** are involved in at least one duplicate group.
- After deduplication: **229,781 unique rows** remain.
- Duplicates are plausible given the survey's low-cardinality integer features.
- **Action**: Retain duplicates in the full dataset. Evaluate deduplication impact in a sensitivity analysis during model training.

## 5. Class Imbalance Challenges

The imbalance creates several modelling risks:

1. **Accuracy paradox** – A naive majority-class classifier achieves ~84% accuracy trivially.
2. **Minority class suppression** – Models may rarely or never predict Class 1 (Prediabetes).
3. **Metric inflation** – Accuracy is misleading; macro-averaged F1, AUC-OVR and Cohen's κ are essential.

### Strategy Comparison

| Strategy | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Class Weights** | No data leakage, fast, works with any estimator | May not fully compensate extreme imbalance | ✅ **Use as baseline** – set `class_weight='balanced'` |
| **Random Oversampling** | Simple, no information loss | Risk of overfitting duplicated samples | ✅ Try as secondary option |
| **SMOTE** | Generates diverse synthetic samples | Assumes feature-space continuity; binary features violate this | ⚠️ Use `SMOTENC` variant for mixed binary/ordinal features |
| **ADASYN** | Adaptive focus on hard samples | Same interpolation concern as SMOTE | 🔬 Experimental only |
| **Undersampling** | Reduces training time | Loses majority-class signal | ❌ Not recommended given dataset size |

> [!IMPORTANT]
> **Any resampling must be applied only to the training split, never before train/test splitting.**
> Doing otherwise constitutes data leakage and will produce artificially optimistic evaluation metrics.

## 6. Recommended Preprocessing

1. **No imputation needed** – Zero missing values.
2. **No encoding needed** – All features are already numeric.
3. **Scaling**: StandardScaler or MinMaxScaler for distance-based models (KNN, SVM, neural nets); tree-based models (XGBoost, LightGBM, RF) do not require scaling.
4. **BMI outliers**: Values above 60 exist but represent real (morbidly obese) individuals; cap at 98th percentile only if they degrade model performance.
5. **Deduplication**: Evaluate as an optional training-only step in a sensitivity analysis.
6. **Feature selection**: Consider dropping one of `HighBP`/`HighChol`/`CholCheck` if Cramér's V confirms strong redundancy.
7. **Resampling**: Apply `class_weight='balanced'` as the default strategy; evaluate SMOTENC as an alternative.

## 7. Recommended Validation Strategy

| Step | Recommendation |
|------|----------------|
| Train/Test Split | Stratified 80/20 split (preserves class ratios) |
| Cross-Validation | Stratified 5-fold CV on training set |
| Hyperparameter Tuning | Optuna or GridSearchCV within CV folds |
| Final Evaluation | Held-out test set (never touched during training/tuning) |
| Deduplication Sensitivity | Repeat eval with and without `drop_duplicates()` on train |

## 8. Recommended Model Evaluation Metrics

| Metric | Rationale |
|--------|-----------|
| **Macro-averaged F1** | Treats all three classes equally; penalises minority class neglect |
| **Per-class Precision / Recall / F1** | Diagnose per-class performance independently |
| **AUC-ROC (OVR)** | Multi-class discriminability, robust to imbalance |
| **Cohen's κ** | Agreement above chance; accounts for class frequency |
| **Confusion Matrix** | Full picture of class-level errors |
| **MCC (Matthews)** | Single balanced metric for multi-class imbalanced problems |
| ~~Accuracy~~ | ⚠️ Misleading due to class imbalance — do not use as primary metric |

---

## Figures Generated

- **[01_target_class_distribution.png](../figures/01_target_class_distribution.png)** – Target class bar chart with counts and percentages
- **[02_class_imbalance_pie.png](../figures/02_class_imbalance_pie.png)** – Pie chart showing proportional class imbalance
- **[03_bmi_distribution_by_class.png](../figures/03_bmi_distribution_by_class.png)** – BMI histogram overlaid per class
- **[04_correlation_heatmap.png](../figures/04_correlation_heatmap.png)** – Pearson correlation heatmap of all features
- **[05_binary_feature_prevalence.png](../figures/05_binary_feature_prevalence.png)** – Binary feature prevalence (%) per class
- **[06_ordinal_distributions.png](../figures/06_ordinal_distributions.png)** – Ordinal feature distributions per class
- **[07_feature_target_correlation.png](../figures/07_feature_target_correlation.png)** – Ranked Pearson r with Diabetes_012
- **[08_mental_physical_health_days.png](../figures/08_mental_physical_health_days.png)** – MentHlth & PhysHlth day distributions per class
- **[09_duplicate_analysis.png](../figures/09_duplicate_analysis.png)** – Duplicate row analysis by class
- **[10_resampling_strategy_comparison.png](../figures/10_resampling_strategy_comparison.png)** – Conceptual comparison of resampling strategies

---

*Report auto-generated by `notebooks/cdc_eda_phase2.py` – Phase 2.*
