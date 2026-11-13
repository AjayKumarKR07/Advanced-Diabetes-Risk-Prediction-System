# Phase 5.2: Split Reproducibility and Audit Consistency Report
## Advanced Diabetes Risk Prediction System

**Investigation Timestamp:** Phase 5.2 Completion  
**Scope:** Forensic reconciliation of Phase 3 versus Phase 4.2/Phase 5 dataset splits, hashing implementations, and evaluation provenance  
**Governance Mode:** Read-only analysis of existing artifacts and code; zero model retraining, zero split regeneration, zero test metric re-runs  
**Overall Verdict:** **PASS (All 5 Provenance & Reproducibility Checks Verified)**  

---

## Executive Summary

Phase 5.2 conducted a forensic audit to establish the exact split identity, hashing algorithm, and data provenance between Phase 3 and Phase 4.2/Phase 5. 

The investigation conclusively resolved the root cause of the 1-row metadata variation (202,943 / 50,737 vs. 202,944 / 50,736). The two splits are **two distinct 5-fold group partitions** resulting from differences in column ordering passed to Python's tuple hash. Crucially, **the Phase 4.2 models and Phase 5 final evaluation share the exact same split definition**, maintaining 100% disjoint index separation and 0.00% group leakage.

| Evaluation Category | Audit Focus | Verdict | Key Finding |
|---|---|:---:|---|
| **1. Hash Implementation** | Hashing algorithms used across workflows | **PASS** | `hash(tuple(row))` for train/test split; Murmur3 for CV |
| **2. Exact Split Provenance** | Cause of Phase 3 vs. Phase 4.2/5 divergence | **PASS** | Feature column order (`FEATURE_ORDER` vs raw CSV) |
| **3. Index Separation** | Disjointness of Phase 5 training and test sets | **PASS** | Exact 0 row overlap between 202,944 train and 50,736 test |
| **4. Group Isolation** | Leakage check across 21-feature profiles | **PASS** | Exact 0 group overlap between 182,387 train and 45,521 test |
| **5. Test Evaluation Provenance** | Integrity of Phase 5 test metrics | **PASS** | Final model evaluated on genuine out-of-sample partition |

---

## 1. Verify the Actual Hash Implementation

Forensic inspection of `src/cdc_group_aware_benchmarking.py` revealed two distinct hashing operations serving two different pipeline stages:

### A. Outer Train/Test Partition (Line 177):
```python
profile_key = X.apply(lambda row: hash(tuple(row)), axis=1)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
```
- **Hashing Function:** Python's built-in `hash()` function applied to a `tuple` of row values.
- **Data Types:** All 21 features in the CDC BRFSS dataset are numeric (`float64`).
- **Determinism:** In CPython, numeric float hashing uses `_Py_HashDouble`, which is deterministic across processes on a fixed 64-bit platform.
- **Position Dependency:** Crucially, Python's `hash(tuple)` is **strictly position-dependent**. Changing the order of features changes the resulting hash value.

### B. Inner and Outer Cross-Validation (Line 142 & Line 192):
```python
def generate_deterministic_groups(X: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(X, index=False)
```
- **Hashing Function:** pandas `pd.util.hash_pandas_object()`.
- **Algorithm:** Deterministic 64-bit Murmur3 hashing.
- **Role:** Exclusively used within `X_train` (202,944 rows) to assign groups for 5-fold outer cross-validation and 3-fold inner hyperparameter tuning.

**Category 1 Verdict:** **PASS**

---

## 2. Establish Exact Split Identity & Provenance

### The Mechanism of Split Divergence
A detailed code audit revealed why Phase 3 produced 202,943 / 50,737 rows while Phase 4.2 produced 202,944 / 50,736 rows:

1. **Phase 3 Split (`src/cdc_preprocessing.py`, line 367):**
   ```python
   X = df[FEATURE_ORDER].copy()
   profile_key = X.apply(lambda row: hash(tuple(row)), axis=1)
   ```
   `FEATURE_ORDER` placed `BMI` at the end (column index 20):  
   `['HighBP', 'HighChol', 'CholCheck', 'Smoker', ..., 'Income', 'MentHlth', 'PhysHlth', 'BMI']`

2. **Phase 4.2 & Phase 5 Split (`src/cdc_group_aware_benchmarking.py` & `src/cdc_final_evaluation.py`):**
   ```python
   X = df.drop(columns=[TARGET_COL])
   profile_key = X.apply(lambda row: hash(tuple(row)), axis=1)
   ```
   Raw CSV column order placed `BMI` at column index 3:  
   `['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', ...]`

### Impact on StratifiedGroupKFold
Because the tuple elements were ordered differently, Python's `hash(tuple)` produced completely different numerical group keys:
- $H_{\text{Phase3}}(\text{row}) \neq H_{\text{Phase4.2}}(\text{row})$ for every unique profile where feature order differed.
- In `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, group allocation depends on the sorted order and numerical values of group IDs.
- Consequently, `StratifiedGroupKFold` produced **two different 5-fold partitions** of the 253,680 rows.

### Index Comparison Table

| Metric | Phase 3 Split (`te_3`) | Phase 4.2 / Phase 5 Split (`te_4`) | Overlap / Variance |
|---|:---:|:---:|:---:|
| **Training Row Count** | 202,943 | 202,944 | +1 row |
| **Test Row Count** | 50,737 | 50,736 | -1 row |
| **Total Rows** | 253,680 | 253,680 | Exactly 253,680 |
| **Unique Feature Profiles** | 227,908 | 227,908 | Exactly 227,908 |
| **Shared Test Rows** | — | — | **10,289 rows** (~20.3%) |
| **Disjoint Test Rows** | — | — | **40,447 rows** |

> **Key Takeaway:** The 1-row difference was not a single row drifting across an otherwise identical partition. Rather, Phase 4.2 created a distinct, fully valid 80/20 group split because the input DataFrame column order differed from Phase 3.

**Category 2 Verdict:** **PASS**

---

## 3. Training / Test Index Separation

To confirm that no data leakage occurred in Phase 5:
- **Phase 4.2 Training Indices (`tr_4`):** Exactly 202,944 rows.
- **Phase 5 Test Indices (`te_4`):** Exactly 50,736 rows.
- **Index Intersection:**
  $$\text{Intersection}(\text{tr}_4, \text{te}_4) = 0 \text{ rows}$$
- **Completeness:**
  $$\text{len}(\text{tr}_4) + \text{len}(\text{te}_4) = 202,944 + 50,736 = 253,680 \text{ rows}$$

Every row in the CDC BRFSS dataset was assigned to either the training set or the test set, with zero index overlap.

**Category 3 Verdict:** **PASS**

---

## 4. Feature-Profile Group Isolation

We conducted an independent group-leakage audit between `tr_4` and `te_4` using all 21 input features:
- Unique profiles in Phase 4.2 Training Set (`tr_4`): **182,387**
- Unique profiles in Phase 5 Test Set (`te_4`): **45,521**
- Unique profiles in Total Dataset: **227,908**
- **Profile Overlap Count:**
  $$\text{Profiles}(\text{tr}_4) \cap \text{Profiles}(\text{te}_4) = \emptyset \quad (\mathbf{0 \text{ profiles}}, \mathbf{0.00\%})$$

Not a single feature profile present in the test set was seen during training. Zero group leakage was achieved.

**Category 4 Verdict:** **PASS**

---

## 5. Frozen Test Evaluation Provenance

### Model and Evaluation Integrity
1. **Model Provenance:** The `LogisticRegression` pipeline saved at `models/cdc/group_aware_benchmarks/LogisticRegression/model.pkl` was fitted strictly on `tr_4` (202,944 rows).
2. **Evaluation Provenance:** Phase 5 (`src/cdc_final_evaluation.py`) loaded this exact model and executed a single forward pass on `te_4` (50,736 rows).
3. **Metric Integrity:** The evaluation output saved in `results/reports/phase5_final_metrics.json` corresponds exactly to the forward pass on `te_4`:
   - Class 0 Support: 42,741
   - Class 1 Support: 926
   - Class 2 Support: 7,069
   - Total Support: 50,736
4. **No Test-Set Retraining:** The test set was evaluated once without threshold tuning, post-test refitting, or multi-candidate test snooping.

**Category 5 Verdict:** **PASS**

---

## Final Sign-Off & Deliverables

All five split reproducibility and data provenance criteria are verified and certified **PASS**.

The authoritative split for all Phase 4.2 benchmarks and Phase 5 evaluation is formally established as:
- **Training Set (`tr_4`):** 202,944 rows (182,387 unique feature profiles)
- **Held-Out Test Set (`te_4`):** 50,736 rows (45,521 unique feature profiles)
- **Group Overlap:** **0.00%** (Zero data leakage)

Phase 5 results are mathematically sound, fully reproducible, and rigorously isolated. Ready for Phase 6.

