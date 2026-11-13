# CDC Diabetes Health Indicators – Data Quality Report
**Generated:** 2026-11-13 11:53  
**Dataset:** `data/raw/diabetes_012_health_indicators_BRFSS2015.csv`

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total rows | 253,680 |
| Total columns | 22 |
| Target column | `Diabetes_012` |
| Feature columns | 21 |
| Missing values | 0 |
| Duplicate rows | 23,899 (9.42%) |
| Rows in duplicate groups | 35,086 |

## 2. Missing Values

> ✅ **No missing values detected.** The dataset is fully populated.

## 3. Data Types

| Column | Dtype | Unique Values | Min | Max |
|--------|-------|--------------|-----|-----|
| `Diabetes_012` | float64 | 3 | 0.0 | 2.0 |
| `HighBP` | float64 | 2 | 0.0 | 1.0 |
| `HighChol` | float64 | 2 | 0.0 | 1.0 |
| `CholCheck` | float64 | 2 | 0.0 | 1.0 |
| `BMI` | float64 | 84 | 12.0 | 98.0 |
| `Smoker` | float64 | 2 | 0.0 | 1.0 |
| `Stroke` | float64 | 2 | 0.0 | 1.0 |
| `HeartDiseaseorAttack` | float64 | 2 | 0.0 | 1.0 |
| `PhysActivity` | float64 | 2 | 0.0 | 1.0 |
| `Fruits` | float64 | 2 | 0.0 | 1.0 |
| `Veggies` | float64 | 2 | 0.0 | 1.0 |
| `HvyAlcoholConsump` | float64 | 2 | 0.0 | 1.0 |
| `AnyHealthcare` | float64 | 2 | 0.0 | 1.0 |
| `NoDocbcCost` | float64 | 2 | 0.0 | 1.0 |
| `GenHlth` | float64 | 5 | 1.0 | 5.0 |
| `MentHlth` | float64 | 31 | 0.0 | 30.0 |
| `PhysHlth` | float64 | 31 | 0.0 | 30.0 |
| `DiffWalk` | float64 | 2 | 0.0 | 1.0 |
| `Sex` | float64 | 2 | 0.0 | 1.0 |
| `Age` | float64 | 13 | 1.0 | 13.0 |
| `Education` | float64 | 6 | 1.0 | 6.0 |
| `Income` | float64 | 8 | 1.0 | 8.0 |

## 4. Invalid Value Check

> ✅ **No invalid values detected.** All columns are within expected ranges.

## 5. Duplicate Record Analysis

**Total duplicate rows:** 23,899 (9.42% of dataset)

### Distribution of Duplicates by Class

| Class | Original | After Dedup | Duplicates Removed |
|-------|----------|-------------|--------------------|
| 0 – No Diabetes | 213,703 | 190,055 | 23,648 |
| 1 – Prediabetes | 4,631 | 4,629 | 2 |
| 2 – Diabetes | 35,346 | 35,097 | 249 |

### Interpretation of Duplicate Rows

The CDC BRFSS is a telephone survey. Each row represents one survey respondent's
self-reported health data. Duplicate rows — where every single one of the 22
column values is identical — may arise from:

1. **Legitimate coincidence**: Two individuals share identical demographic and
   health profiles (e.g., same age bracket, BMI category, binary health indicators).
   Given the survey's categorical and binary structure (most values are integers
   in small ranges), hash-collisions are statistically probable at scale.
2. **Data processing artefacts**: Re-uploads, merges, or ETL steps in the
   original Kaggle dataset may have introduced row duplication.
3. **Survey non-uniqueness by design**: BRFSS does not enforce uniqueness across
   respondents; two people can and will give identical responses.

**Recommendation:** Retain all duplicate rows in the full dataset. Apply
`drop_duplicates()` only *within the training split* as an optional preprocessing
step, never before train/test splitting. Evaluate model performance with and
without deduplication to measure impact.

## 6. Schema Verification

> ✅ All 22 expected columns present and accounted for.

## 7. Security Observations

> ⚠️ No PII detected in column names. Data appears fully anonymised per BRFSS protocol.

> ✅ No credentials, API keys, or sensitive tokens were accessed during this audit.

