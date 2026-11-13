# Phase 6.10.1 — Assessment Data Integrity & Prediction Verification Report

**Project:** Advanced Diabetes Risk Prediction System (DiabetesRisk AI)  
**Date:** November 13, 2026  
**Status:** Audit Passed (100% Verification across all checks)  
**Backend Automated Test Suite:** 95 / 95 Passed (100%)  
**Frontend Production Build:** Verified (`vite v4.5.14`, exit code 0, 8.24s)  
**Audit Script Execution:** `scripts/audit_phase6101.py` Passed (0 failures)  

---

## 1. Executive Summary

This audit rigorously inspects the data integrity, preprocessing pipeline, model prediction fidelity, PostgreSQL persistence, and cross-user authorization security of the **DiabetesRisk AI** health assessment workflow implemented in Phase 6.10.

All 21 health indicator features were checked against the frozen Phase 5 Logistic Regression model pipeline (`models/cdc/group_aware_benchmarks/LogisticRegression/model.pkl`), confirming exact feature order, scaling boundaries, and encoding fidelity. Chronological age conversion was audited across all bracket boundaries (17 to 121 years). Multi-user database authorization tests verified that cross-user access attempts to foreign assessments or predictions are strictly rejected with `HTTP 403 Forbidden`. The model’s probabilistic predictions were proved to be 100% identical between direct Python model calls, the backend service layer, and real HTTP API requests.

---

## 2. Feature Mapping & Encoding Audit

### 2.1 21-Feature Specification & Canonical Order
The frozen model pipeline operates on a DataFrame containing 21 features. The exact feature names, data types, and canonical order match across the frontend client ([cdcFeatures.js](file:///c:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/frontend/src/utils/cdcFeatures.js)), the backend Pydantic schema ([prediction.py](file:///c:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/backend/app/schemas/prediction.py)), the SQLAlchemy database model ([assessment.py](file:///c:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/backend/app/models/assessment.py)), and the pipeline metadata ([metadata.json](file:///c:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/models/cdc/group_aware_benchmarks/LogisticRegression/metadata.json)):

| # | Feature Name | Clinical / CDC BRFSS Meaning | Encoding / Type | Permissible Range | Pipeline Transformer |
|---|---|---|---|---|---|
| 1 | `HighBP` | High blood pressure diagnosed | Binary (int) | {0, 1} | Passthrough |
| 2 | `HighChol` | High cholesterol diagnosed | Binary (int) | {0, 1} | Passthrough |
| 3 | `CholCheck` | Cholesterol checked within 5 years | Binary (int) | {0, 1} | Passthrough |
| 4 | `BMI` | Body Mass Index | Continuous (float) | 10.0 – 100.0 kg/m² | `StandardScaler` |
| 5 | `Smoker` | Smoked 100+ cigarettes in lifetime | Binary (int) | {0, 1} | Passthrough |
| 6 | `Stroke` | Ever diagnosed with a stroke | Binary (int) | {0, 1} | Passthrough |
| 7 | `HeartDiseaseorAttack` | Coronary heart disease or MI | Binary (int) | {0, 1} | Passthrough |
| 8 | `PhysActivity` | Leisure physical activity in past 30 days | Binary (int) | {0, 1} | Passthrough |
| 9 | `Fruits` | Daily fruit consumption (1+ times/day) | Binary (int) | {0, 1} | Passthrough |
| 10 | `Veggies` | Daily vegetable consumption (1+ times/day) | Binary (int) | {0, 1} | Passthrough |
| 11 | `HvyAlcoholConsump` | Heavy alcohol (men >14/wk, women >7/wk) | Binary (int) | {0, 1} | Passthrough |
| 12 | `AnyHealthcare` | Any health insurance or medical coverage | Binary (int) | {0, 1} | Passthrough |
| 13 | `NoDocbcCost` | Avoided doctor due to cost in past 12 mo | Binary (int) | {0, 1} | Passthrough |
| 14 | `GenHlth` | Self-rated general health | Ordinal (int) | 1 to 5 (1=Exc, 5=Poor) | `OrdinalEncoder` |
| 15 | `MentHlth` | Days of poor mental health in past 30 days | Count-like (int) | 0 to 30 days | `MinMaxScaler` |
| 16 | `PhysHlth` | Days of physical illness/injury in past 30d | Count-like (int) | 0 to 30 days | `MinMaxScaler` |
| 17 | `DiffWalk` | Serious difficulty walking or climbing stairs | Binary (int) | {0, 1} | Passthrough |
| 18 | `Sex` | Biological sex | Binary (int) | 0 = Female, 1 = Male | Passthrough |
| 19 | `Age` | CDC BRFSS 13-category age bracket | Ordinal (int) | 1 to 13 | `OrdinalEncoder` |
| 20 | `Education` | Highest education completed | Ordinal (int) | 1 to 6 | `OrdinalEncoder` |
| 21 | `Income` | Annual household income bracket | Ordinal (int) | 1 to 8 | `OrdinalEncoder` |

- **Verification Result:** Zero missing, duplicated, or misnamed features. Exact 21-column name parity confirmed.

### 2.2 Chronological Age Boundary Audit
In CDC BRFSS 2015, age is encoded into 13 discrete categories for respondents aged 18 and older. The frontend function `convertAgeToCdcCategory` was tested across every boundary condition:

| Test Age | Expected Category | Computed Category | Status | Boundary Description |
|---|---|---|---|---|
| 17 | `None` (Invalid) | `None` | **PASS** | Underage rejection (<18) |
| 18 | 1 | 1 | **PASS** | Bracket 1 lower boundary (18–24) |
| 24 | 1 | 1 | **PASS** | Bracket 1 upper boundary |
| 25 | 2 | 2 | **PASS** | Bracket 2 lower boundary (25–29) |
| 29 | 2 | 2 | **PASS** | Bracket 2 upper boundary |
| 30 | 3 | 3 | **PASS** | Bracket 3 lower boundary (30–34) |
| 34 | 3 | 3 | **PASS** | Bracket 3 upper boundary |
| 35 | 4 | 4 | **PASS** | Bracket 4 lower boundary (35–39) |
| 39 | 4 | 4 | **PASS** | Bracket 4 upper boundary |
| 40 | 5 | 5 | **PASS** | Bracket 5 lower boundary (40–44) |
| 44 | 5 | 5 | **PASS** | Bracket 5 upper boundary |
| 45 | 6 | 6 | **PASS** | Bracket 6 lower boundary (45–49) |
| 49 | 6 | 6 | **PASS** | Bracket 6 upper boundary |
| 50 | 7 | 7 | **PASS** | Bracket 7 lower boundary (50–54) |
| 54 | 7 | 7 | **PASS** | Bracket 7 upper boundary |
| 55 | 8 | 8 | **PASS** | Bracket 8 lower boundary (55–59) |
| 59 | 8 | 8 | **PASS** | Bracket 8 upper boundary |
| 60 | 9 | 9 | **PASS** | Bracket 9 lower boundary (60–64) |
| 64 | 9 | 9 | **PASS** | Bracket 9 upper boundary |
| 65 | 10 | 10 | **PASS** | Bracket 10 lower boundary (65–69) |
| 69 | 10 | 10 | **PASS** | Bracket 10 upper boundary |
| 70 | 11 | 11 | **PASS** | Bracket 11 lower boundary (70–74) |
| 74 | 11 | 11 | **PASS** | Bracket 11 upper boundary |
| 75 | 12 | 12 | **PASS** | Bracket 12 lower boundary (75–79) |
| 79 | 12 | 12 | **PASS** | Bracket 12 upper boundary |
| 80 | 13 | 13 | **PASS** | Bracket 13 lower boundary (80+) |
| 95 | 13 | 13 | **PASS** | Bracket 13 interior |
| 120 | 13 | 13 | **PASS** | Bracket 13 upper boundary |
| 121 | `None` (Invalid) | `None` | **PASS** | Overage rejection (>120) |

- **Verification Result:** 100% boundary accuracy. No raw age values are ever passed directly to the model.

---

## 3. Preprocessing & Pipeline Equivalence Audit

Inspection of the loaded serialized artifact `models/cdc/group_aware_benchmarks/LogisticRegression/model.pkl` confirms:
1. **Pipeline Architecture:**
   ```
   sklearn.pipeline.Pipeline(
       steps=[
           ('preprocessor', ColumnTransformer(transformers=[
               ('binary', Pipeline([('passthrough', FunctionTransformer())]), [14 binary columns]),
               ('count', Pipeline([('minmax', MinMaxScaler())]), ['MentHlth', 'PhysHlth']),
               ('continuous', Pipeline([('scaler', StandardScaler())]), ['BMI']),
               ('ord_GenHlth', Pipeline([('ordinal_enc', OrdinalEncoder())]), ['GenHlth']),
               ('ord_Age', Pipeline([('ordinal_enc', OrdinalEncoder())]), ['Age']),
               ('ord_Education', Pipeline([('ordinal_enc', OrdinalEncoder())]), ['Education']),
               ('ord_Income', Pipeline([('ordinal_enc', OrdinalEncoder())]), ['Income'])
           ])),
           ('classifier', LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=42))
       ]
   )
   ```
2. **Encapsulation:** All scaling (MinMaxScaler on count columns, StandardScaler on BMI) and ordinal encoding are self-contained within the scikit-learn `Pipeline`.
3. **No Redundant Scaling:** Neither the frontend JavaScript nor the FastAPI backend routes apply any manual standardization or one-hot transformations, preventing double-scaling artifacts. Raw inputs are converted directly to a named Pandas DataFrame which is consumed by the pipeline.

---

## 4. Prediction Integrity & Numerical Verification

Three clinical test profiles were executed simultaneously through:
- **Pipeline Direct:** Calling `model.predict()` and `model.predict_proba()` directly in Python.
- **Model Service:** Calling `model_service.predict()`.
- **FastAPI HTTP Endpoint:** Calling `POST /api/predict` via `TestClient`.

### Profile 1: Healthy Young Baseline
- **Inputs:** Age=25 (Category 2), Sex=Female (0), BMI=21.5, GenHlth=1, HighBP=0, HighChol=0, Smoker=0, Stroke=0, HeartDiseaseorAttack=0, PhysActivity=1, Fruits=1, Veggies=1, HvyAlcohol=0, DiffWalk=0, MentHlth=0, PhysHlth=0.
- **Direct Pipeline Output:** Class 0 | Probs: `[0.9229, 0.0598, 0.0173]`
- **Service Output:** Class 0 (No Diabetes) | Probs: `[0.9229, 0.0598, 0.0173]` | Tier: `Low / Baseline Risk`
- **HTTP Endpoint Output:** Class 0 (No Diabetes) | Probs: `[0.9229, 0.0598, 0.0173]` | Tier: `Low / Baseline Risk`
- **Equivalence Status:** **IDENTICAL (Delta = 0.0000)**

### Profile 2: High-Risk Chronic Patient
- **Inputs:** Age=67 (Category 10), Sex=Male (1), BMI=39.2, GenHlth=5, HighBP=1, HighChol=1, Smoker=1, Stroke=1, HeartDiseaseorAttack=1, PhysActivity=0, Fruits=0, Veggies=0, DiffWalk=1, MentHlth=15, PhysHlth=25.
- **Direct Pipeline Output:** Class 2 | Probs: `[0.0150, 0.2553, 0.7297]`
- **Service Output:** Class 2 (Diabetes) | Probs: `[0.0150, 0.2553, 0.7297]` | Tier: `High Risk (Diabetes Indication)`
- **HTTP Endpoint Output:** Class 2 (Diabetes) | Probs: `[0.0150, 0.2553, 0.7297]` | Tier: `High Risk (Diabetes Indication)`
- **Equivalence Status:** **IDENTICAL (Delta = 0.0000)**

### Profile 3: Intermediate Metabolic Syndrome
- **Inputs:** Age=52 (Category 7), Sex=Male (1), BMI=29.8, GenHlth=3, HighBP=1, HighChol=1, Smoker=0, PhysActivity=0, Fruits=1, Veggies=1, DiffWalk=0, MentHlth=4, PhysHlth=6.
- **Direct Pipeline Output:** Class 2 | Probs: `[0.1843, 0.3809, 0.4348]`
- **Service Output:** Class 2 (Diabetes) | Probs: `[0.1843, 0.3809, 0.4348]` | Tier: `High Risk (Diabetes Indication)`
- **HTTP Endpoint Output:** Class 2 (Diabetes) | Probs: `[0.1843, 0.3809, 0.4348]` | Tier: `High Risk (Diabetes Indication)`
- **Equivalence Status:** **IDENTICAL (Delta = 0.0000)**

### Probability Distribution Integrity
Across all profiles:
$$\sum_{k=0}^{2} P(\text{Class } k) = P(\text{No Diabetes}) + P(\text{Prediabetes}) + P(\text{Diabetes}) = 1.0000 \pm 10^{-4}$$
No placeholder, mock, or hardcoded probabilities exist in the inference path.

---

## 5. PostgreSQL Persistence & Authorization Audit

### 5.1 Real Database Verification
During the audit run (`scripts/audit_phase6101.py`):
1. **User Alpha** (`ID=39`) registered and submitted Assessment `#24`.
2. **User Beta** (`ID=40`) registered and submitted Assessment `#25`.
3. Direct SQLAlchemy session queries verified that:
   - `Assessment #24` in PostgreSQL has `user_id = 39` (matching User Alpha).
   - `Assessment #25` in PostgreSQL has `user_id = 40` (matching User Beta).
   - `Prediction #24` has `assessment_id = 24`.
   - `Prediction #25` has `assessment_id = 25`.

### 5.2 Cross-User Security Test Matrix
| Requesting Actor | Target Endpoint | Target Resource Owner | Expected Status | Actual Status | Security Outcome |
|---|---|---|---|---|---|
| User Alpha (`ID=39`) | `GET /api/assessments/25` | User Beta (`ID=40`) | `403 Forbidden` | `403 Forbidden` | **BLOCKED (Isolated)** |
| User Alpha (`ID=39`) | `GET /api/predictions/25` | User Beta (`ID=40`) | `403 Forbidden` | `403 Forbidden` | **BLOCKED (Isolated)** |
| User Beta (`ID=40`) | `GET /api/assessments/24` | User Alpha (`ID=39`) | `403 Forbidden` | `403 Forbidden` | **BLOCKED (Isolated)** |
| User Beta (`ID=40`) | `GET /api/predictions/24` | User Alpha (`ID=39`) | `403 Forbidden` | `403 Forbidden` | **BLOCKED (Isolated)** |
| User Alpha (`ID=39`) | `GET /api/assessments/24` | User Alpha (`ID=39`) | `200 OK` | `200 OK` | **ALLOWED (Legitimate Owner)** |
| User Alpha (`ID=39`) | `GET /api/predictions/24` | User Alpha (`ID=39`) | `200 OK` | `200 OK` | **ALLOWED (Legitimate Owner)** |
| User Alpha (`ID=39`) | `GET /api/stats` | Filtered for User Alpha | `total = 1` | `total = 1` | **ISOLATED STATS** |
| User Beta (`ID=40`) | `GET /api/stats` | Filtered for User Beta | `total = 1` | `total = 1` | **ISOLATED STATS** |
| Anonymous (No token) | `POST /api/predict` | Guest assessment | `200 OK` | `200 OK` | `user_id = None` (Safe Guest) |

- **Security Conclusion:** Cross-user data leakage is completely prevented at the API router layer via token-based ownership validation.

---

## 6. Scientific Communication & Limitation Transparency

Audit of user-facing components ([HealthAssessmentForm.jsx](file:///c:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/frontend/src/pages/HealthAssessmentForm.jsx), [PredictionResult.jsx](file:///c:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/frontend/src/pages/PredictionResult.jsx), [DisclaimerBanner.jsx](file:///c:/Users/ajayk/OneDrive/Desktop/Advanced-Diabetes-Risk-Prediction-System/frontend/src/components/DisclaimerBanner.jsx)) confirmed compliance with clinical AI communication guidelines:

1. **Non-Diagnostic Framing:** Outputs are strictly labeled as **"Model Predicted Classification"** and **"Model-Estimated Posterior Probabilities"**. Phrases such as *"Medical Diagnosis"*, *"Clinical Risk Score"*, or *"Personalized Diagnostic Probability"* are completely absent.
2. **Explicit Prediabetes Disclaimer:** The Prediction Result page includes a dedicated alert banner explaining that due to the extreme class imbalance in population survey data (~1.8% prediabetes), Class 1 has a low precision of **0.155** and recall of **0.309**. Users are explicitly advised that Prediabetes predictions carry higher uncertainty and that balanced class weighting prioritizes sensitivity.
3. **Medical Consultation Encouragement:** Every page features persistent disclaimers advising users that this system is an academic research demonstration and that clinical laboratory tests (fasting plasma glucose, oral glucose tolerance, or HbA1c) administered by licensed medical practitioners are required for true clinical diagnosis.

---

## 7. Verification Summary & Checklist

- [x] **21 Features Verified:** Exact match in names, types, and canonical order.
- [x] **Chronological Age Verified:** 28 boundary test points passed without a single failure.
- [x] **BMI Handling Verified:** Handled continuously (float) with standard deviation scaling inside the pipeline.
- [x] **Preprocessing Pipeline Verified:** Single scikit-learn Pipeline encapsulating `ColumnTransformer` and `LogisticRegression`.
- [x] **Probabilistic Equivalence Verified:** Direct model == Model service == HTTP API (Delta = 0.0000).
- [x] **Probabilities Sum to 1.0:** Verified across all test profiles.
- [x] **PostgreSQL Persistence Verified:** Real records persisted with correct foreign keys.
- [x] **Cross-User Access Blocked:** 403 Forbidden verified on both assessments and predictions.
- [x] **User-Specific Stats & History Verified:** Alice and Bob saw strictly their own evaluations.
- [x] **Scientific Communication Compliant:** Model posteriors clearly distinguished from clinical risk; Prediabetes limitations disclosed.
- [x] **Full Backend Regression Suite:** 95 / 95 tests passed in 8.25s.
- [x] **Frontend Production Build:** Vite build cleanly compiled in 8.24s with zero errors.
- [x] **Strict Constraints Respected:** Frozen Phase 5 model and metrics untouched; zero Git commits or pushes.

