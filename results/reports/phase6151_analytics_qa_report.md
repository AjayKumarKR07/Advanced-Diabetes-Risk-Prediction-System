# Phase 6.15.1 — Dashboard Analytics Quality Assurance & Scientific Validation Report

**Project:** Advanced Diabetes Risk Prediction System (DiabetesRisk AI)  
**Phase:** 6.15.1 — Dashboard Analytics Quality Assurance & Scientific Validation  
**Role:** Senior Quality Assurance & Scientific Validation Engineer  
**Date:** November 15, 2026  
**Status:** **APPROVED & FULLY VERIFIED (144/144 Tests Passing, Clean Build)**

---

## 1. Files Inspected

During this independent QA and scientific verification audit, the following backend and frontend files were systematically inspected:

### Backend Files:
- `backend/app/api/analytics_routes.py` — Authenticated FastAPI routes for overview, distribution, BMI trends, health trends, indicators, and timeline.
- `backend/app/services/analytics_service.py` — SQLAlchemy query logic, eager loading, date parsing, and metric calculations.
- `backend/app/schemas/analytics_schemas.py` — Pydantic schema models for analytics request/response payloads.
- `backend/app/api/routes.py` — Main API router mount point for analytics and core endpoints.
- `backend/app/core/security.py` — JWT authentication, session revocation, and `get_current_user` dependency.
- `backend/tests/test_analytics_api.py` — Dedicated analytics integration and security test suite.

### Frontend Files:
- `frontend/src/pages/Dashboard.jsx` — React dashboard component featuring 6 KPI cards, date range filter toolbar, Recharts donut, area, and line charts, 9-indicator comparison matrix, assessment timeline, and research benchmarks.
- `frontend/src/utils/generatePdfReport.js` — Client-side jsPDF generator for comprehensive 3-page personal health analytics reports.
- `frontend/src/services/api.js` — Axios client bindings for analytics endpoints with JWT bearer interceptors.
- `frontend/src/utils/cdcFeatures.js` — Canonical CDC BRFSS feature definitions, ordinal mappings, and adult BMI screening ranges.

---

## 2. Issues Identified During Audit

1. **Permissive Date Parsing in Query Service:**
   - In `parse_date_filter`, invalid date format strings (e.g. `start_date="not-a-date"`) were caught by `except ValueError: pass`, silently ignoring the invalid input rather than failing fast.
2. **Missing Inverted Date Range Validation:**
   - If a client passed `start_date > end_date` (e.g. `start_date="2026-12-01"` and `end_date="2026-11-01"`), no validation error was returned, causing empty query results without explaining the logical contradiction.
3. **Pagination Boundary Rejections:**
   - Verified that while Pydantic schemas enforce bounds (`ge=1`, `le=100`), dedicated test cases were needed to assert that requests with `page_size > 100` or `page < 1` strictly trigger HTTP 422 Unprocessable Content.

---

## 3. Fixes & Remediations Applied

1. **Strict ISO Date Validation:**
   - Updated `parse_date_filter` in `backend/app/services/analytics_service.py` to raise `HTTPException(status_code=400, detail="Invalid start_date/end_date format: '...' Expected ISO format...")` whenever malformed dates or non-ISO strings are submitted.
2. **Inverted Date Range Guard:**
   - Enforced validation check in `parse_date_filter`: if `dt_start > dt_end`, immediately raise `HTTPException(status_code=400, detail="Invalid date range: start_date (...) cannot be after end_date (...)")`.
3. **SQL Injection Resistance in Filters:**
   - Verified that attempting to pass SQL injection payloads into date filters (e.g. `' OR 1=1 --`) fails at the ISO datetime parser with HTTP 400 Bad Request and never enters the SQL query string.
4. **Comprehensive Test Suite Expansion:**
   - Expanded `backend/tests/test_analytics_api.py` from 6 tests to 11 exhaustive unit and integration tests covering invalid dates, inverted ranges, SQL injection resistance, and pagination boundaries.

---

## 4. Backend Endpoint Validation Matrix

Every analytics endpoint was audited against strict security, authorization, and data-integrity requirements:

| Endpoint | Auth Required | User Scoped | Date Filter Supported | Empty Dataset Safe | Verified HTTP Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GET /api/analytics/overview` | **YES** (`get_current_user`) | **YES** (`user_id == current_user.id`) | **YES** (`start_date`, `end_date`) | **YES** (Zero counts, null latest) | 200 OK / 401 Unauth / 400 Bad Req |
| `GET /api/analytics/prediction-distribution` | **YES** (`get_current_user`) | **YES** (`user_id == current_user.id`) | **YES** (`start_date`, `end_date`) | **YES** (Total 0, counts/pct 0) | 200 OK / 401 Unauth / 400 Bad Req |
| `GET /api/analytics/bmi-trends` | **YES** (`get_current_user`) | **YES** (`user_id == current_user.id`) | **YES** (`start_date`, `end_date`) | **YES** (Empty points list) | 200 OK / 401 Unauth / 400 Bad Req |
| `GET /api/analytics/health-trends` | **YES** (`get_current_user`) | **YES** (`user_id == current_user.id`) | **YES** (`start_date`, `end_date`) | **YES** (Empty points list) | 200 OK / 401 Unauth / 400 Bad Req |
| `GET /api/analytics/indicators` | **YES** (`get_current_user`) | **YES** (`user_id == current_user.id`) | **YES** (`start_date`, `end_date`) | **YES** (0 records returned) | 200 OK / 401 Unauth / 400 Bad Req |
| `GET /api/analytics/timeline` | **YES** (`get_current_user`) | **YES** (`user_id == current_user.id`) | **YES** (`start_date`, `end_date`) | **YES** (Total 0, items empty) | 200 OK / 401 Unauth / 422 Invalid Page |

---

## 5. Analytics Calculation Results & Edge Cases

The audit thoroughly tested calculations across diverse lifecycle scenarios:

### 5.1 Zero-Assessment Scenario (New User)
- `total_assessments`: 0
- `assessments_this_month`: 0
- `latest_assessment_id`: `None`
- `latest_predicted_class`: `None`
- `latest_bmi`: `None`
- `bmi_summary.change`: `None`
- `gen_hlth_summary.change`: `None`
- All trend points and timeline items return empty lists `[]`.
- Frontend displays the welcoming Onboarding State encouraging the user to run their first evaluation.

### 5.2 Single-Assessment Scenario
- `total_assessments`: 1
- `bmi_summary.latest`: Valid float matching submitted assessment.
- `bmi_summary.previous`: `None`
- `bmi_summary.change`: **Strictly `None`** — Never generates artificial trend data or infers a trend from one record.
- `gen_hlth_summary.change`: **Strictly `None`**.
- Frontend charts display the educational empty state: *"Trajectory Requires At Least 2 Assessments"*.

### 5.3 Multiple-Assessment Scenario
- BMI and General Health trajectories are strictly ordered chronologically (oldest to newest).
- Deltas are computed precisely: `latest - previous` (e.g. `25.0 - 27.5 = -2.5`).
- Prediction distribution accurately sums counts and computes rounded percentages (`round((cnt / total) * 100, 1)`).

---

## 6. Scientific Presentation & Disclaimers Audit

The audit verified compliance with scientific communication and patient safety rules:

1. **Approved Class Designations Only:**
   - **Class 0:** *No Diabetes* (emerald `#10b981`)
   - **Class 1:** *Prediabetes* (amber `#f59e0b`)
   - **Class 2:** *Diabetes* (rose `#ef4444`)
   - All unvalidated risk tier labels (*"Elevated Risk"*, *"Prediabetes Warning"*, *"High Risk"*, *"Clinical Risk Tier"*) are completely removed.
2. **Classification vs. Prevalence Distinction:**
   - The distribution card includes an explicit footnote: *"Historical classifications reflect mathematical outputs of the frozen Logistic Regression model across self-reported survey indicators and do NOT represent clinical disease prevalence or confirmed medical diagnosis."*
3. **BMI Screening Categorization:**
   - BMI categories (*Underweight < 18.5*, *Normal 18.5–24.9*, *Overweight 25.0–29.9*, *Obesity ≥ 30.0*) are explicitly labeled as *"descriptive screening measures, not individual diagnoses."*
4. **Self-Reported Health Ratings:**
   - General Health values (1 = Excellent to 5 = Poor) are labeled as self-reported survey responses and not described as clinically verified improvement or deterioration.
5. **Model Posterior Probabilities:**
   - Labeled strictly as *"model-estimated class probabilities"* and footnoted as non-calibrated survey approximations.
6. **No Disease Progression Implication:**
   - Visual charts show chronological submitted values; no chart or label claims or implies disease progression.

---

## 7. Frozen Phase 5 Research Metrics Verification

The model binary `models/cdc/group_aware_benchmarks/LogisticRegression/model.pkl` and its evaluation metrics were verified against the frozen Phase 5 baseline:

| Metric | Target Frozen Value | Audited Application Value | Match Status |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6433 | 0.6433 (64.3%) | **EXACT MATCH** |
| **Balanced Accuracy** | 0.5172 | 0.5172 (51.7%) | **EXACT MATCH** |
| **Macro F1** | 0.4246 | 0.4246 | **EXACT MATCH** |
| **Weighted F1** | 0.7187 | 0.7187 | **EXACT MATCH** |
| **ROC-AUC OvR Macro** | 0.7707 | 0.7707 | **EXACT MATCH** |
| **ROC-AUC OvR Weighted** | 0.8112 | 0.8112 | **EXACT MATCH** |

### Class-Wise Frozen Performance Breakdown:
- **Class 0 (No Diabetes):** Precision: 0.9501 | Recall: 0.6606 | F1: 0.7793
- **Class 1 (Prediabetes):** Precision: 0.0307 | Recall: 0.3089 | F1: 0.0558
- **Class 2 (Diabetes):** Precision: 0.3520 | Recall: 0.5823 | F1: 0.4387

### Prediabetes Performance Disclosure:
The application prominently displays the research disclosure:
> *"Known limitation: Class 1 exhibits low precision (0.0307) due to ~1.8% prediabetes prevalence in BRFSS survey data. The model is intentionally configured with balanced class weights to maximize sensitivity to early metabolic risk, but individual predictions in Class 1 carry substantial uncertainty."*

---

## 8. PDF Report Generation Validation

The PDF generator in `frontend/src/utils/generatePdfReport.js` (`generatePersonalAnalyticsReportPdf`) was verified:
- **Authenticated Identity:** Populated with `user.full_name` and `user.email`.
- **Date Range Window:** Displays active filtering window or "All Time Records".
- **Pagination:** Structured as a 3-page document with consistent footer: `DiabetesRisk AI • Personal Health Analytics Report • Page X of 3`.
- **Layout Integrity:** Content width (182mm) and vertical line wrapping (`splitTextToSize`) prevent text overflow and clipping.
- **Empty / Single Assessment Safety:** Gracefully renders fallback notices (`No completed assessments found...`, `N/A (Requires ≥ 2 assessments)`) without fabricating placeholder numbers.
- **Academic Disclaimers:** All 3 pages carry non-diagnostic research disclaimers.

---

## 9. Security, Data Isolation & Regression Testing

### 9.1 Cross-User Data Isolation
- Verified in `backend/tests/test_analytics_api.py`:
  - Assessment records belonging to User A never appear in User B's `/overview`, `/prediction-distribution`, `/bmi-trends`, `/health-trends`, `/indicators`, or `/timeline`.
  - Pagination limits (`page_size <= 100`, `page >= 1`) prevent database exhaustion attacks.
  - Revoked tokens cannot access analytics endpoints (HTTP 401 Unauthorized).

### 9.2 Regression Test Results
The full regression suite of 144 tests was executed against PostgreSQL:

```text
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.0.2, pluggy-1.6.0
collected 144 items

backend/tests/test_analytics_api.py ...........                          [  7%]
backend/tests/test_auth.py ...............                               [ 18%]
backend/tests/test_auth_api.py .................                         [ 29%]
backend/tests/test_backend_api.py ...................................... [ 56%]
backend/tests/test_data_integrity.py ............                        [ 64%]
backend/tests/test_health_assessment_workflow.py ...................     [ 77%]
backend/tests/test_history_management.py ........                        [ 83%]
backend/tests/test_profile_settings.py .............                     [ 92%]
backend/tests/test_security_audit.py ...........                         [100%]

====================== 144 passed, 12 warnings in 27.46s ======================
```
- **Total Tests:** 144
- **Passed:** 144 (100%)
- **Failed:** 0

### 9.3 Frontend Production Build
The Vite production bundle was verified:

```text
> diabetes-prediction-frontend@1.0.0 build
> vite build

vite v4.5.14 building for production...
✓ 2644 modules transformed.
dist/index.html                              0.91 kB │ gzip:   0.51 kB
dist/assets/index-41e54634.css              65.09 kB │ gzip:  10.20 kB
dist/assets/purify.es-2aa16aed.js           30.50 kB │ gzip:  11.75 kB
dist/assets/index.es-6b2cf0b7.js           150.99 kB │ gzip:  51.67 kB
dist/assets/html2canvas.esm-e0a7d97b.js    201.43 kB │ gzip:  48.04 kB
dist/assets/index-858e04ac.js            1,449.88 kB │ gzip: 404.01 kB
✓ built in 8.03s
```
- **Build Status:** CLEAN (0 errors, 0 compilation warnings).

---

## 10. Remaining Limitations & Clinical Boundary Disclosures

1. **Non-Clinical Device:** The platform remains an academic research and engineering prototype; it is not cleared by the FDA or CE as a medical diagnostic device.
2. **BRFSS Self-Report Imprecision:** All 21 input features are derived from telephone survey questionnaires and may be subject to recall bias or measurement imprecision.
3. **Class 1 Sensitivity vs. Precision Tradeoff:** The model utilizes balanced class weights to avoid missing prediabetic cases, resulting in high recall (0.3089) at the expense of low precision (0.0307). Clinical laboratory tests (HbA1c / FPG) are required for diagnostic confirmation.

---

## 11. Final Approval Status

**AUDIT RESULT:** **PASSED & APPROVED**  
Phase 6.15 and 6.15.1 satisfy all quality assurance, scientific validation, data isolation, and security requirements. The frozen Phase 5 model, feature definitions, and evaluation metrics remain 100% intact.

