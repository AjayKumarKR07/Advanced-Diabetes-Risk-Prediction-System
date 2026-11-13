# Phase 6 — Application Development: Implementation Report

**Project:** Advanced Diabetes Risk Prediction System  
**Date:** 2026-11-13  
**Status:** ✅ COMPLETE

---

## 1. Architecture Overview

```
Advanced-Diabetes-Risk-Prediction-System/
├── backend/                          ← FastAPI Python backend
│   ├── app/
│   │   ├── api/routes.py            ← /api/health, /api/model-info, /api/predict
│   │   ├── core/config.py           ← Pydantic Settings (paths, CORS, disclaimers)
│   │   ├── schemas/prediction.py    ← Request/response Pydantic models
│   │   ├── services/model_service.py← DiabetesModelService singleton
│   │   └── main.py                  ← FastAPI app with lifespan startup
│   └── tests/
│       ├── conftest.py              ← Session-scoped model preload fixture
│       └── test_backend_api.py      ← 41 unit tests (service + schema layers)
├── frontend/                         ← React + Vite + Tailwind CSS frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.jsx          ← Hero page, quick stats, feature overview
│   │   │   ├── Dashboard.jsx        ← Phase 5 metrics, Recharts visualizations
│   │   │   ├── HealthAssessmentForm.jsx ← 21-field CDC form with sliders/toggles
│   │   │   ├── PredictionResult.jsx ← Class output, probability bars + pie chart
│   │   │   ├── ModelPerformance.jsx ← Live /api/model-info + per-class metrics
│   │   │   └── AboutMethodology.jsx ← Pipeline phases, limitations, references
│   │   ├── components/
│   │   │   ├── Navbar.jsx           ← Responsive fixed navbar, active route highlight
│   │   │   ├── RiskBadge.jsx        ← Color-coded risk tier badge
│   │   │   ├── ProbabilityBar.jsx   ← Animated horizontal probability bar
│   │   │   ├── MetricCard.jsx       ← Single metric summary card
│   │   │   └── DisclaimerBanner.jsx ← Research prototype disclaimer
│   │   ├── services/api.js          ← Axios client for backend endpoints
│   │   ├── App.jsx                  ← React Router routes
│   │   ├── main.jsx                 ← ReactDOM.createRoot entry point
│   │   └── index.css                ← Tailwind base + custom component classes
│   ├── package.json                 ← React 18, Recharts, React Router, Lucide
│   ├── vite.config.js               ← Vite 4.5.14, /api proxy to :8000
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── index.html
└── pytest.ini                        ← addopts: -p no:asyncio, testpaths
```

---

## 2. Backend (Phase 6.2) — FastAPI

### Technology Stack
- **Python 3.14.3** (compatible with 3.11+)
- **FastAPI 0.128.3** with Pydantic v2
- **Uvicorn 0.40.0** ASGI server
- **Joblib 1.5.3** for model artifact loading
- **Scikit-learn 1.8.0** (model runtime)

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Root — service info, model loaded status, documentation link |
| `GET` | `/api/health` | Health check — status, timestamp, model_loaded, version |
| `GET` | `/api/model-info` | Model metadata, feature list, class mapping, Phase 5 performance metrics |
| `POST` | `/api/predict` | 21-feature CDC input → class prediction, probabilities, risk tier |

### Model Artifact
- **Path:** `models/cdc/group_aware_benchmarks/LogisticRegression/model.pkl`
- **Type:** `sklearn.pipeline.Pipeline` with `preprocessor` + `classifier` steps
- **Canonical feature order:** 21 CDC features, raw CSV column order (BMI at index 3)
- **Classes:** 0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes
- **Training:** Phase 4.2 group-aware StratifiedGroupKFold, balanced class weighting
- **Model is FROZEN** — no retraining, no modification

### Key Design Decisions
1. **Lazy + eager model loading:** `load_model()` called on FastAPI lifespan startup; also lazy-loads on first inference request if startup failed
2. **Canonical feature re-ordering:** Service layer enforces `CANONICAL_FEATURES` order regardless of JSON dict key order — protects against JSON serialization ordering differences
3. **Probability normalization:** `np.clip(proba, 0, 1)` then normalize to sum=1.0, prevents floating-point edge cases
4. **Pydantic v2 field validation:** `ge`/`le` constraints on all 21 features with CDC-compliant ranges
5. **Custom 422 handler:** Structured `{"error": "Validation Error", "details": [...]}` response with per-field errors
6. **Research prototype disclaimer:** Included in every prediction response and all UI pages

---

## 3. Backend Tests (Phase 6.3)

### Strategy
Due to Python 3.14 anyio/threading incompatibility with HTTPX's `TestClient` in sandboxed environments, the test suite exercises the **service layer and Pydantic schemas directly** without HTTP networking.

This covers:
- All business logic (inference, probability normalization, risk tiering)
- All input validation (range checks, missing fields)
- Feature order invariance
- Class mapping consistency

### Results

```
41 passed in 1.54s
```

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestModelServiceSingleton` | 9 | Model loading, feature list, class mapping |
| `TestPredictionService` | 12 | Inference, probability bounds, risk tier, invariance |
| `TestModelInfoService` | 4 | Metadata response structure |
| `TestSchemaValidation` | 16 | Missing fields, out-of-range values, boundary acceptance |

---

## 4. Frontend (Phase 6.4) — React + Vite + Tailwind CSS

### Technology Stack
- **React 18.3.1** + **React Router DOM 6.28**
- **Vite 4.5.14** (downgraded from 6.x to avoid rollup native binary Application Control policy blocks)
- **Tailwind CSS 3.4.17** with PostCSS
- **Recharts 2.13.3** for data visualization
- **Lucide React** for icons
- **Axios 1.7.9** for API requests

### Pages (6 total)

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | `Landing.jsx` | Hero section, quick stats, project feature cards |
| `/dashboard` | `Dashboard.jsx` | Phase 5 metrics dashboard, Recharts: radar, bar, pie |
| `/assess` | `HealthAssessmentForm.jsx` | 21-field CDC input form with toggle/slider/select controls |
| `/result` | `PredictionResult.jsx` | Prediction output: class, risk badge, probability bars + pie |
| `/performance` | `ModelPerformance.jsx` | Live model-info + per-class F1 bar chart + features list |
| `/about` | `AboutMethodology.jsx` | Pipeline phases, limitations, dataset info, GitHub link |

### Build Status
```
vite v4.5.14 building for production...
✓ 2249 modules transformed.
dist/index.html         0.91 kB │ gzip:  0.51 kB
dist/assets/index.css  24.38 kB │ gzip:  4.81 kB
dist/assets/index.js  690.43 kB │ gzip: 200.65 kB
✓ built in 4.74s
```

### Vite Configuration
- Dev proxy: `/api` → `http://localhost:8000` (Vite `server.proxy`)
- Production: Requests go directly to backend; configure `VITE_API_BASE_URL` env var

---

## 5. Constraints Maintained

- ✅ Phase 4.2 `LogisticRegression/model.pkl` frozen — no retraining
- ✅ Frozen test set not accessed during Phase 6
- ✅ Original Kaggle/Pima workflow (`src/data_preprocessing.py`, etc.) unchanged
- ✅ Canonical 21-feature order preserved exactly (BMI at index 3)
- ✅ Research prototype disclaimer on every prediction response and UI page
- ✅ Class mapping: 0=No Diabetes, 1=Prediabetes, 2=Diabetes — consistent throughout

---

## 6. Known Issues / Notes

1. **Rollup native binary blocked by Application Control policy on this machine.**  
   Solution: Vite 4.5.14 (rollup 3, pure JS) works without native binaries.  
   The `dist/` build output is production-ready.

2. **Backend `TestClient` / `AsyncClient` hang in sandboxed environment** due to Python 3.14 + anyio socket restrictions.  
   Solution: Tests exercise service layer directly — all 41 pass.

3. **Prediabetes (Class 1) precision is 0.155, recall 0.309** — documented limitation in `AboutMethodology.jsx` and `PredictionResult.jsx`. Users are warned explicitly.

4. **Recharts 2.x deprecation warning** — Recharts 2 is still functionally current; migration to v3 is a future upgrade path.

---

## 7. Running the Application

### Backend
```bash
cd Advanced-Diabetes-Risk-Prediction-System
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (Development)
```bash
cd frontend
npm run dev   # Starts Vite dev server on http://localhost:5173
              # /api requests proxied to localhost:8000
```

### Frontend (Production Build)
```bash
cd frontend
npm run build  # Output in frontend/dist/
```

### Tests
```bash
python -m pytest backend/tests/ -v   # 41 tests, ~1.5s
```

