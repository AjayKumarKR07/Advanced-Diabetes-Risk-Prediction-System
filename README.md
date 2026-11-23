# 🩺 DiabetesRisk AI — Advanced Clinical Diabetes Risk Prediction System

[![Python 3.14 / 3.12](https://img.shields.io/badge/Python-3.14%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3+-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-4.5+-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-3.4+-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%7C%2018-4169E1.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic-grade, full-stack clinical analytics and machine learning system for 3-class diabetes risk screening (`No Diabetes`, `Prediabetes`, and `Diabetes`). Built upon the CDC BRFSS 2015 dataset (253,680 records, 21 clinical and lifestyle features), the system integrates a mathematically transparent, frozen **Logistic Regression** model pipeline with Explainable AI (XAI), longitudinal patient health tracking, rule-based clinical insights, multi-section PDF reporting, and secure role-isolated patient portals.

---

## 📑 Table of Contents

- [Overview & Clinical Mission](#-overview--clinical-mission)
- [System Architecture](#-system-architecture)
- [Key Features & Capabilities](#-key-features--capabilities)
- [Machine Learning & Scientific Methodology](#-machine-learning--scientific-methodology)
  - [CDC BRFSS 2015 Dataset](#cdc-brfss-2015-dataset)
  - [Leakage Prevention & Group-Aware Cross-Validation](#leakage-prevention--group-aware-cross-validation)
  - [Model Selection & Evaluation Metrics](#model-selection--evaluation-metrics)
- [Technology Stack](#-technology-stack)
- [Application Structure](#-application-structure)
- [Installation & Local Setup](#-installation--local-setup)
  - [Prerequisites](#prerequisites)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Backend Setup & PostgreSQL](#2-backend-setup--postgresql)
  - [3. Frontend Setup](#3-frontend-setup)
- [Running the Application](#-running-the-application)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [API Reference](#-api-reference)
- [Production Deployment with Docker](#-production-deployment-with-docker)
- [Medical & Ethical Disclaimer](#-medical--ethical-disclaimer)
- [Author & License](#-author--license)

---

## 🎯 Overview & Clinical Mission

Diabetes affects hundreds of millions of people globally, yet a vast number of individuals living with prediabetes remain undiagnosed. Traditional standard machine learning benchmarks frequently fail in clinical settings because:

1. **Extreme Class Imbalance:** Minority classes like prediabetes (~1.8% prevalence in population surveys) are collapsed by algorithms (such as unweighted Random Forest or XGBoost) optimizing solely for overall nominal accuracy.
2. **Data Leakage via Duplicate Feature Vectors:** Survey datasets contain identical demographic/behavioral response patterns; naïve random splitting causes severe train-test leakage.
3. **Black-Box Opacity:** Clinicians and patients require interpretable factor attribution to understand *why* a particular risk level was assigned.

**DiabetesRisk AI** resolves these issues through a strictly validated, leakage-free, group-aware cross-validation framework and a production-grade clinical analytics application designed for both healthcare researchers and patients.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT / BROWSER INTERFACE                            │
│  React 18 + Vite + Tailwind CSS + Recharts + Lucide Icons + jsPDF / html2canvas  │
└────────────────────────┬────────────────────────────────┬───────────────────────┘
                         │                                │
         REST API (JSON) │                                │ Auth Bearer JWT
                         ▼                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND GATEWAY (Python)                         │
│  - Correlation ID Middleware (X-Request-ID)   - Security Headers (HSTS, CSP)    │
│  - Role / User Scoped Dependencies            - Pydantic v2 Schema Validation   │
├────────────────────────────────┬────────────────────────────────────────────────┤
│          API ROUTERS           │               CORE SERVICES                    │
│  • /api/auth                   │  • ModelService (Frozen Phase 5 Pipeline)      │
│  • /api/assessments            │  • ExplainService (XAI Feature Breakdown)      │
│  • /api/analytics              │  • InsightsService (Rule-Based Clinical Flags) │
│  • /api/insights               │  • DBService (SQLAlchemy 2.0 ORM CRUD)         │
│  • /api/explain                │  • AvatarService & Notifications Engine        │
│  • /api/notifications          │  • ChatService (AI Clinical Assistant)         │
│  • /api/chat                   │                                                │
└───────────────┬────────────────┴────────────────────────┬───────────────────────┘
                │                                         │
                ▼                                         ▼
┌───────────────────────────────┐        ┌────────────────────────────────────────┐
│     POSTGRESQL DATABASE       │        │         FROZEN ML ENGINE               │
│  • Users & Profiles           │        │  • LogisticRegression Pipeline         │
│  • Assessments & Predictions  │        │  • 21 Standardized Features            │
│  • Security Audit & History   │        │  • Balanced Class Weighting            │
│  • User Notification Records  │        │  • SHA-256 Verified (767b082c...)      │
└───────────────────────────────┘        └────────────────────────────────────────┘
```

---

## ✨ Key Features & Capabilities

### 1. Clinical 3-Class Diabetes Risk Assessment
- Evaluates risk across canonical diagnostic tiers:
  - **Class 0:** No Diabetes
  - **Class 1:** Prediabetes *(high-sensitivity detection for early lifestyle intervention)*
  - **Class 2:** Diabetes
- Inputs 21 standardized biometric, clinical, and lifestyle variables directly derived from the CDC BRFSS standard.

### 2. Model Explainability & Factor Attribution (XAI)
- Powered by `FeatureContributionCard.jsx` and `/api/explain`.
- Decomposes the model's standardized log-odds coefficients multiplied by normalized patient inputs.
- Clearly displays the **Top 5 Risk-Elevating Factors** (e.g., elevated BMI, High Blood Pressure, Age) and **Top Risk-Mitigating Factors** (e.g., active physical lifestyle, fruit/vegetable consumption).

### 3. Professional 12-Item Navigation Hierarchy
The persistent desktop and mobile drawer navigation is organized into four distinct functional domains:

```
WORKSPACE
  ├── Dashboard               (/dashboard)          — Longitudinal summary & quick actions
  ├── New Assessment          (/assess) [New Badge] — 21-variable interactive questionnaire
  └── Assessment History      (/history)            — Comprehensive historical log with filtering

HEALTH & ANALYTICS
  ├── Personal Analytics      (/analytics)          — Multi-dimensional probability trend charts
  ├── Health Insights         (/insights)           — Evidence-based clinical recommendations
  └── Health Reports          (/reports)            — Formal PDF report exporter

RESEARCH & INFORMATION
  ├── Model Performance       (/model-performance) — Live validation confusion matrix & metrics
  └── Research & Methodology  (/research)          — Full scientific documentation & CDC overview

ACCOUNT & SUPPORT
  ├── Notifications           (/notifications)      — Clinical reassessment check-ins & alerts
  ├── User Profile            (/profile)            — Avatar management & personal information
  ├── Account Settings        (/settings)           — Security preferences & password controls
  └── Help & About            (/help)               — Educational documentation & clinical FAQ
```

### 4. Enhanced Multi-Section Clinical PDF Exporter
- Employs dynamic, on-demand imports (`jsPDF` and `html2canvas`) keeping initial web bundle sizes below **110 kB**.
- Generates formatted clinical summaries featuring:
  - Patient demographics & assessment metadata
  - Model-predicted probability distribution bar
  - Canonical risk classification badge
  - Factor-by-factor risk contribution table
  - Personalized evidence-based health recommendations
  - Prominent scientific and non-diagnostic disclaimers

### 5. Automated Notification & Check-in Center
- Integrated `NotificationBell.jsx` and dedicated `/notifications` management page.
- Tracks 30-day and 60-day reassessment reminders and clinical check-in suggestions.
- **Strict Clinical Consistency:** Exclusively surfaces canonical classes (`Class 0 — No Diabetes`, `Class 1 — Prediabetes`, `Class 2 — Diabetes`). Completely eliminates arbitrary unvalidated risk labels (e.g., "High Risk", "Moderate Risk") and removes unwarranted emergency urgency flags.

### 6. Embedded AI Clinical Assistant
- Accessible via a floating widget (`AIChatWidget.jsx`) and `/api/chat`.
- Provides context-aware answers to user inquiries regarding lab tests (HbA1c, FPG), clinical terminology, and lifestyle interventions, guarded with mandatory clinical boundary disclaimers.

### 7. Enterprise Security & Architecture
- **Stateless JWT Authentication:** Access token expiration, cryptographic signature verification, and secure password hashing using `bcrypt`.
- **Hardened HTTP Headers:** Enforces Content Security Policy (CSP), HSTS, `X-Content-Type-Options: nosniff`, and `X-Frame-Options: DENY`.
- **Strict Data Isolation:** PostgreSQL queries enforce user-scoped ownership; cross-user data tampering is strictly blocked and audited.

---

## 🧠 Machine Learning & Scientific Methodology

### CDC BRFSS 2015 Dataset

The model is trained and benchmarked on the CDC's **Behavioral Risk Factor Surveillance System (BRFSS) 2015** annual survey:
- **Total Population Size:** 253,680 survey respondents
- **Split Breakdown:** 202,944 train rows (80%) / 50,736 held-out test rows (20%)
- **Target Distribution:**
  - `Class 0 (No Diabetes):` 213,703 (84.2%)
  - `Class 1 (Prediabetes):` 4,631 (1.8%)
  - `Class 2 (Diabetes):` 35,346 (13.9%)

### 21 Standardized Input Features

| Feature Name | Clinical / Demographic Meaning | Values |
|--------------|--------------------------------|--------|
| `HighBP` | Has the patient been diagnosed with high blood pressure? | 0 = No, 1 = Yes |
| `HighChol` | Has the patient been diagnosed with high cholesterol? | 0 = No, 1 = Yes |
| `CholCheck` | Cholesterol check performed within past 5 years | 0 = No, 1 = Yes |
| `BMI` | Body Mass Index ($kg/m^2$) | Continuous |
| `Smoker` | Has smoked at least 100 cigarettes in lifetime? | 0 = No, 1 = Yes |
| `Stroke` | Ever diagnosed with a stroke? | 0 = No, 1 = Yes |
| `HeartDiseaseorAttack` | Coronary heart disease (CHD) or myocardial infarction (MI) | 0 = No, 1 = Yes |
| `PhysActivity` | Any physical activity in past 30 days (excluding job) | 0 = No, 1 = Yes |
| `Fruits` | Consumes fruit 1 or more times per day | 0 = No, 1 = Yes |
| `Veggies` | Consumes vegetables 1 or more times per day | 0 = No, 1 = Yes |
| `HvyAlcoholConsump` | Heavy alcohol consumption (Adult men $\ge 14$/wk, women $\ge 7$/wk) | 0 = No, 1 = Yes |
| `AnyHealthcare` | Has any health insurance coverage or healthcare plan | 0 = No, 1 = Yes |
| `NoDocbcCost` | Needed to see a doctor in past 12 mos but could not due to cost | 0 = No, 1 = Yes |
| `GenHlth` | Self-reported general health status | 1 = Excellent to 5 = Poor |
| `MentHlth` | Days of poor mental health in past 30 days | 0 to 30 days |
| `PhysHlth` | Days of physical illness or injury in past 30 days | 0 to 30 days |
| `DiffWalk` | Serious difficulty walking or climbing stairs | 0 = No, 1 = Yes |
| `Sex` | Biological sex | 0 = Female, 1 = Male |
| `Age` | 13-level age category | 1 = 18–24 to 13 = 80+ |
| `Education` | Highest educational level completed | 1 = Never attended to 6 = 4+ yrs college |
| `Income` | Annual household income scale | 1 = <$10k to 8 = $\ge$ $75k |

### Leakage Prevention & Group-Aware Cross-Validation

Standard randomized cross-validation on large health surveys creates catastrophic data leakage: individuals with identical 21-feature vectors are split across both train and validation folds. 
- **Methodology:** We compute an MD5 hash profile for every unique feature combination across the 253,680 records and enforce **StratifiedGroupKFold (5-fold)**.
- **Guarantee:** Zero group overlap between training and validation folds ($Overlap = 0$).

### Model Selection & Evaluation Metrics

While tree-based ensembles (Random Forest, XGBoost) achieve marginally higher nominal accuracy, they achieve this by predicting only Class 0 and Class 2, achieving **<0.5% recall on Class 1 (Prediabetes)**.

For clinical screening, early disease interception is paramount. **Logistic Regression with Balanced Class Weighting** was selected based on strict Out-Of-Fold (OOF) cross-validation and frozen in Phase 5:

#### Frozen Phase 5 Test Set Metrics (50,736 Held-Out Patients)

| Evaluation Metric | Frozen Value | Clinical Significance |
|-------------------|--------------|-----------------------|
| **Accuracy** | **64.33%** | Reflects balanced class penalization across all categories |
| **Balanced Accuracy** | **51.72%** | Significantly outperforms random guessing across 3 classes |
| **Macro F1-Score** | **0.4246** | Unweighted harmonic mean across majority & minority classes |
| **Weighted F1-Score** | **0.7187** | Prevalence-weighted multi-class F1 metric |
| **ROC-AUC (Macro OvR)** | **0.7707** | High discriminative capability across all 3 classes |
| **ROC-AUC (Weighted OvR)** | **0.8112** | Population-level screening discrimination power |
| **Cohen's Kappa ($\kappa$)** | **0.2428** | Statistically significant agreement beyond chance |
| **Matthews Corr. Coeff. (MCC)** | **0.2912** | Robust multi-class correlation metric |

#### Per-Class Performance Breakdown

| Class Identifier | Class Name | Precision | Recall (Sensitivity) | F1-Score | Support |
|:----------------:|------------|:---------:|:--------------------:|:--------:|:-------:|
| **Class 0** | No Diabetes | 0.9501 | 0.6606 | 0.7793 | 42,741 |
| **Class 1** | Prediabetes *(Weak Class)* | 0.0307 | **0.3089** | 0.0558 | 926 |
| **Class 2** | Diabetes | 0.3520 | **0.5823** | 0.4387 | 7,069 |

> ⚠️ **Scientific Limitation:** Class 1 (Prediabetes) exhibits low precision (0.0307) alongside a high screening recall (30.89%). This reflects a deliberate clinical trade-off: capturing nearly a third of all prediabetic cases for early blood glucose testing rather than silently missing 99.5% of them (as default unweighted models do).

- **Production Model Artifact:** `models/cdc/group_aware_benchmarks/LogisticRegression/model.pkl`  
- **Cryptographic Hash (SHA-256):** `767b082ce6bfd76d65b7b9c769dcdccd6e6966a8568348684f8abc04e4d66413`

---

## 🛠 Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend UI** | React 18, Vite 4.5, Tailwind CSS 3.4, Lucide Icons, Recharts 2.13 |
| **Reporting & Export** | jsPDF 4.2, html2canvas (Dynamic / Lazy-Loaded) |
| **Backend API** | FastAPI 0.115+, Uvicorn (ASGI), Python 3.14 / 3.12, Pydantic v2 |
| **Database & ORM** | PostgreSQL 16/18, SQLAlchemy 2.0, Psycopg 3, Alembic |
| **Machine Learning** | Scikit-Learn 1.4+, NumPy, Pandas, Joblib |
| **Security & Auth** | Python-Jose / PyJWT, Passlib (bcrypt), CORSMiddleware |
| **Testing** | Pytest 9.0+, AnyIO |
| **DevOps / Containers** | Docker, Docker Compose, Nginx Reverse Proxy |

---

## 📁 Application Structure

```
Advanced-Diabetes-Risk-Prediction-System/
├── backend/                          # FastAPI Backend Application
│   ├── app/
│   │   ├── api/                      # REST Endpoints
│   │   │   ├── analytics_routes.py   # Longitudinal trend & breakdown endpoints
│   │   │   ├── auth_routes.py        # Authentication & profile endpoints
│   │   │   ├── chat_routes.py        # AI clinical assistant endpoints
│   │   │   ├── explain_routes.py     # XAI feature attribution endpoints
│   │   │   ├── insights_routes.py    # Evidence-based health recommendations
│   │   │   ├── notifications_routes.py # User notifications & check-in reminders
│   │   │   └── routes.py             # Root router & health probes
│   │   ├── core/                     # Configuration, database engine & security
│   │   ├── models/                   # SQLAlchemy ORM database models
│   │   ├── schemas/                  # Pydantic v2 validation schemas
│   │   ├── services/                 # Model inference, DB & notification services
│   │   └── main.py                   # FastAPI lifespan application entrypoint
│   ├── tests/                        # 163 Live integration test functions
│   └── .env.example                  # Backend environment template
│
├── frontend/                         # React 18 + Vite SPA
│   ├── src/
│   │   ├── components/               # UI Components
│   │   │   ├── AuthenticatedLayout.jsx # 12-item 4-group clinical navigation
│   │   │   ├── FeatureContributionCard.jsx # XAI horizontal bar chart card
│   │   │   ├── NotificationBell.jsx  # Real-time notification header popover
│   │   │   ├── AIChatWidget.jsx      # Floating clinical assistant widget
│   │   │   └── ProtectedRoute.jsx    # Client-side JWT auth boundary
│   │   ├── pages/                    # 13 Application views
│   │   │   ├── Dashboard.jsx         # Clinical patient dashboard
│   │   │   ├── HealthAssessmentForm.jsx # 21-feature assessment questionnaire
│   │   │   ├── PredictionResult.jsx  # Detailed prediction analysis & XAI
│   │   │   ├── History.jsx           # Assessment audit log & deletion
│   │   │   ├── Analytics.jsx         # Longitudinal multi-chart analytics
│   │   │   ├── Insights.jsx          # Personalized health action items
│   │   │   ├── HealthReports.jsx     # On-demand PDF report exporter
│   │   │   ├── AuthModelPerformance.jsx # Live model validation metrics
│   │   │   ├── ResearchMethodology.jsx # Comprehensive CDC methodology
│   │   │   ├── Notifications.jsx     # Notification center management
│   │   │   ├── Profile.jsx           # Patient profile & avatar management
│   │   │   ├── Settings.jsx          # Security preferences & controls
│   │   │   └── HelpAbout.jsx         # Clinical FAQ & documentation
│   │   └── App.jsx                   # Central route registry & theme provider
│   ├── package.json
│   └── tailwind.config.js
│
├── models/                           # Machine Learning Artifacts
│   └── cdc/
│       ├── cdc_feature_names.json    # 21 Canonical feature definitions
│       └── group_aware_benchmarks/   # Frozen benchmark artifacts
│           └── LogisticRegression/
│               ├── model.pkl         # Production LR model pipeline
│               └── metadata.json     # Hyperparameters & data schemas
│
├── results/                          # Scientific Reports & Validation Logs
│   └── reports/
│       ├── phase5_final_metrics.json # Frozen Phase 5 test metrics
│       └── phase6174_live_integration_report.md
│
├── deploy/                           # Production Container Orchestration
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── docker-compose.prod.yml
│   └── nginx/default.conf
│
├── tests/                            # CDC & Model Offline Unit Test Suite
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## ⚙️ Installation & Local Setup

### Prerequisites

- **Python:** 3.12 or 3.14 recommended
- **Node.js:** v18.0+ & **pnpm** (or npm / yarn)
- **PostgreSQL:** Version 15, 16, or 18 running locally or on Docker
- **Git**

---

### 1. Clone Repository

```bash
git clone https://github.com/AjayKumarKR07/Advanced-Diabetes-Risk-Prediction-System.git
cd Advanced-Diabetes-Risk-Prediction-System
```

---

### 2. Backend Setup & PostgreSQL

#### Create Database
Create a database named `diabetes_risk_db` in your PostgreSQL instance:

```sql
CREATE DATABASE diabetes_risk_db;
```

#### Configure Environment Variables
Copy `backend/.env.example` to `backend/.env` and update your database credentials:

```bash
cp backend/.env.example backend/.env
```

*Example `backend/.env`:*
```env
DATABASE_URL=postgresql+psycopg://postgres:your_password@localhost:5432/diabetes_risk_db
SECRET_KEY=generate_a_random_32_character_hex_secret_key_here
ENVIRONMENT=development
LOG_LEVEL=INFO
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

#### Install Python Dependencies
```bash
python -m venv venv

# Windows PowerShell:
venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
pip install fastapi uvicorn sqlalchemy psycopg pydantic pydantic-settings python-jose bcrypt passlib
```

---

### 3. Frontend Setup

```bash
cd frontend
pnpm install
```

---

## ▶️ Running the Application

### Start Backend Server
From the root directory with the Python environment active:

```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```
- API Base URL: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/api/health`

### Start Frontend Server
In a separate terminal window:

```bash
cd frontend
pnpm run dev
```
- Web Application: `http://localhost:5173`

---

## 🧪 Testing & Quality Assurance

The codebase includes a dual-layer test suite maintaining strict academic and functional reliability.

### 1. Offline Model & Scientific Preprocessing Tests (143 Tests)
Validates dataset fingerprints, zero-leakage cross-validation, feature consistency, and model integrity without requiring a database:

```bash
python -m pytest -k "test_model or test_phase5 or (test_cdc and not real_dataset)" --ignore=backend/tests -v --tb=short
```
*Status:* **143 PASSED, 3 DESELECTED** ✅

### 2. Full-Stack Backend Integration Suite (163 Tests)
Covers authentication, JWT lifecycle, assessment submission, prediction caching, cross-user isolation, health insights, notifications, and profile management:

```bash
python -m pytest backend/tests/ -v --tb=short
```
*Status:* **163 Tests** covering 13 test files.

### 3. Frontend Production Build Check
```bash
cd frontend
pnpm run build
```
*Status:* **2655 modules transformed, 0 errors** ✅ (Total bundle gzip < 110 kB).

---

## 📡 API Reference

A sample of core endpoints exposed by the backend gateway:

| Method | Endpoint | Description | Auth Required |
|:------:|----------|-------------|:-------------:|
| `POST` | `/api/auth/register` | Register new user account | No |
| `POST` | `/api/auth/login` | Authenticate and obtain JWT Bearer token | No |
| `GET` | `/api/auth/me` | Fetch active user credentials & profile | Yes |
| `POST` | `/api/predict` | Screen 21 clinical variables & calculate risk probabilities | Optional |
| `POST` | `/api/assessments` | Save clinical assessment with predictions | Yes |
| `GET` | `/api/assessments` | Retrieve paginated assessment history | Yes |
| `POST` | `/api/explain` | Calculate XAI feature contribution weights | Yes |
| `GET` | `/api/analytics/trends` | Fetch longitudinal risk probability trajectories | Yes |
| `GET` | `/api/insights/latest` | Retrieve rule-based clinical health recommendations | Yes |
| `GET` | `/api/notifications` | List user clinical reassessment notifications | Yes |
| `PATCH` | `/api/notifications/{id}/read` | Mark individual notification as read | Yes |
| `POST` | `/api/chat` | Send prompt to AI clinical assistant | Yes |
| `GET` | `/api/model/info` | Inspect frozen model metadata, features & hash | No |
| `GET` | `/api/health` | Service liveness and PostgreSQL readiness check | No |

---

## 🐳 Production Deployment with Docker

The system includes preconfigured multi-stage production container definitions and an Nginx reverse proxy.

```bash
# Build and run the complete stack in detached mode:
docker compose -f deploy/docker-compose.prod.yml up --build -d
```

- **Frontend & Web Gateway:** `http://localhost` (Port 80)
- **Backend API:** `http://localhost/api`
- **PostgreSQL:** Internal Docker network (Port 5432)

---

## ⚠️ Medical & Ethical Disclaimer

> **IMPORTANT CLINICAL NOTICE**
> 
> **DiabetesRisk AI is strictly an academic research prototype.**  
> 
> - It is **NOT** a certified medical diagnostic device and is **NOT** approved by the FDA, EMA, or other regulatory authorities.
> - Assessments and probability estimates are generated using self-reported survey statistics from the CDC BRFSS 2015.
> - This tool **DOES NOT** replace clinical laboratory testing (such as Glycated Hemoglobin [HbA1c], Fasting Plasma Glucose [FPG], or Oral Glucose Tolerance Tests [OGTT]).
> - **Never ignore, delay, or modify professional medical diagnosis or treatment** based on the results, insights, or automated feedback generated by this software.
> - Consult a licensed physician or healthcare professional for any health concerns.

---

## 👨‍💻 Author & License

- **Author:** [Ajay Kumar KR](https://github.com/AjayKumarKR07)
- **Repository:** [Advanced-Diabetes-Risk-Prediction-System](https://github.com/AjayKumarKR07/Advanced-Diabetes-Risk-Prediction-System)
- **License:** Open-sourced under the [MIT License](LICENSE).
