# AgroPredict — Crop Yield Prediction System

> End-to-end machine learning system for regional crop yield prediction across 5 Indian states, combining satellite vegetation data (NDVI), NASA POWER climate variables, soil organic carbon, and historical yield records (2011–2022).

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?logo=vercel)](https://crop-prediction-model-seven.vercel.app)
[![API](https://img.shields.io/badge/API-Render-46E3B7?logo=render)](https://crop-prediction-model.onrender.com/docs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

---

## Live Demo

| Service | URL |
|---------|-----|
| Frontend | https://crop-prediction-model-seven.vercel.app |
| REST API (Swagger UI) | https://crop-prediction-model.onrender.com/docs |
| API Health | https://crop-prediction-model.onrender.com/health |

---

## Features

- **Yield prediction** — single-row and batch (up to 50 rows) predictions with a single API call
- **Conformal prediction intervals** — statistically valid uncertainty bounds at 80 %, 90 %, and 95 % coverage
- **Risk classification** — automatic Low / Medium / High risk labelling based on interval width
- **SHAP explainability** — feature-importance values for every prediction
- **Leakage-safe pipeline** — all preprocessing (missing-value imputation, outlier handling, feature engineering) is fitted exclusively on the training split
- **Temporal feature engineering** — lag and rolling-window features that respect split boundaries
- **Knowledge distillation** — compact LightGBM student distilled from a Stacking Regressor teacher for fast inference
- **Firebase authentication** — Google Sign-In; every prediction is persisted to Firestore per user
- **Prediction history dashboard** — time-series charts of past predictions per authenticated user
- **Admin panel** — aggregated platform metrics visible to admin role accounts

---

## Tech Stack

### Frontend
| Layer | Technology |
|-------|-----------|
| UI framework | React 19 + TypeScript |
| Build tool | Vite 6 |
| Styling | Tailwind CSS 4 |
| Animations | Motion (Framer Motion) |
| Charts | Recharts |
| Auth & database | Firebase 12 (Google Auth + Firestore) |
| Deployment | Vercel |

### Backend
| Layer | Technology |
|-------|-----------|
| API framework | FastAPI 0.95 + Uvicorn |
| ML runtime | scikit-learn 1.1, LightGBM 3.3, XGBoost 1.6 |
| Data | pandas 2.0, NumPy 1.24 |
| Explainability | SHAP |
| Uncertainty | Conformal prediction (custom) |
| Config | PyYAML |
| Deployment | Render (free tier) |

### ML Pipeline
| Component | Technology |
|-----------|-----------|
| Orchestration | ZenML |
| Data sources | NASA POWER API, NDVI (satellite), SOC dataset |
| Teacher model | Stacking Regressor (RandomForest + GradientBoosting + Lasso + ElasticNet → LinearRegression meta) |
| Student model | LightGBM (deployed artifact) |
| Training split | Temporal — 2011–2018 train / 2019–2020 val / 2021–2022 test |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                         │
│         React + Vite  (Vercel — crop-prediction-model-      │
│                         seven.vercel.app)                   │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Home / Hero │  │ Predict Form │  │    Dashboard     │  │
│  └──────────────┘  └──────┬───────┘  └──────────────────┘  │
│                            │ POST /predict                   │
└────────────────────────────┼────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  FastAPI (Render)│
                    │  backend/api.py  │
                    │                 │
                    │  /predict        │
                    │  /predict/batch  │
                    │  /health         │
                    │  /metrics        │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │      ML Inference Layer      │
              │  backend/ml/src/inference/   │
              │                             │
              │  FeatureEngineering.joblib   │
              │  preprocessor.joblib         │
              │  student_lightgbm.joblib     │
              │  conformal_quantiles.json    │
              └─────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │      Firebase (Google)       │
              │  Authentication (Google SSO) │
              │  Firestore (prediction logs) │
              └─────────────────────────────┘
```

**Model design — teacher → student distillation**

```
Training data (2011–2018)
        │
        ▼
 Stacking Regressor  ◄── Teacher (high accuracy, slow)
  ├─ RandomForest
  ├─ GradientBoosting
  ├─ Lasso
  └─ ElasticNet
        │ soft labels
        ▼
   LightGBM          ◄── Student (deployed, fast)
        │
        ▼
  Conformal wrapper  ◄── Calibrated on val set (2019–2020)
        │
        ▼
  /predict response (point estimate + intervals + risk)
```

---

## Model Performance

Evaluated on the held-out test set (2021–2022, ~30 rows).

| Metric | Value |
|--------|-------|
| R² | 0.9145 |
| RMSE | 0.233 t/ha |
| MAE | 0.173 t/ha |

Conformal prediction intervals (calibrated on 2019–2020 val set):

| Coverage | Half-width |
|----------|-----------|
| 80 % | ±0.317 t/ha |
| 90 % | ±0.523 t/ha |
| 95 % | ±0.622 t/ha |

> **Note:** ~180 total panel rows; metric variance is high at this sample size. Interpret R² with caution.

---

## Repository Layout

```
Crop_Yield_System/
├── Procfile                          # Render entry point (uvicorn backend.api:app)
├── render.yaml                       # Render service config
├── build.sh                          # Fetches model artifacts from remote storage
│
├── backend/
│   ├── api.py                        # FastAPI app — /health /metrics /predict /predict/batch
│   ├── requirements.txt              # Full ML + API dependencies
│   ├── conifgs/
│   │   └── config.example.yaml       # Copy → config.yaml and fill local paths
│   │
│   ├── ml/
│   │   ├── src/
│   │   │   ├── feature_engineering.py    # Leakage-safe FE (fit on train only)
│   │   │   ├── inference/predict.py      # run_prediction() — core inference
│   │   │   └── explainer/                # SHAP explainability
│   │   ├── steps/                        # ZenML @step functions
│   │   └── pipelines/training_pipeline.py
│   │
│   ├── models/                       # Trained artifacts (not committed — fetched by build.sh)
│   │   ├── student_lightgbm.joblib
│   │   ├── preprocessor.joblib
│   │   ├── feature_engineering.joblib
│   │   └── conformal_quantiles.json
│   │
│   ├── scripts/run_pipeline.py       # Entry point for training
│   └── evaluation/
│       ├── evaluation_report.json
│       └── evaluation_predictions.csv
│
└── frontend/
    └── src/
        ├── App.tsx                   # All UI — Home, Predict, Dashboard, Admin
        ├── firebase.ts               # Firebase initialisation
        ├── firebase-applet-config.json
        └── services/predictionService.ts   # API client
```

---

## Running Locally

### Prerequisites

- Python 3.10
- Node.js 18+
- A Firebase project with Google Auth and Firestore enabled

---

### 1. Clone the repository

```bash
git clone https://github.com/Alfeenubaidh/crop_prediction_model.git
cd crop_prediction_model
```

### 2. Backend — install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Backend — set up config and model artifacts

Copy the example config and fill in your local data paths:

```bash
cp backend/conifgs/config.example.yaml config.yaml
```

Place trained model artifacts in `backend/models/`:

```
backend/models/
├── student_lightgbm.joblib
├── preprocessor.joblib
├── feature_engineering.joblib
└── conformal_quantiles.json
```

If you have the `MODELS_ZIP_URL` environment variable set, `build.sh` fetches them automatically:

```bash
MODELS_ZIP_URL=<your-url> bash build.sh
```

### 4. Backend — start the API

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Verify it is running:

```bash
curl http://localhost:8000/health
# {"status":"ok","artifacts":{"model":true,"encoder":true,"feature_engineering":true}}
```

Interactive API docs are available at http://localhost:8000/docs.

### 5. Frontend — install dependencies

```bash
cd frontend
npm install
```

### 6. Frontend — configure environment

Create `frontend/.env.local`:

```env
VITE_API_URL=http://localhost:8000
```

Place your Firebase web config in `frontend/src/firebase-applet-config.json`:

```json
{
  "projectId": "YOUR_PROJECT_ID",
  "appId": "YOUR_APP_ID",
  "apiKey": "YOUR_API_KEY",
  "authDomain": "YOUR_AUTH_DOMAIN",
  "firestoreDatabaseId": "YOUR_FIRESTORE_DB_ID",
  "storageBucket": "YOUR_STORAGE_BUCKET",
  "messagingSenderId": "YOUR_SENDER_ID",
  "measurementId": ""
}
```

### 7. Frontend — start the dev server

```bash
npm run dev
# http://localhost:3000
```

---

### (Optional) Re-train the model

Requires raw data files configured in `config.yaml`:

```bash
python backend/scripts/run_pipeline.py
```

Pipeline stages (leakage-safe, temporal order):

1. **Ingest** — NASA POWER weather, NDVI, SOC, yield CSV
2. **Merge** — join all sources into a single panel
3. **Split** — temporal split by year (no shuffle)
4. **Missing values** — fit on train, transform all splits
5. **Outliers** — IQR group-wise, fit on train only
6. **Feature engineering** — fit on train; lag-aware transform for val / test
7. **Train** — teacher stacking → student LightGBM distillation
8. **Evaluate** — RMSE / R² / MAE + conformal intervals on test set

---

## API Reference

### `POST /predict`

```json
{
  "state":       "PUNJAB",
  "year":        2022,
  "season":      "Rabi",
  "T2M":         18.5,
  "T2M_MAX":     26.0,
  "T2M_MIN":     10.0,
  "PRECTOTCORR": 95.0,
  "RH2M":        62.0,
  "WS2M":        2.1,
  "ALLSKY_SFC_SW_DWN": 14.3
}
```

Climate fields are optional — missing values are filled from training-set group means.

**Response**

```json
{
  "state": "PUNJAB",
  "year": 2022,
  "season": "Rabi",
  "predicted_yield": 4.9991,
  "intervals": [
    {"coverage": 0.80, "lower": 4.68, "upper": 5.32, "half_width": 0.3166},
    {"coverage": 0.90, "lower": 4.48, "upper": 5.52, "half_width": 0.5232},
    {"coverage": 0.95, "lower": 4.38, "upper": 5.62, "half_width": 0.6224}
  ],
  "risk_level": "Medium",
  "model": "student_lightgbm"
}
```

### `POST /predict/batch`

Same schema as `/predict`, body is a JSON array (max 50 rows).

### `GET /health`

Returns artifact availability.

### `GET /metrics`

Returns the latest `evaluation/evaluation_report.json`.

---

## Supported States & Seasons

| State | Seasons |
|-------|---------|
| Punjab | Rabi, Kharif |
| Haryana | Rabi, Kharif |
| Rajasthan | Rabi, Kharif |
| Uttar Pradesh | Rabi, Kharif |

Data range: 2011–2022.

---

## Citation

```bibtex
@software{alfeen2026agropredict,
  author  = {Alfeen, Ubaidh},
  title   = {Explainable Crop Yield Prediction Using Earth Observation Data},
  year    = {2026},
  url     = {https://github.com/Alfeenubaidh/crop_prediction_model},
  license = {Apache-2.0}
}
```

---

## License

Apache 2.0 — see [CITATION.cff](CITATION.cff) for full details.
