# Crop Yield Prediction System

End-to-end machine learning system for regional crop yield prediction across
5 Indian states, combining satellite vegetation data (NDVI), climate variables,
soil organic carbon, and historical yield records.

**Model performance (held-out test set, 2021–2022)**

| Metric | Value |
|--------|-------|
| R² | 0.9145 |
| RMSE | 0.233 |
| MAE | 0.173 |

Conformal prediction intervals: ±0.317 (80%), ±0.523 (90%), ±0.622 (95%).

---

## Architecture

```
frontend/          React + Vite UI (Firebase auth, prediction form, dashboard)
    │
    │  POST /predict
    ▼
api.py             FastAPI inference service (project root)
    │
    ▼
ml/src/            Core ML — feature engineering, inference, explainability
models/            Trained artifacts (student_lightgbm, preprocessor, FE)
```

**Model design**
- Teacher: Stacking Regressor (RandomForest + GradientBoosting + Lasso + ElasticNet → LinearRegression meta)
- Student: LightGBM (distilled from teacher, deployed artifact)
- Training split: 2011–2018 train / 2019–2020 val / 2021–2022 test (temporal, no shuffle)

---

## Repository Layout

```
Crop_Yield_System/
├── api.py                        # FastAPI app — /health /metrics /predict /predict/batch
├── requirements_api.txt          # Deps to run api.py
├── requirements.txt              # Full ML pipeline deps
├── conifgs/
│   └── config.example.yaml       # Copy to config.yaml and fill in local paths
│
├── ml/
│   ├── src/
│   │   ├── feature_engineering.py   # Leakage-safe FE (fit on train only)
│   │   ├── inference/
│   │   │   └── predict.py           # run_prediction() — core inference function
│   │   ├── explainer/               # SHAP explainability
│   │   └── ...                      # ingest, merge, splitter, outlier, etc.
│   ├── steps/                       # ZenML @step functions
│   └── pipelines/
│       └── training_pipeline.py     # Canonical training DAG
│
├── models/                          # Trained artifacts (committed)
│   ├── student_lightgbm.joblib
│   ├── preprocessor.joblib
│   ├── feature_engineering.joblib
│   └── conformal_quantiles.json
│
├── frontend/                        # React + Vite + Firebase
│   └── src/
│       ├── App.tsx
│       └── services/predictionService.ts
│
├── scripts/
│   └── run_pipeline.py              # Entry point: python scripts/run_pipeline.py
│
├── evaluation/
│   ├── evaluation_report.json
│   └── evaluation_predictions.csv
│
└── data/
    ├── raw/                         # NASA POWER weather, NDVI, SOC, yield CSV
    └── Processed/                   # Merged panel dataset
```

---

## Quickstart

### 1. Install API dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Start the inference API

```bash
uvicorn api:app --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok","artifacts":{"model":true,"encoder":true,"feature_engineering":true}}
```

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

The frontend reads `VITE_API_URL` from `frontend/.env.local`:

```
VITE_API_URL=http://localhost:8000
```

---

## API

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

All climate fields are optional — missing values are filled from training-set
group means by the fitted `FeatureEngineering` object.

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

## Training Pipeline

Requires `config.yaml` (copy from `conifgs/config.example.yaml` and set local
paths to raw data).

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Pipeline order (leakage-safe):

1. **Ingest** — NASA POWER weather, NDVI, SOC, yield CSV
2. **Merge** — join all sources into a single panel
3. **Split** — temporal split by year (no shuffle)
4. **Missing values** — fit on train, transform all splits
5. **Outliers** — IQR-groupwise, fit on train only
6. **Feature engineering** — fit on train, `lag_history`-aware transform for val/test
7. **Train** — teacher stacking → student LightGBM distillation
8. **Evaluate** — RMSE / R² / MAE + conformal intervals on test set

---

## Supported States

| State | Season |
|-------|--------|
| Punjab | Rabi, Kharif |
| Haryana | Rabi, Kharif |
| Rajasthan | Rabi, Kharif |
| Uttar Pradesh | Rabi, Kharif |
| Chandigarh | Reference only — insufficient test rows |

Data range: **2011–2022** (~180 panel rows). Metric variance is high at this
sample size; interpret R² with caution.

---

## Feature Engineering Highlights

- NDVI × climate interaction terms (NDVI_Rain, NDVI_Temp, NDVI_Rain_Hybrid)
- NDVI and Rain anomalies relative to group (State × Season) medians
- SOC × NDVI hybrid and stress indicators
- Season one-hot encoding (Kharif / Rabi / Zaid)
- Temporal lag features: Yield_Lag1/2, NDVI_Lag1/2, Rain_Lag1/2
- Rolling statistics (3-year window) for yield, NDVI, rainfall
- Compound stress flag (NDVI and Rain both below group median)

All statistics used in feature engineering (group medians, SOC max, rain max)
are fitted on the training split only and serialised to
`models/feature_engineering.joblib`.

---

## Citation

See `CITATION.cff`.
