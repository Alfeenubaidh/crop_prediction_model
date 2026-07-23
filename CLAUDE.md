# CLAUDE.md — Crop Yield Prediction System

## Project Overview

ML pipeline for crop yield prediction across 5 Indian states using:

* Weather (NASA POWER)
* NDVI
* Soil Organic Carbon
* Historical yield (2011–2022)

Architecture:

* Teacher: Stacking Regressor
* Student: LightGBM (distilled)

**Project root:** `D:/Crop_Yield_System`  
**Python:** 3.10  
**Core code:** `ml/src/`  
**ZenML steps:** `ml/steps/`  
**ZenML pipelines:** `ml/pipelines/`

---

## Directory Structure

```
D:/Crop_Yield_System/
│
├── config.yaml                    # local only (not committed)
├── config.example.yaml            # or conifgs/config.example.yaml
├── requirements.txt
│
├── ml/
│   ├── src/                       # data_splitter, feature_engineering, merge, model_*, etc.
│   ├── steps/                     # ZenML @step functions
│   ├── pipelines/
│   │   ├── training_pipeline.py   # canonical training DAG
│   │   └── deployment_pipeline.py
│   └── ...
│
├── pipelines/
│   └── training_pipeline.py       # mirror of ml/pipelines (same imports from ml.steps)
│
├── scripts/
│   ├── run_pipeline.py
│   └── train_business_model.py
│
├── models/                        # artifacts (joblib, feature_schema.json)
├── data/
└── ...
```

---

## Training pipeline order (leakage-safe)

1. **Ingest** → **Merge** (single panel with `Yield`, `Year`, `State`, `Season`, …).
2. **Split** — `data_split_step`: `DataSplitter.split_full()` → `train_df`, `val_df`, `test_df` **including `Yield`** (needed for lagged yield and row-safe imputation).
3. **Missing values** — `MissingValueHandler.fit(train_df)` then `transform` on train / val / test.
4. **Outliers** — `OutlierHandler.fit(train_df)` then `transform` on each split.
5. **Feature engineering** — `FeatureEngineering.fit` on train (via `fit_transform(train)`), then `transform(val, lag_history=train)` and `transform(test, lag_history=train+val)` so **lags/rolling do not reset** at split boundaries. Step returns `X_*`, `y_*` with target dropped from features.
6. **Train / export / evaluate** — `training_run`, `export_test_set_step`, `evaluation_run`.

Entry point: `python scripts/run_pipeline.py` (imports `ml.pipelines.training_pipeline`).

---

## Core Rules

### No data leakage

* **Fit** only on training rows for: `MissingValueHandler`, `OutlierHandler`, `FeatureEngineering`, and the sklearn `ColumnTransformer` inside `training_run`.
* **Transform** validation and test with those fitted objects only.

### Time split

* No shuffle; split by `Year` via `config.yaml` → `data_split`.
* `DataSplitterConfig` can be built from a loaded `cfg` dict (used by `data_split_step` + `load_config`).

### Schema validation

* Inference must match `models/feature_schema.json` column order from training.

---

## Model

Teacher: Stacking (see `ml/steps/model_training_step.py`)  
Student: LightGBM (deployment artifact `student_lightgbm.joblib`)

---

## Evaluation

* RMSE, R², MAE
* Expected R² (honest time split): ~0.55–0.75
* R² < 0.40 → weak generalisation; R² > 0.85 → check leakage / overfitting

---

## Dataset constraint

~180 rows → high metric variance; interpret with caution.

---

# Execution protocol

## Role

Act as ML engineer + debugger.

## Modes

PLAN → analysis only  
ACT → implement changes

## Debug order

1. Data  
2. Features  
3. Model  
4. Pipeline  

## Debug triggers

* RMSE very high or R² very low → investigate data and splits first  
* Prefer simple fixes; respect small-sample limits  

## Goal

Production-ready, leakage-safe ML system
