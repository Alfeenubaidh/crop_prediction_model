# 🌾 Crop Yield Prediction System  
**An Explainable, Research-First Machine Learning Pipeline Using Earth Observation Data**

---

## Repository Positioning (Important)

This repository is a **research-grade machine learning project** focused on
crop yield prediction using Earth observation data.

While the repository contains **API and dashboard components**, these are
**reference implementations only**, included to demonstrate how the research
outputs *can* be operationalized.

> ⚠️ **This repository is NOT a commercial product.**  
> Production deployment, scalability, security, and business logic are
> intentionally out of scope.

Future commercial systems should implement **independent APIs, dashboards,
and infrastructure**.

---

## Overview

This project implements an **end-to-end, production-oriented research pipeline**
for **regional crop yield prediction** using:

- Satellite-derived vegetation indices (NDVI)
- Climate variables (temperature, rainfall, humidity, wind)
- Soil Organic Carbon (SOC)
- Historical yield observations

Unlike notebook-only experiments, the system is designed around the **full
machine learning lifecycle**, with emphasis on:

- reproducibility,
- strict feature parity between training and inference,
- explainability,
- and operational feasibility.

The project follows **Earth observation ML best practices**, prioritizing
**scientific validity and interpretability** over speculative forecasting.

---

## Research Questions

This research investigates:

1. How effectively can seasonal NDVI and climate variables predict regional crop yield?
2. Do NDVI–climate interaction features outperform climate-only baselines?
3. How does soil organic carbon (SOC) influence vegetation–yield relationships?
4. Can explainable ML methods (SHAP) produce agronomically meaningful insights?

---

## Why This Project Matters

Many crop-yield ML studies stop at:

> *“The model trains.”*

This project goes further:

> *“The system predicts, explains, validates assumptions, and can be safely operationalized.”*

It is designed as a **decision-support system**, not a black-box predictor,
making it suitable for **agricultural analysis, research, and policy-facing work**.

---

## Intended Use

This system is intended for:

- agricultural research,
- extension and advisory analysis,
- policy and planning studies,
- climate–yield impact assessment.

Given observed vegetation, climate, and soil conditions for a growing season,
the system:

- estimates expected crop yield,
- identifies dominant environmental drivers,
- supports post-season assessment and early risk analysis.

The system **does not extrapolate beyond observed environmental data**, ensuring
trustworthy and scientifically disciplined outputs.

---

## Key Contributions

### 1. Multi-Source Data Integration
- NDVI (satellite vegetation health)
- Climate variables
- Soil Organic Carbon (SOC)
- Historical yield records

All sources are harmonized temporally and spatially.

---

### 2. Domain-Aware Feature Engineering

Feature engineering incorporates agronomic reasoning, including:

- NDVI–climate interaction features
- NDVI–SOC hybrid indicators
- Vegetation stress and anomaly metrics
- Temporal lag features
- Rolling window statistics

This improves interpretability and reduces overfitting.

---

### 3. Explainable Machine Learning (SHAP)

Explainability is treated as a **first-class component**:

- Feature-level contribution analysis
- Case-based explanations for individual predictions
- Identification of dominant environmental drivers

This transparency is critical for agricultural and policy-facing systems.

---

### 4. Production-Oriented Research Design

The system follows production ML principles without being a deployed product:

- Modular pipeline design
- Configuration-driven execution
- Clear separation of ingestion, features, modeling, inference, and explanation
- Schema validation at inference boundaries
- Lazy loading of heavy components (models, explainers)

---

## Prediction Scope and Temporal Constraints

Predictions are restricted to years with **observed NDVI and climate data**.

- NDVI observations: **2011–2022**
- Climate variables aligned to the same period

This constraint is intentional:

- Tree-based models do not extrapolate reliably without future covariates
- Prevents silent feature drift and misleading predictions

Future-year predictions require either:
- projected climate inputs, or
- scenario-based NDVI estimates.

---

## Evaluation Methodology

Models are evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score

Temporal splits are used to prevent data leakage.
Baseline comparisons include climate-only and NDVI-only models, with ablation
studies assessing the contribution of engineered features.

---

## Repository Structure

The repository separates **research artifacts** from **reference system components**.

### Research Components
- `notebook/` — Research notebooks and experiments
- `evaluation/` — Model evaluation and metrics
- `steps/` — Modular, research-oriented pipeline steps
- `pipelines/` — Reproducible research pipelines

### Core ML Logic
- `src/` — Feature engineering, NDVI processing, explainability
- `data/` — Sample and processed datasets (no proprietary raw data)
- `models/` — Research model artifacts (not production models)

### Reference Implementations
- `api/` — Reference FastAPI inference service
- `dashboard/` — Demonstration dashboard for visualization

> The `api/` and `dashboard/` directories are **illustrative only** and are not
> intended for real-world production deployment.

---

## API Reference Example (Illustrative Only)

The following example demonstrates the **inference contract and output schema**.

**POST** `/predict?explain=false`

```json
{
  "state": "Punjab",
  "district": "Ludhiana",
  "crop": "Wheat",
  "season": "Rabi",
  "year": 2021
}

{
  "predicted_yield": 3.66
}

{
  "predicted_yield": 3.66,
  "explanation": {
    "NDVI_SeasonalMean": 0.45,
    "Rain_anomaly": -0.12,
    "Mean_SOC_NDVI_Hybrid": 0.08
  }
}

---
