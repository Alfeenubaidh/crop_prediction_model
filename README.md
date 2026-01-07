# 🌾 Crop Yield Prediction System  
**An Explainable, Production-Oriented Machine Learning Pipeline Using Earth Observation Data**

---

## Overview

This project implements an **end-to-end, production-ready machine learning system** for **regional crop yield prediction** using:

- satellite-derived vegetation indices (NDVI),
- climate variables (temperature, rainfall, humidity, wind),
- soil organic carbon (SOC),
- and historical yield dynamics.

Unlike notebook-centric ML experiments, this system is designed around the **complete ML lifecycle**, with explicit emphasis on:

- reproducibility,
- feature parity between training and inference,
- explainability,
- and deployment readiness.

The system is inspired by **operational Earth science ML practices** (e.g., NASA-style workflows), prioritizing **scientific validity and interpretability** over speculative forecasting.

---

## Why This Project Matters

Many crop-yield ML studies stop at:

> *“The model trains.”*

This project goes further:

> *“The system predicts, explains, validates assumptions, and can be deployed safely.”*

It is designed as a **decision-support system** rather than a black-box predictor, making it suitable for **real-world agricultural analysis and policy-facing applications**.

---

## Intended Use and Real-World Applicability

This system is intended to support:

- agricultural extension agencies,
- policy and planning teams,
- crop insurance and agri-finance analysts,
- climate–yield researchers.

Given observed vegetation, climate, and soil conditions for a growing season, the system:

- estimates expected crop yield,
- identifies dominant environmental drivers affecting yield,
- supports post-season assessment and early risk analysis.

The system **explicitly avoids extrapolating beyond observed environmental data**, prioritizing trustworthiness and scientific discipline.

---

## Key Contributions

### 1. Multi-Source Data Integration
- Satellite-based NDVI (vegetation health)
- Climate variables (temperature, rainfall, humidity, wind)
- Soil organic carbon (SOC)
- Historical yield records

All data sources are harmonized temporally and spatially.

---

### 2. Domain-Aware Feature Engineering

Feature engineering incorporates agronomic knowledge rather than raw signals alone, including:

- NDVI–climate interaction terms
- NDVI–SOC hybrid indicators
- Vegetation stress and anomaly features
- Temporal lag features
- Rolling window statistics (multi-year context)

This reduces reliance on model memorization and improves interpretability.

---

### 3. Explainable Machine Learning (SHAP)

Explainability is treated as a **first-class system component**:

- Feature-level contribution analysis
- Case-based explanations for individual predictions
- Top contributing environmental drivers per inference

This enables **transparent interpretation**, critical for agricultural and policy-facing systems.

---

### 4. Production-Oriented System Design

The system follows production ML best practices:

- Configuration-driven execution
- Modular pipeline architecture
- Clear separation of ingestion, features, modeling, inference, and explanation
- Schema validation at API boundaries
- Lazy loading of heavy components (models, explainers)

---

## Prediction Scope and Temporal Constraints

Although the model architecture supports inference for arbitrary years, **predictions are restricted to years with observed NDVI and climate data**.

This design choice is intentional.

- NDVI observations are available for **2011–2022**
- Climate variables align with the same temporal window
- Tree-based models (e.g., LightGBM) do not extrapolate reliably without future covariates

To avoid silent feature drift and misleading outputs, the system enforces a hard constraint on prediction years.  
Future-year predictions require either:

- projected climate inputs, or
- scenario-based NDVI estimates.

This reflects **best practices in operational Earth observation ML systems**.

---

## System Architecture (High Level)

Raw Data
├── NDVI (Satellite)
├── Climate Variables
├── Soil Data
└── Yield Records
↓
Data Ingestion & Validation
↓
Feature Engineering
↓
Preprocessing Pipeline
↓
Trained ML Model
↓
FastAPI Inference Service
↓
Prediction + SHAP Explanation

Yield Prediction

POST /predict?explain=false

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


POST /predict?explain=true

{
  "predicted_yield": 3.66,
  "explanation": {
    "NDVI_SeasonalMean": 0.45,
    "Rain_anomaly": -0.12,
    "Mean_SOC_NDVI_Hybrid": 0.08
  }
}

