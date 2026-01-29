# 🌾 Agri Yield Intelligence Platform
**Scenario-Based, Explainable Crop Yield Forecasting System**

---

## Overview

The **Agri Yield Intelligence Platform** is an **explainable decision-support system**
for estimating regional crop yield under observed or scenario-based environmental
conditions.

The system combines:
- satellite-derived vegetation indicators,
- climate variables,
- soil quality information,
- and historical yield patterns

to produce **transparent, interpretable yield estimates** suitable for
agricultural planning and risk assessment.

This repository represents a **business-oriented prototype** derived from a
research pipeline, demonstrating how explainable machine learning can support
real-world agricultural decisions.

---

## What This System Does

The platform allows users to:

- estimate expected crop yield for a region,
- explore how yield responds to environmental changes,
- understand *why* the model produces a prediction,
- compare baseline and stress scenarios,
- support planning under climate variability.

The system is designed for **post-season assessment, early-season monitoring,
and scenario exploration**, not speculative long-term forecasting.

---

## Key Capabilities

### 🌱 Scenario-Based Forecasting
Users can adjust:
- temperature,
- rainfall,
- solar radiation,
- vegetation indices,
- soil organic carbon

to evaluate **“what-if” yield outcomes** under different environmental conditions.

---

### 🔍 Explainable Predictions
Explainability is built into the core workflow:

- Feature-level contribution analysis (SHAP)
- Identification of dominant yield drivers
- Transparent, auditable predictions

This enables trust, debugging, and stakeholder communication.

---

### 📊 Interactive Dashboard
A Streamlit dashboard provides:

- intuitive input controls,
- real-time predictions,
- uncertainty awareness,
- side-by-side scenario comparison.

---

### ⚙️ API-First Architecture
A FastAPI service exposes the inference logic, enabling:
- integration with other systems,
- future automation,
- decoupled frontend development.

---

## System Architecture

User / Analyst
│
▼
Streamlit Dashboard
│
▼
FastAPI Inference Service
│
▼
Trained ML Model + Explainability Engine
---

## Running the System Locally

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt

2️⃣ Start the API
uvicorn api.main:app --reload

3️⃣ Launch the Dashboard
streamlit run dashboard_business/dashboard.py

Example API Request (Illustrative)

POST /predict?explain=true

{
  "climate": {
    "avg_temperature": 26,
    "max_temperature": 35,
    "min_temperature": 20,
    "total_rainfall": 600,
    "solar_radiation": 20,
    "relative_humidity": 65,
    "wind_speed": 2.5
  },
  "soil": {
    "soil_organic_carbon": 0.7
  },
  "vegetation": {
    "ndvi_early": 0.6
  }
}
Example Response
{
  "prediction": {
    "expected_yield": 3.37,
    "yield_lower": 3.04,
    "yield_upper": 3.71,
    "risk_level": "Low",
    "confidence": "High"
  },
  "explanation": {
    "soil_organic_carbon": 0.61,
    "wind_speed": -0.20,
    "total_rainfall": -0.08
  }
}
