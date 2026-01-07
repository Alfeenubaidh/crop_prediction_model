# Crop Yield Prediction System  
**An Explainable, Production-Oriented Machine Learning Pipeline using Earth Observation Data**

## Overview

This project implements an **end-to-end, production-ready machine learning system** for **regional crop yield prediction** using:

- satellite-derived vegetation indices (NDVI),
- climate variables,
- soil organic carbon (SOC),
- and temporal yield history.

Unlike typical experimental ML notebooks, this system is designed around the **full ML lifecycle**, with a focus on **reproducibility, explainability, and deployment readiness**.

The project is inspired by **NASA-style Earth science ML lifecycle practices**, emphasizing interpretability and operational robustness.

---

## Why This Project Matters

Most crop-yield ML studies stop at:
> *“The model trains.”*

This system goes further:
> *“The system predicts, explains, and can be deployed.”*

It is designed for **real-world agricultural decision support**, not just offline experimentation.

---

## Key Contributions

### 1. Multi-Source Data Integration
- Climate variables (temperature, rainfall, humidity, wind)
- Satellite-based NDVI
- Soil organic carbon (SOC)
- Historical crop yield

### 2. Domain-Aware Feature Engineering
- NDVI–climate interaction terms  
- Soil–vegetation hybrid features  
- Stress and anomaly indicators  
- Temporal lag and rolling statistics  

These features encode **agronomic knowledge**, not just raw signals.

### 3. Explainable Machine Learning (SHAP)
- Global feature importance
- Group-level agronomic interpretation
- Case-based explanations for individual years or regions

Explainability is treated as a **first-class system component**, not an afterthought.

### 4. Production-Oriented Design
- Configuration-driven execution
- Modular pipeline structure
- Clear separation of ingestion, features, modeling, inference, and explanation
- Reproducible runs across environments


