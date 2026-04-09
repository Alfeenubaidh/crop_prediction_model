# Baseline Experiments

## Objective
The objective of these experiments is to evaluate the predictive performance and robustness of the proposed crop yield forecasting system relative to standard baseline models. The comparison is designed to assess whether the proposed approach provides competitive accuracy while maintaining stability under environmental variability, rather than to establish state-of-the-art performance.

---

## Hypothesis
The proposed machine learning system will achieve predictive performance within the range of standard baseline models while exhibiting improved robustness to controlled environmental perturbations.

---

## Baseline Models
The following baseline models are used for comparison:

### Historical Mean Baseline
Predicts crop yield using the mean of historical yields for the corresponding region and season.

### Linear Regression
A multivariate linear regression model trained on the same input features as the proposed system.

### Random Forest Regressor
An ensemble-based nonlinear regression model commonly used in crop yield forecasting tasks.

These baselines are selected to represent increasing levels of model complexity and are commonly employed in prior crop yield prediction studies.

---

## Data and Splitting Strategy
The dataset is partitioned into training, validation, and test sets using a temporally consistent split to prevent information leakage. Models are trained on historical seasons and evaluated on held-out seasons to simulate real-world deployment conditions.

Where applicable, no spatial overlap between training and test regions is permitted.

---

## Evaluation Metrics
Model performance is evaluated using the following metrics:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

Metrics are computed on the test set and reported as aggregate statistics across evaluation periods.

---

## Robustness Evaluation
To assess robustness, controlled perturbations are applied to selected environmental input variables, including rainfall, temperature, and vegetation indices. Performance degradation under these perturbations is measured relative to unperturbed inputs.

Robustness is quantified as the relative change in evaluation metrics under perturbed inputs compared to baseline inputs.

---

## Expected Failure Modes
The following failure modes are anticipated and explicitly monitored:

- Underfitting of nonlinear relationships by linear regression models
- Sensitivity of ensemble-based models to feature distribution shifts
- Degradation of predictive accuracy under extreme but plausible environmental conditions

Observed failure cases are documented rather than suppressed.

---

## Reproducibility Considerations
All experiments are conducted with fixed random seeds. Model configurations, data splits, evaluation procedures, and experimental parameters are logged to ensure reproducibility.

---

## Scope Limitations
These experiments do not attempt to establish superiority over all existing crop yield prediction models. The focus is on relative performance, robustness, and suitability within a production-oriented decision-support context.

---

## Non-Claims
- These experiments do not claim causal interpretation of feature effects.
- These experiments do not establish generalization beyond the evaluated regions and seasons.

---

## Experimental Status
At the time of writing, this document specifies experimental intent and evaluation protocol only. No experimental results are included. All analyses are performed following the procedures defined herein to prevent post-hoc result selection.
