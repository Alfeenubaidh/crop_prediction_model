# Scenario Robustness Experiments

## Objective
The objective of these experiments is to evaluate the robustness of the proposed crop yield forecasting system under controlled, realistic environmental perturbations. The goal is to determine whether the model exhibits stable and physically plausible behavior when input conditions deviate from nominal values, and to assess whether scenario-based analysis reveals sensitivities not observable through point predictions alone.

---

## Hypothesis
The proposed system will exhibit bounded and interpretable changes in predicted crop yield under realistic environmental perturbations, without exhibiting unstable or non-physical behavior.

---

## Selected Scenario Variables
Scenario-based perturbations are limited to the following three variables to maintain interpretability and experimental control:

1. **Total Rainfall**
2. **Average Temperature**
3. **Vegetation Index (NDVI)**

These variables are selected due to their known relevance to crop growth processes and their frequent use in prior crop yield modeling studies.

---

## Perturbation Strategy
Each scenario variable is perturbed independently while all other inputs are held constant at baseline values. Perturbations are applied symmetrically around baseline conditions to simulate both favorable and adverse environmental scenarios.

Perturbation ranges are constrained to values that are extreme but plausible within the historical data distribution to avoid extrapolation beyond the model’s intended operating domain.

---

## Evaluation Procedure
For each perturbed input configuration:

- Yield predictions are generated using the trained model.
- Changes in predicted yield are measured relative to the baseline (unperturbed) prediction.
- Performance metrics under perturbation are compared to baseline metrics.

No retraining is performed during scenario evaluation.

---

## Robustness Metrics
Robustness is assessed using the following criteria:

- Relative change in predicted yield with respect to perturbation magnitude
- Variance of prediction error under perturbed inputs
- Presence or absence of discontinuities in yield response curves

---

## Sensitivity Analysis
Yield response curves are generated for each scenario variable by sweeping the variable across its defined perturbation range. These curves are used to identify:

- Regions of high sensitivity
- Saturation effects
- Nonlinear response behavior

---

## Monotonicity and Plausibility Checks
Model responses are evaluated for monotonic consistency where domain knowledge suggests a directional relationship (e.g., extreme rainfall or temperature stress).

Violations of expected monotonic trends are recorded and analyzed rather than corrected.

---

## Stress Testing
In addition to individual perturbations, a limited set of combined stress scenarios is evaluated to assess compound effects (e.g., high temperature combined with low rainfall). These scenarios are exploratory and not used for model tuning.

---

## Expected Failure Modes
The following failure modes are explicitly monitored:

- Excessive sensitivity to small perturbations
- Non-monotonic or discontinuous yield responses
- Unrealistically large yield changes under plausible conditions

Observed failure cases are documented and discussed.

---

## Scope Limitations
These experiments do not aim to simulate long-term climate change scenarios or establish causal relationships between environmental variables and yield outcomes. Scenario analysis is intended solely as a tool for robustness assessment and decision-support exploration.

---

## Experimental Status
This document defines the scenario robustness evaluation protocol prior to execution. No experimental results are included at this stage. All scenario analyses are conducted according to the procedures specified here to ensure consistency and reproducibility.
