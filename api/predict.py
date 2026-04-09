from typing import Dict
from pathlib import Path

from api.schemas import (
    BusinessInputRequest,
    YieldPredictionResponse
)
from api.business.feature_builder import BusinessFeatureBuilder
from api.models.model_loader import ModelLoader
from src.explainer.shap_explainer_buisness import compute_shap_values


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# Feature names (MUST match training & feature builder order)
# ============================================================
FEATURE_NAMES = [
    "avg_temperature",
    "max_temperature",
    "min_temperature",
    "total_rainfall",
    "solar_radiation",
    "relative_humidity",
    "wind_speed",
    "soil_organic_carbon",
    "ndvi_early",
]


# ============================================================
# Lazy singletons
# ============================================================
_feature_builder = None
_model = None


def _get_feature_builder() -> BusinessFeatureBuilder:
    global _feature_builder
    if _feature_builder is None:
        _feature_builder = BusinessFeatureBuilder()
    return _feature_builder


def _get_model():
    global _model
    if _model is None:
        _model = ModelLoader(
            model_path=str(MODEL_DIR / "student_lightgbm.joblib")
        ).load_model()
    return _model


# ============================================================
# Core prediction function
# ============================================================
def predict_yield(payload: Dict, explain: bool = False) -> Dict:
    """
    Business-grade, scenario-based yield prediction.
    Independent of calendar year.
    """

    # --------------------------------------------------------
    # 1️⃣ Validate request
    # --------------------------------------------------------
    request = BusinessInputRequest(**payload)

    # --------------------------------------------------------
    # 2️⃣ Build business features (NumPy array: shape = (1, 9))
    # --------------------------------------------------------
    X = _get_feature_builder().build(request)

    # --------------------------------------------------------
    # 3️⃣ Predict yield
    # --------------------------------------------------------
    model = _get_model()
    y_pred = float(model.predict(X)[0])

    # --------------------------------------------------------
    # 4️⃣ Simple uncertainty band (±10%)
    # --------------------------------------------------------
    lower = y_pred * 0.9
    upper = y_pred * 1.1

    # --------------------------------------------------------
    # 5️⃣ Risk classification (transparent rules)
    # --------------------------------------------------------
    risk = "Low"
    if request.climate.max_temperature > 35 or request.climate.total_rainfall < 300:
        risk = "Medium"
    if request.climate.max_temperature > 40:
        risk = "High"

    # --------------------------------------------------------
    # 6️⃣ Optional SHAP explainability
    # --------------------------------------------------------
    explanation = None
    if explain:
        explanation = compute_shap_values(
            X_encoded=X,
            model_path=str(MODEL_DIR / "student_lightgbm.joblib"),
            feature_names=FEATURE_NAMES,
        ).iloc[0].to_dict()

    # --------------------------------------------------------
    # 7️⃣ Business-safe response
    # --------------------------------------------------------
    return {
        "prediction": YieldPredictionResponse(
            expected_yield=round(y_pred, 2),
            yield_lower=round(lower, 2),
            yield_upper=round(upper, 2),
            risk_level=risk,
            confidence="High",
            key_drivers=[
                "Early NDVI",
                "Seasonal rainfall",
                "Temperature stress",
                "Soil organic carbon",
            ],
        ).dict(),
        "explanation": explanation,
    }
