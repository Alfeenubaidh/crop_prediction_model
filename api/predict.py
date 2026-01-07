from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path

from api.schemas import UserInputRequest
from api.feature_builder import FeatureBuilder
from api.model_loader import ModelLoader
from src.explainer.shap_explainer import ShapExplainer


# ============================================================
# Lazy singletons (loaded once per process)
# ============================================================

_feature_builder: FeatureBuilder | None = None
_model = None
_shap_explainer: ShapExplainer | None = None
BASE_DIR = Path(__file__).resolve().parents[1]


def _get_feature_builder() -> FeatureBuilder:
    global _feature_builder
    if _feature_builder is None:
        _feature_builder = FeatureBuilder(
            preprocessor_path=str(BASE_DIR / "models" / "preprocessor.joblib"),
            feature_schema_path=str(BASE_DIR / "models" / "encoded_feature_schema.json"),
            base_data_path=str(
                BASE_DIR / "data" / "Processed" / "versions" / "merged_output.csv"
            ),
            config_path=str(BASE_DIR / "config.yaml"),
        )
    return _feature_builder


def _get_model():
    global _model
    if _model is None:
        loader = ModelLoader()
        _model = loader.load_model()
    return _model


def _get_shap_explainer() -> ShapExplainer:
    global _shap_explainer
    if _shap_explainer is None:
        _shap_explainer = ShapExplainer(_get_model())
    return _shap_explainer


# ============================================================
# PUBLIC INFERENCE ENTRY POINT
# ============================================================

def predict_yield(payload: Dict, explain: bool = False):
    request = UserInputRequest(**payload)

    X_enc = _get_feature_builder().build(request.dict())
    model = _get_model()

    y_pred = float(model.predict(X_enc)[0])

    explanation = None
    if explain:
        explainer = _get_shap_explainer()
        explanation = explainer.explain(X_enc)

    return {
        "predicted_yield": y_pred,
        "explanation": explanation
    }

