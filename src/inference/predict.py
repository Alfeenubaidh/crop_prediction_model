import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List

from src.feature_engineering import FeatureEngineering


def run_prediction(
    features: pd.DataFrame,
    model_path: str,
    encoder_path: str,
    config: dict,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # Feature engineering
    fe = FeatureEngineering(config)
    features_fe = fe.transform(features)

    # Drop target if present
    if "Yield" in features_fe.columns:
        features_fe = features_fe.drop(columns=["Yield"])

    # Align with training schema
    expected_features = encoder.feature_names_in_

    for col in expected_features:
        if col not in features_fe.columns:
            features_fe[col] = 0.0

    features_fe = features_fe[expected_features]

    # Encode + predict
    X_enc = encoder.transform(features_fe)
    preds = model.predict(X_enc)

    # ✅ CRITICAL FIX
    encoded_feature_names = encoder.get_feature_names_out().tolist()

    output = features.copy()
    output["Predicted_Yield"] = preds

    return output, X_enc, encoded_feature_names
