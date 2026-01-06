import json
import os
import pandas as pd
import joblib


class FeatureSchemaMismatchError(Exception):
    pass


class FeatureBuilder:
    """
    Inference-time feature builder that:
    - Uses TRAINED preprocessor
    - Enforces ENCODED feature schema
    """

    def __init__(
        self,
        preprocessor_path: str,
        feature_schema_path: str,
    ):
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

        if not os.path.exists(feature_schema_path):
            raise FileNotFoundError(f"Feature schema not found: {feature_schema_path}")

        self.preprocessor = joblib.load(preprocessor_path)

        with open(feature_schema_path) as f:
            self.feature_schema = json.load(f)

    def build(self, payload: dict) -> pd.DataFrame:
        """
        Convert raw API payload → encoded model-ready DataFrame
        """

        # Raw → DataFrame
        raw_df = pd.DataFrame([payload])

        # Apply SAME preprocessing as training
        X_enc = self.preprocessor.transform(raw_df)

        X_enc_df = pd.DataFrame(X_enc, columns=self.feature_schema)

        # -----------------------------
        # Final safety check
        # -----------------------------
        missing = set(self.feature_schema) - set(X_enc_df.columns)
        if missing:
            raise FeatureSchemaMismatchError(
                f"Missing encoded features at inference: {sorted(missing)}"
            )

        # Enforce exact order
        X_enc_df = X_enc_df[self.feature_schema]

        return X_enc_df
