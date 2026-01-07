import json
import os
import pandas as pd
import numpy as np
import joblib
import yaml

from src.feature_engineering import FeatureEngineering
from src.ndvi_processor import NDVIProcessor


class FeatureSchemaMismatchError(Exception):
    pass


class FeatureBuilder:
    """
    Inference-time FeatureBuilder that EXACTLY reproduces
    the training feature pipeline (Option A: historical reconstruction).
    """

    def __init__(
        self,
        preprocessor_path: str,
        feature_schema_path: str,
        base_data_path: str,
        config_path: str,
    ):
        # -----------------------
        # Validate paths
        # -----------------------
        for p in [
            preprocessor_path,
            feature_schema_path,
            base_data_path,
            config_path,
        ]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        # -----------------------
        # Load artifacts
        # -----------------------
        self.preprocessor = joblib.load(preprocessor_path)

        with open(feature_schema_path, "r") as f:
            self.feature_schema = json.load(f)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.base_data_path = base_data_path

    # ======================================================
    # PUBLIC ENTRY POINT
    # ======================================================
    def build(self, payload: dict) -> pd.DataFrame:
        """
        Build a single model-ready encoded feature row
        for inference.
        """

        # 1. Load historical base data
        base = self._load_base_history(payload)

        # 2. Compute NDVI seasonal features (same as training)
        ndvi_seasonal = self._compute_ndvi_seasonal()

        # 3. Merge NDVI into base
        base = self._merge_ndvi(base, ndvi_seasonal)

        # 4. Append the prediction year row
        base = self._append_prediction_row(base, payload)

        # 5. Run FULL feature engineering (rolling, lag, interactions)
        engineered = self._run_feature_engineering(base)

        # 6. Select only the prediction row
        final_row = self._select_prediction_row(engineered, payload)

        # 7. Encode using trained preprocessor
        return self._encode(final_row)

    # ======================================================
    # INTERNAL STEPS
    # ======================================================
    def _load_base_history(self, payload: dict) -> pd.DataFrame:
        df = pd.read_csv(self.base_data_path)

        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["State"] = df["State"].astype(str)
        df["Season"] = df["Season"].astype(str)

        df = df[
            (df["State"] == payload["state"])
            & (df["Season"] == payload["season"])
            & (df["Year"] <= payload["year"])
        ].copy()

        # Rolling(3) + Lag(1) require history
        if df["Year"].nunique() < 3:
            raise ValueError(
                "Not enough historical data to compute lag/rolling features"
            )

        return df

    def _compute_ndvi_seasonal(self) -> pd.DataFrame:
        proc = NDVIProcessor(self.config)
        return proc.run_full()

    def _merge_ndvi(self, base: pd.DataFrame, ndvi: pd.DataFrame) -> pd.DataFrame:
        return base.merge(
            ndvi,
            on=["State", "Year", "Season"],
            how="left",
            validate="m:1",
        )

    def _append_prediction_row(self, df: pd.DataFrame, payload: dict) -> pd.DataFrame:
        """
        Append the inference year as a new row.
        Yield must be NaN (unknown).
        """
        row = {
            "State": payload["state"],
            "Season": payload["season"],
            "Year": payload["year"],
            "Yield": np.nan,
        }

        return pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True,
        )

    def _run_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        fe = FeatureEngineering(self.config)
        return fe.transform(df)

    def _select_prediction_row(
        self, df: pd.DataFrame, payload: dict
    ) -> pd.DataFrame:
        row = df[df["Year"] == payload["year"]]

        if row.empty:
            raise ValueError(
                "Prediction year row missing after feature engineering"
            )

        # Always take the last row (safe with concat)
        return row.iloc[[-1]]

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply trained preprocessor and return encoded features.
        IMPORTANT: schema applies AFTER preprocessing.
        """

        # Apply preprocessing
        X_enc = self.preprocessor.transform(df)

        # Convert to DataFrame with encoded feature names
        X_enc_df = pd.DataFrame(
            X_enc,
            columns=self.feature_schema,
        )

        return X_enc_df
