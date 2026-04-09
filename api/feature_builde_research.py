import os
import json
import pandas as pd
import numpy as np
import joblib
import yaml

from src.feature_engineering import FeatureEngineering
from src.ndvi_processor import NDVIProcessor


class FeatureBuilder:
    """
    Inference-time FeatureBuilder.
    RETURNS NAMED FEATURES — NO NUMERIC COLUMNS.
    """

    def __init__(
        self,
        preprocessor_path: str,
        feature_schema_path: str,
        base_data_path: str,
        config_path: str,
    ):
        for p in [
            preprocessor_path,
            feature_schema_path,
            base_data_path,
            config_path,
        ]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        self.preprocessor = joblib.load(preprocessor_path)

        with open(feature_schema_path, "r") as f:
            self.feature_names = json.load(f)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.base_data_path = base_data_path

    # --------------------------------------------------
    def build(self, payload: dict) -> pd.DataFrame:
        base = self._load_base_history(payload)
        ndvi = self._compute_ndvi_seasonal()
        base = self._merge_ndvi(base, ndvi)
        base = self._append_prediction_row(base, payload)

        engineered = self._run_feature_engineering(base)
        final_row = self._select_prediction_row(engineered, payload)

        return self._encode(final_row)

    # --------------------------------------------------
    def _load_base_history(self, payload):
        df = pd.read_csv(self.base_data_path)

        df = df[
            (df["State"] == payload["state"])
            & (df["Season"] == payload["season"])
            & (df["Year"] <= payload["year"])
        ].copy()

        if df["Year"].nunique() < 3:
            raise ValueError("Not enough historical data")

        return df

    def _compute_ndvi_seasonal(self):
        return NDVIProcessor(self.config).run_full()

    def _merge_ndvi(self, base, ndvi):
        return base.merge(
            ndvi, on=["State", "Year", "Season"], how="left"
        )

    def _append_prediction_row(self, df, payload):
        row = {
            "State": payload["state"],
            "Season": payload["season"],
            "Year": payload["year"],
            "Yield": np.nan,
        }
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    def _run_feature_engineering(self, df):
        return FeatureEngineering(self.config).transform(df)

    def _select_prediction_row(self, df, payload):
        return df[df["Year"] == payload["year"]].iloc[[-1]]

    # --------------------------------------------------
    def _encode(self, df):
        X = self.preprocessor.transform(df)
        print("ENCODED COLUMNS:", self.feature_names[:5], "...", len(self.feature_names))


        # 🔥 CRITICAL FIX
        return pd.DataFrame(
            X,
            columns=self.feature_names,
            index=[0],
        )
