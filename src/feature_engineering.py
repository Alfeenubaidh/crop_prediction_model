"""
Extended Feature Engineering for Crop Yield Prediction.
FINAL VERSION — TRAINING = INFERENCE
"""

from __future__ import annotations
import os
import logging
import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class FeatureEngineering:
    def __init__(self, config):
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)

        self.config = config
        self.ndvi_path = config.get("data_sources", {}).get("ndvi", {}).get("path")

        self.ndvi_col = "NDVI_SeasonalMean"
        self.rain_col = "PRECTOTCORR"
        self.temp_col = "T2M"

        self.soc_columns = [
            "Mean_SOC", "Median_SOC", "Min_SOC", "Max_SOC", "Std_SOC"
        ]

    # =====================================================
    # SAFE ACCESSOR (CRITICAL)
    # =====================================================
    def _safe_series(self, df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        if col in df.columns:
            return df[col].astype(float)
        return pd.Series(default, index=df.index, dtype=float)

    # =====================================================
    # MAIN PIPELINE
    # =====================================================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Year"] = df["Year"].apply(self._normalize_year)

        if not {"State", "Year", "Season"}.issubset(df.columns):
            logger.warning("Missing State/Year/Season columns — skipping feature engineering")
            return df


        # NDVI merge
        if self.ndvi_path and os.path.exists(self.ndvi_path):
            try:
                ndvi_monthly = self._validate_ndvi_monthly(pd.read_csv(self.ndvi_path))
                ndvi_seasonal = self._compute_seasonal_ndvi_features(ndvi_monthly)
                df = self._merge_ndvi(df, ndvi_seasonal)
            except Exception as e:
                logger.warning("NDVI merge failed: %s", e)

        df = self._add_ndvi_variability_metrics(df)
        df = self._create_ndvi_rain_interactions(df)
        df = self._create_ndvi_temp_interactions(df)
        df = self._add_ndvi_level(df)
        df = self._create_ndvi_soc_interactions(df)
        df = self._add_ndvi_stress_indicators(df)
        df = self._add_rain_ndvi_efficiency(df)
        df = self._create_lag_features(df, ["State", "Season"], lag_years=1)
        df = self._add_rolling_features(df, ["State", "Season"], window=3)

        return self._finalize(df)

    # =====================================================
    # HELPERS
    # =====================================================
    def _normalize_year(self, v):
        try:
            return int(v)
        except Exception:
            return v

    def _validate_ndvi_monthly(self, df):
        req = {"State", "Year", "Month", "NDVI"}
        if not req.issubset(df.columns):
            raise KeyError("NDVI monthly file missing columns")

        df["Year"] = df["Year"].apply(self._normalize_year)
        df["Month"] = df["Month"].astype(int)
        df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")
        return df.dropna()

    def _month_to_season(self, m):
        if m in (6, 7, 8, 9, 10):
            return "Kharif"
        if m in (11, 12, 1, 2, 3, 4):
            return "Rabi"
        return "Zaid"

    # =====================================================
    # NDVI AGGREGATION
    # =====================================================
    def _compute_seasonal_ndvi_features(self, df):
        df["Season"] = df["Month"].apply(self._month_to_season)
        g = df.groupby(["State", "Year", "Season"])["NDVI"]

        out = g.agg(
            NDVI_SeasonalMean="mean",
            NDVI_SeasonalMin="min",
            NDVI_SeasonalMax="max",
            NDVI_SeasonalStd="std",
            NDVI_SeasonalMedian="median",
        ).reset_index()

        q = g.quantile([0.25, 0.75]).unstack()
        out["NDVI_SeasonalQ1"] = q.get(0.25, np.nan).values
        out["NDVI_SeasonalQ3"] = q.get(0.75, np.nan).values
        out["NDVI_SeasonalStd"] = out["NDVI_SeasonalStd"].fillna(0.0)

        return out

    def _merge_ndvi(self, df, ndvi):
        return df.merge(ndvi, on=["State", "Year", "Season"], how="left")

    # =====================================================
    # FEATURE BLOCKS
    # =====================================================
    def _add_ndvi_variability_metrics(self, df):
        required_cols = [
            "NDVI_SeasonalMax",
            "NDVI_SeasonalMin",
            "NDVI_SeasonalQ1",
            "NDVI_SeasonalQ3",
            "NDVI_SeasonalStd",
        ]

        # Defensive check: only compute variability metrics if all required columns exist
        if not all(col in df.columns for col in required_cols):
            return df

        df["NDVI_Range"] = df["NDVI_SeasonalMax"] - df["NDVI_SeasonalMin"]
        df["NDVI_IQR"] = df["NDVI_SeasonalQ3"] - df["NDVI_SeasonalQ1"]
        df["NDVI_Std_Norm"] = (
            df["NDVI_SeasonalStd"] / (df[self.ndvi_col].abs() + 1e-6)
        )

        return df


    def _create_ndvi_rain_interactions(self, df):
        ndvi = self._safe_series(df, self.ndvi_col)
        rain = self._safe_series(df, self.rain_col)
        eps = 1e-6

        df["NDVI_plus_Rain"] = ndvi + rain
        df["NDVI_minus_Rain"] = ndvi - rain
        df["NDVI_mul_Rain"] = ndvi * rain
        df["NDVI_div_Rain"] = ndvi / (rain + eps)
        df["Rain_div_NDVI"] = rain / (ndvi + eps)
        df["NDVI_Rain_Ratio"] = ndvi / (rain + eps)

        df["NDVI_anomaly"] = ndvi - ndvi.groupby(
            [df["State"], df["Season"]]
        ).transform("median")

        df["Rain_anomaly"] = rain - rain.groupby(
            [df["State"], df["Season"]]
        ).transform("median")

        return df

    def _create_ndvi_temp_interactions(self, df):
        ndvi = self._safe_series(df, self.ndvi_col)
        temp = self._safe_series(df, self.temp_col)

        df["Temp_NDVI"] = temp * ndvi
        df["NDVI_Temp_Interaction"] = df["Temp_NDVI"]
        return df

    def _add_ndvi_level(self, df):
        def level(v):
            if pd.isna(v):
                return "moderate"
            if v < 0.3:
                return "low"
            if v < 0.6:
                return "moderate"
            return "high"

        df["NDVI_Level"] = self._safe_series(df, self.ndvi_col, 0.5).apply(level)
        return df

    def _create_ndvi_soc_interactions(self, df):
        ndvi = self._safe_series(df, self.ndvi_col)
        eps = 1e-6

        for c in self.soc_columns:
            if c not in df:
                continue

            soc = df[c].astype(float)
            df[f"{c}_NDVI_mul"] = soc * ndvi
            df[f"{c}_NDVI_div"] = ndvi / (soc + eps)
            df[f"{c}_SOC_div_NDVI"] = soc / (ndvi + eps)
            df[f"{c}_NDVI_Hybrid"] = 0.5 * ndvi + 0.5 * (soc / (soc.max() + eps))
            df[f"{c}_NDVI_Stress"] = (1 - ndvi) * (1 - (soc / (soc.max() + eps)))

        return df

    def _add_ndvi_stress_indicators(self, df):
        ndvi = self._safe_series(df, self.ndvi_col)
        df["NDVI_Drought_Flag"] = (ndvi < 0.3).astype(int)
        return df

    def _add_rain_ndvi_efficiency(self, df):
        ndvi = self._safe_series(df, self.ndvi_col)
        rain = self._safe_series(df, self.rain_col)
        eps = 1e-6

        df["NDVI_efficiency_per_mmRain"] = ndvi / (rain + eps)
        df["Rain_to_NDVI_Ratio"] = rain / (ndvi + eps)
        df["NDVI_Rain_Hybrid"] = 0.6 * ndvi + 0.4 * rain / (rain.max() + eps)
        return df

    def _create_lag_features(self, df, group_cols, lag_years=1):
        df = df.sort_values(group_cols + ["Year"])
        for src in ["Yield", self.ndvi_col, self.rain_col]:
            if src in df:
                df[f"{src}_Lag{lag_years}"] = (
                    df.groupby(group_cols)[src].shift(lag_years)
                )
        return df

    def _add_rolling_features(self, df, group_cols, window=3):
        df = df.sort_values(group_cols + ["Year"])
        for col, label in {
            "Yield": "Yield",
            self.ndvi_col: "NDVI",
            self.rain_col: "Rain",
        }.items():
            if col in df:
                df[f"{label}_RollingMean_{window}"] = (
                    df.groupby(group_cols)[col]
                    .rolling(window)
                    .mean()
                    .reset_index(level=[0, 1], drop=True)
                )
                df[f"{label}_RollingStd_{window}"] = (
                    df.groupby(group_cols)[col]
                    .rolling(window)
                    .std()
                    .reset_index(level=[0, 1], drop=True)
                )
        return df

    def _finalize(self, df):
        for c in df.columns:
            if any(k in c for k in ["mul", "div", "Hybrid", "Stress", "Rolling"]):
                df[c] = df[c].fillna(0.0)
        return df
