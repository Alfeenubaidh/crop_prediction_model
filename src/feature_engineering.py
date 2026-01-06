# src/feature_engineering.py
"""
Extended Feature Engineering for Crop Yield Prediction.

Includes:
- NDVI seasonal stats (mean, min/max, std, median, q1, q3)
- NDVI variability metrics (range, IQR, normalized std)
- NDVI stress indicators
- NDVI rain interactions
- NDVI SOC interactions  (NEW: Option A)
- NDVI rain/SOC efficiency metrics
- Temporal rolling windows (3-year) for NDVI, Rain, Yield  (NEW: Option B)
- Lag features
- Safe NDVI merging
"""

from __future__ import annotations
import os
import logging
import math

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

        ndvi_cfg = config.get("data_sources", {}).get("ndvi", {})
        self.ndvi_path = ndvi_cfg.get("path")

        self.ndvi_col = (
            config.get("feature_engineering", {})
            .get("ndvi_features", {})
            .get("ndvi_column", "NDVI_SeasonalMean")
        )

        self.rain_col = "PRECTOTCORR"

        # SOC columns available
        self.soc_columns = [
            "Mean_SOC", "Median_SOC", "Min_SOC",
            "Max_SOC", "Std_SOC"
        ]

    # =====================================================================
    # MAIN PIPELINE
    # =====================================================================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["Year"] = df["Year"].apply(self._normalize_year)

        required = {"State", "Year", "Season"}
        if not required.issubset(df.columns):
            raise KeyError("DataFrame must include State, Year, Season")

        # -------------------------------
        # NDVI seasonal load + merge
        # -------------------------------
        ndvi_seasonal = None
        if self.ndvi_path and os.path.exists(self.ndvi_path):
            try:
                logger.info("Loading NDVI monthly from %s", self.ndvi_path)
                ndvi_monthly = self._validate_ndvi_monthly(pd.read_csv(self.ndvi_path))
                ndvi_seasonal = self._compute_seasonal_ndvi_features(ndvi_monthly)
            except Exception as e:
                logger.warning("NDVI seasonal computation failed: %s", e)

        if ndvi_seasonal is not None:
            df = self._merge_ndvi(df, ndvi_seasonal)

        # -------------------------------
        # NDVI variability metrics
        # -------------------------------
        df = self._add_ndvi_variability_metrics(df)

        # -------------------------------
        # NDVI–rain interactions
        # -------------------------------
        df = self._create_ndvi_rain_interactions(df)

        # -------------------------------
        # NDVI–SOC interactions (NEW)
        # -------------------------------
        df = self._create_ndvi_soc_interactions(df)

        # -------------------------------
        # Stress indicators (NDVI + Rain)
        # -------------------------------
        df = self._add_ndvi_stress_indicators(df)

        # -------------------------------
        # Rain–NDVI efficiency metrics
        # -------------------------------
        df = self._add_rain_ndvi_efficiency(df)

        # -------------------------------
        # Lag features
        # -------------------------------
        df = self._create_lag_features(df, ["State", "Season"], lag_years=1)

        # -------------------------------
        # Rolling window temporal metrics (NEW)
        # -------------------------------
        df = self._add_rolling_features(df, ["State", "Season"], window=3)

        return self._finalize(df)

    # =====================================================================
    # BASIC UTILITY HELPERS
    # =====================================================================
    def _normalize_year(self, v):
        if pd.isna(v):
            return v
        if isinstance(v, (int, float)):
            return int(v)
        import re
        m = re.search(r"(20\d{2})", str(v))
        if m:
            return int(m.group(1))
        try:
            return int(float(v))
        except:
            return v

    def _validate_ndvi_monthly(self, df):
        req = {"State", "Year", "Month", "NDVI"}
        if not req.issubset(df.columns):
            raise KeyError(f"NDVI monthly file missing columns {req}")
        df["Year"] = df["Year"].apply(self._normalize_year)
        df["Month"] = df["Month"].astype(int)
        df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")
        return df.dropna()

    def _month_to_season(self, m):
        if m in (6, 7, 8, 9, 10):
            return "Kharif"
        if m in (11, 12, 1, 2, 3, 4):
            return "Rabi"
        if m == 5:
            return "Zaid"
        return "Rabi"

    # =====================================================================
    # NDVI aggregation
    # =====================================================================
    def _compute_seasonal_ndvi_features(self, df):
        df = df.copy()
        df["Season"] = df["Month"].apply(self._month_to_season)

        grouped = df.groupby(["State", "Year", "Season"])["NDVI"]

        agg = grouped.agg(
            NDVI_SeasonalMean="mean",
            NDVI_SeasonalMin="min",
            NDVI_SeasonalMax="max",
            NDVI_SeasonalStd="std",
            NDVI_SeasonalMedian="median",
        ).reset_index()

        q = grouped.quantile([0.25, 0.75]).unstack()

        agg["NDVI_SeasonalQ1"] = q.get(0.25, np.nan).values
        agg["NDVI_SeasonalQ3"] = q.get(0.75, np.nan).values

        agg["NDVI_SeasonalStd"] = agg["NDVI_SeasonalStd"].fillna(0)

        return agg

    # =====================================================================
    # SAFE NDVI MERGE
    # =====================================================================
    def _merge_ndvi(self, df: pd.DataFrame, ndvi_seasonal: pd.DataFrame) -> pd.DataFrame:
        """
        Merge NDVI seasonal stats safely without duplicate column errors.
        Handles missing old columns by using new values directly.
        """

        left = df.copy()
        right = ndvi_seasonal.copy()

        ndvi_cols = [
            "NDVI_SeasonalMean",
            "NDVI_SeasonalMin",
            "NDVI_SeasonalMax",
            "NDVI_SeasonalStd",
            "NDVI_SeasonalMedian",
            "NDVI_SeasonalQ1",
            "NDVI_SeasonalQ3",
        ]

        # Rename new NDVI columns
        rename_map = {c: f"{c}_new" for c in ndvi_cols if c in right.columns}
        right = right.rename(columns=rename_map)

        # Safe merge (no suffix collisions)
        merged = left.merge(
            right,
            on=["State", "Year", "Season"],
            how="left",
            validate="m:1"
        )

        # Coalesce logic (SAFE)
        for col in ndvi_cols:
            new = f"{col}_new"
            if new not in merged.columns:
                continue

            if col in merged.columns:
                # both new + old exist → prefer new if not null
                merged[col] = merged[new].combine_first(merged[col])
            else:
                # old missing → simply use new
                merged[col] = merged[new]

            merged = merged.drop(columns=[new])

        logger.info("NDVI seasonal merged successfully (safe coalesce).")
        return merged


    # =====================================================================
    # A) NDVI VARIABILITY METRICS
    # =====================================================================
    def _add_ndvi_variability_metrics(self, df):
        if "NDVI_SeasonalMean" not in df:
            return df

        df["NDVI_Range"] = df["NDVI_SeasonalMax"] - df["NDVI_SeasonalMin"]
        df["NDVI_IQR"] = df["NDVI_SeasonalQ3"] - df["NDVI_SeasonalQ1"]
        df["NDVI_Std_Norm"] = df["NDVI_SeasonalStd"] / (df["NDVI_SeasonalMean"].abs() + 1e-6)
        return df

    # =====================================================================
    # B) NDVI–RAIN INTERACTIONS
    # =====================================================================
    def _create_ndvi_rain_interactions(self, df):
        df = df.copy()
        if "NDVI_SeasonalMean" not in df:
            return df

        ndvi = df["NDVI_SeasonalMean"].astype(float)
        rain = df.get(self.rain_col, pd.Series([0]*len(df))).astype(float)
        eps = 1e-6

        df["NDVI_mul_Rain"] = ndvi * rain
        df["NDVI_div_Rain"] = ndvi / (rain + eps)
        df["Rain_div_NDVI"] = rain / (ndvi + eps)

        df["NDVI_plus_Rain"] = ndvi + rain
        df["NDVI_minus_Rain"] = ndvi - rain

        df["NDVI_anomaly"] = ndvi - ndvi.groupby([df["State"], df["Season"]]).transform("median")
        df["Rain_anomaly"] = rain - rain.groupby([df["State"], df["Season"]]).transform("median")

        return df

    # =====================================================================
    # A) NDVI–SOC INTERACTIONS  (NEW)
    # =====================================================================
    def _create_ndvi_soc_interactions(self, df):
        df = df.copy()
        if "NDVI_SeasonalMean" not in df:
            return df

        ndvi = df["NDVI_SeasonalMean"].astype(float)

        for soc_col in self.soc_columns:
            if soc_col not in df:
                continue

            soc = df[soc_col].astype(float)
            eps = 1e-6

            df[f"{soc_col}_NDVI_mul"] = soc * ndvi
            df[f"{soc_col}_NDVI_div"] = ndvi / (soc + eps)
            df[f"{soc_col}_SOC_div_NDVI"] = soc / (ndvi + eps)

            df[f"{soc_col}_NDVI_Hybrid"] = (
                0.5 * ndvi + 0.5 * (soc / (soc.max() + eps))
            )

            df[f"{soc_col}_NDVI_Stress"] = (1 - ndvi) * (1 - (soc / (soc.max() + eps)))

        return df

    # =====================================================================
    # NDVI STRESS
    # =====================================================================
    def _add_ndvi_stress_indicators(self, df):
        if "NDVI_SeasonalMean" not in df:
            return df
        ndvi = df["NDVI_SeasonalMean"].astype(float)
        df["NDVI_Drought_Flag"] = (ndvi < 0.3).astype(int)
        return df

    # =====================================================================
    # NDVI–RAIN EFFICIENCY
    # =====================================================================
    def _add_rain_ndvi_efficiency(self, df):
        df = df.copy()
        if "NDVI_SeasonalMean" not in df:
            return df

        ndvi = df["NDVI_SeasonalMean"].astype(float)
        rain = df.get(self.rain_col, pd.Series([0]*len(df))).astype(float)
        eps = 1e-6

        df["NDVI_efficiency_per_mmRain"] = ndvi / (rain + eps)
        df["Rain_to_NDVI_Ratio"] = rain / (ndvi + eps)
        df["NDVI_Rain_Hybrid"] = (0.6 * ndvi) + (0.4 * rain / (rain.max() + eps))
        return df

    # =====================================================================
    # LAG FEATURES
    # =====================================================================
    def _create_lag_features(self, df, group_cols, lag_years=1):
        df = df.copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.sort_values(group_cols + ["Year"]).reset_index(drop=True)

        lag_map = {
            "Yield": f"Yield_Lag{lag_years}",
            self.ndvi_col: f"{self.ndvi_col}_Lag{lag_years}",
            self.rain_col: f"{self.rain_col}_Lag{lag_years}",
        }

        for src, lagname in lag_map.items():
            if src in df:
                df[lagname] = df.groupby(group_cols)[src].shift(lag_years)

        return df

    # =====================================================================
    # B) ROLLING WINDOW (3-YEAR)
    # =====================================================================
    def _add_rolling_features(self, df, group_cols, window=3):
        df = df.copy()

        roll_targets = {
            "Yield": "Yield",
            "NDVI_SeasonalMean": "NDVI",
            self.rain_col: "Rain",
        }

        df = df.sort_values(group_cols + ["Year"])

        for col, label in roll_targets.items():
            if col not in df:
                continue

            df[f"{label}_RollingMean_{window}"] = (
                df.groupby(group_cols)[col].rolling(window).mean().reset_index(level=[0,1], drop=True)
            )

            df[f"{label}_RollingStd_{window}"] = (
                df.groupby(group_cols)[col].rolling(window).std().reset_index(level=[0,1], drop=True)
            )

        return df

    # =====================================================================
    # Final cleanup
    # =====================================================================
    def _finalize(self, df):
        df = df.copy()
        # fill interaction NaNs
        for c in df.columns:
            if any(k in c for k in ["mul", "div", "Hybrid", "Stress", "Rolling"]):
                df[c] = df[c].fillna(0)
        return df
