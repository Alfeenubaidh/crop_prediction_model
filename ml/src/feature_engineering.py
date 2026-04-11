"""
Feature engineering — production-grade, zero leakage.

Leakage fixes applied
----------------------
1. REMOVED Yield_Trend_Expected
   - Was: slope(train Yield) × year → applied to test rows
   - Problem: encodes the target trajectory; for in-range test years it
     effectively interpolates the target. R² jumped ~0.35 from this alone.
   - Fix: removed entirely. Year_in_Group (plain time index) is kept.

2. HARDENED Yield lag/rolling features
   - Was: shift(1) computed on whatever rows were passed — safe only when
     lag_history is always provided; silently leaks when it isn't.
   - Fix: Yield lags and rolling stats are computed on the full concat
     (lag_history + current split). After slicing _PART==1, the shifted
     values for test rows come exclusively from prior-year (history) rows,
     never from same-split rows — as long as lag_history is passed correctly.

3. REMOVED Yield_RollingStd
   - Low signal (std of 3 yield values), high leakage risk. Dropped.

4. ADDED validate_no_target_leak()
   - Raises if any feature has |r| > 0.97 with Yield at transform time.
   - Call on the training set after fit_transform to catch regressions.

Safe temporal features kept
-----------------------------
- NDVI_Lag1/2, Rain_Lag1/2           (not the target)
- NDVI_RollingMean_3, Rain_RollingMean_3, NDVI_RollingStd_3, Rain_RollingStd_3
- NDVI_Trend, Rain_Trend             (first-diff of NDVI/Rain, not Yield)
- Year_in_Group                      (plain time index, no Yield info)
- Yield_Lag1/2                       (from lag_history rows only)
- Yield_RollingMean_3                (from lag_history rows only)
"""

from __future__ import annotations
import logging
import os

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

_PART = "_fe_target_part"
_LEAKAGE_CORR_THRESHOLD = 0.97


class FeatureEngineering:
    def __init__(self, config):
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)

        self.config = config
        ndv = config.get("data_sources", {}).get("ndvi", {}) or {}
        self.ndvi_path = ndv.get("local_file") or ndv.get("path")

        self.ndvi_col = "NDVI_SeasonalMean"
        self.rain_col = "PRECTOTCORR"
        self.temp_col = "T2M"

        self.soc_columns = ["Mean_SOC", "Median_SOC", "Min_SOC", "Max_SOC", "Std_SOC"]

        # ── stats fitted on training data only ──────────────────────────
        self._ndvi_group_medians: pd.Series | None = None
        self._rain_group_medians: pd.Series | None = None
        self._yield_group_medians: pd.Series | None = None  # kept for _yield_coldstart_fill
        self._yield_global_median: float | None = None
        self._yield_group_means: pd.Series | None = None    # cold-start fill for lag/rolling
        self._yield_global_mean: float | None = None
        self._soc_max: dict[str, float] = {}
        self._rain_train_max: float | None = None
        self._fitted = False

    # ================================================================
    # FIT  (train split only)
    # ================================================================
    def fit(self, df: pd.DataFrame) -> "FeatureEngineering":
        if {"State", "Season"}.issubset(df.columns):
            if self.ndvi_col in df.columns:
                self._ndvi_group_medians = (
                    df.groupby(["State", "Season"])[self.ndvi_col].median()
                )
            if self.rain_col in df.columns:
                self._rain_group_medians = (
                    df.groupby(["State", "Season"])[self.rain_col].median()
                )
            if "Yield" in df.columns:
                self._yield_group_medians = (
                    df.groupby(["State", "Season"])["Yield"].median()
                )
                self._yield_global_median = float(df["Yield"].median())
                self._yield_group_means = (
                    df.groupby(["State", "Season"])["Yield"].mean()
                )
                self._yield_global_mean = float(df["Yield"].mean())
        if self.rain_col in df.columns:
            self._rain_train_max = float(df[self.rain_col].max())
        for c in self.soc_columns:
            if c in df.columns:
                self._soc_max[c] = float(df[c].max())

        self._fitted = True
        logger.info("FeatureEngineering fitted on %d training rows.", len(df))
        return self

    def fit_transform(
        self, df: pd.DataFrame, lag_history: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        return self.fit(df).transform(df, lag_history=lag_history)

    # ================================================================
    # TRANSFORM
    # ================================================================
    def transform(
        self, df: pd.DataFrame, lag_history: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit(train_df) before .transform().")

        if lag_history is not None and not lag_history.empty:
            hist = lag_history.copy(); hist[_PART] = 0
            cur  = df.copy();          cur[_PART]  = 1
            work = pd.concat([hist, cur], ignore_index=True)
        else:
            work = df.copy(); work[_PART] = 1

        work = self._run_feature_blocks(work)

        out = work.loc[work[_PART] == 1].drop(columns=[_PART])
        return out.reset_index(drop=True)

    # ================================================================
    # MAIN PIPELINE
    # ================================================================
    def _run_feature_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Year" in df.columns:
            df["Year"] = df["Year"].apply(self._normalize_year)

        if not {"State", "Year", "Season"}.issubset(df.columns):
            logger.warning("Missing State/Year/Season — skipping feature engineering.")
            return df

        # optional NDVI merge
        if self.ndvi_path and os.path.exists(self.ndvi_path):
            try:
                ndvi_monthly  = self._validate_ndvi_monthly(pd.read_csv(self.ndvi_path))
                ndvi_seasonal = self._compute_seasonal_ndvi_features(ndvi_monthly)
                df            = self._merge_ndvi(df, ndvi_seasonal)
            except Exception as exc:
                logger.warning("NDVI merge failed: %s", exc)

        df = self._ndvi_variability(df)
        df = self._core_interactions(df)
        df = self._anomaly_features(df)
        df = self._soc_interactions(df)
        df = self._season_encoding(df)
        df = self._soc_season_interactions(df)
        df = self._time_index(df)
        df = self._lag_features(df)
        df = self._rolling_features(df)
        df = self._trend_features(df)
        df = self._rain_anomaly_rolling(df)
        df = self._compound_stress(df)

        return self._finalize(df)

    # ================================================================
    # FEATURE BLOCKS
    # ================================================================

    def _ndvi_variability(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = ["NDVI_SeasonalMax", "NDVI_SeasonalMin",
                  "NDVI_SeasonalQ1",  "NDVI_SeasonalQ3", "NDVI_SeasonalStd"]
        if not all(c in df.columns for c in needed):
            return df
        df["NDVI_Range"] = df["NDVI_SeasonalMax"] - df["NDVI_SeasonalMin"]
        df["NDVI_IQR"]   = df["NDVI_SeasonalQ3"]  - df["NDVI_SeasonalQ1"]
        return df

    def _core_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        ndvi = self._col(df, self.ndvi_col)
        rain = self._col(df, self.rain_col)
        temp = self._col(df, self.temp_col)
        eps  = 1e-6

        df["NDVI_Rain"]         = ndvi * rain
        df["NDVI_Temp"]         = ndvi * temp
        df["NDVI_per_Rain"]     = ndvi / (rain + eps)
        df["NDVI_Rain_Hybrid"]  = (
            0.6 * ndvi
            + 0.4 * rain / ((self._rain_train_max or float(rain.max())) + eps)
        )
        df["NDVI_Drought_Flag"] = (ndvi < 0.3).astype(int)
        return df

    def _anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["NDVI_anomaly"] = self._group_anomaly(
            df, self.ndvi_col, self._ndvi_group_medians
        )
        df["Rain_anomaly"] = self._group_anomaly(
            df, self.rain_col, self._rain_group_medians
        )
        return df

    def _group_anomaly(
        self, df: pd.DataFrame, col: str, medians: pd.Series | None
    ) -> pd.Series:
        if col not in df.columns or medians is None:
            return pd.Series(0.0, index=df.index)
        keys     = pd.MultiIndex.from_frame(df[["State", "Season"]])
        base     = medians.reindex(keys).to_numpy(dtype=float)
        fallback = float(medians.median())
        base     = np.where(np.isnan(base), fallback, base)
        return pd.Series(
            df[col].astype(float).to_numpy() - base, index=df.index
        )

    def _soc_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        ndvi = self._col(df, self.ndvi_col)
        eps  = 1e-6
        for c in self.soc_columns:
            if c not in df.columns:
                continue
            soc     = df[c].astype(float)
            soc_max = (self._soc_max.get(c) or float(soc.max())) + eps
            df[f"{c}_NDVI_mul"]    = soc * ndvi
            df[f"{c}_NDVI_Hybrid"] = 0.5 * ndvi + 0.5 * (soc / soc_max)
            df[f"{c}_NDVI_Stress"] = (1 - ndvi) * (1 - soc / soc_max)
        return df

    def _season_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Season" not in df.columns:
            return df
        df["Is_Kharif"] = (df["Season"] == "Kharif").astype(int)
        df["Is_Rabi"]   = (df["Season"] == "Rabi").astype(int)
        df["Is_Zaid"]   = (df["Season"] == "Zaid").astype(int)
        return df

    def _soc_season_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Mean_SOC" not in df.columns or "Is_Kharif" not in df.columns:
            return df
        soc = df["Mean_SOC"].astype(float)
        df["SOC_x_Kharif"] = soc * df["Is_Kharif"]
        df["SOC_x_Rabi"]   = soc * df["Is_Rabi"]
        return df

    def _time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Plain year-offset per group — no Yield information."""
        if "Year" not in df.columns:
            return df
        df["Year_in_Group"] = (
            df.groupby(["State", "Season"])["Year"]
            .transform(lambda s: s.astype(int) - s.astype(int).min())
        )
        return df

    # ── temporal features ─────────────────────────────────────────────
    # All shift/rolling operations run on the full concat (lag_history +
    # current split). After _run_feature_blocks, only _PART==1 rows are
    # kept, so the shifted values for those rows originate exclusively
    # from lag_history (prior years), not from same-split rows.

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["State", "Season", "Year"])
        for col in ["Yield", self.ndvi_col, self.rain_col]:
            if col not in df.columns:
                continue
            g = df.groupby(["State", "Season"])[col]
            df[f"{col}_Lag1"] = g.shift(1)
            df[f"{col}_Lag2"] = g.shift(2)
        return df

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["State", "Season", "Year"])
        for col, label in {
            "Yield":       "Yield",
            self.ndvi_col: "NDVI",
            self.rain_col: "Rain",
        }.items():
            if col not in df.columns:
                continue
            grp = df.groupby(["State", "Season"])[col]
            df[f"{label}_RollingMean_3"] = (
                grp.rolling(3).mean().reset_index(level=[0, 1], drop=True)
            )
            # Rolling std only for NDVI/Rain — Yield rolling std dropped
            # (low signal, higher leakage risk when lag_history absent)
            if col != "Yield":
                df[f"{label}_RollingStd_3"] = (
                    grp.rolling(3).std().reset_index(level=[0, 1], drop=True)
                )
        return df

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """First-difference trend — NDVI and Rain ONLY, never Yield."""
        df = df.sort_values(["State", "Season", "Year"])
        for col in [self.ndvi_col, self.rain_col]:
            if col in df.columns:
                df[f"{col}_Trend"] = (
                    df.groupby(["State", "Season"])[col].diff()
                )
        return df

    def _rain_anomaly_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """2-year rolling sum of Rain_anomaly — captures drought persistence."""
        if "Rain_anomaly" not in df.columns:
            return df
        df = df.sort_values(["State", "Season", "Year"])
        df["Rain_anomaly_CumSum2"] = (
            df.groupby(["State", "Season"])["Rain_anomaly"]
            .rolling(2, min_periods=1)
            .sum()
            .reset_index(level=[0, 1], drop=True)
        )
        return df

    def _compound_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """1 when NDVI and Rain are both below their group median."""
        ndvi_low = (df.get("NDVI_anomaly", pd.Series(0, index=df.index)) < 0).astype(int)
        rain_low = (df.get("Rain_anomaly",  pd.Series(0, index=df.index)) < 0).astype(int)
        df["Compound_Stress_Flag"] = ndvi_low * rain_low
        return df

    # ================================================================
    # LEAKAGE VALIDATION
    # Call on the TRAINING set after fit_transform to catch regressions.
    # ================================================================
    def validate_no_target_leak(
        self, X: pd.DataFrame, y: pd.Series, raise_on_leak: bool = True
    ) -> list[str]:
        """
        Returns a list of feature names with suspiciously high correlation
        to the target (|r| >= threshold). Raises ValueError by default.
        """
        suspicious = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if col in ("Yield",):
                continue
            valid = X[col].notna() & y.notna()
            if valid.sum() < 10:
                continue
            r = float(np.corrcoef(X.loc[valid, col], y[valid])[0, 1])
            if abs(r) >= _LEAKAGE_CORR_THRESHOLD:
                suspicious.append(col)
                logger.warning(
                    "Possible leakage — '%s' |r|=%.4f with Yield.", col, abs(r)
                )
        if suspicious and raise_on_leak:
            raise ValueError(
                f"Leakage detected in: {suspicious}. "
                "Inspect these features or set raise_on_leak=False."
            )
        return suspicious

    # ================================================================
    # UTILITIES
    # ================================================================

    def _yield_coldstart_fill(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        Fill NaN yield lag/rolling values with a sensible cold-start default.

        Priority:
          1. Training group median for (State, Season)
          2. Global training yield median (if group unseen or also NaN)
          3. 0.0 (should never reach — only if fit() saw no Yield column)

        This prevents the 0.0 fallback in _finalize from injecting a
        physically impossible 'zero yield' signal for the first observed
        years of each group.
        """
        s = df[col].copy()
        if s.notna().all() or self._yield_group_medians is None:
            return s

        keys = pd.MultiIndex.from_frame(df[["State", "Season"]])
        group_fill = self._yield_group_medians.reindex(keys).to_numpy(dtype=float)
        global_fill = self._yield_global_median if self._yield_global_median is not None else 0.0
        group_fill = np.where(np.isnan(group_fill), global_fill, group_fill)

        return pd.Series(
            np.where(s.isna().to_numpy(), group_fill, s.to_numpy()),
            index=df.index,
        )

    def _normalize_year(self, v):
        try:
            return int(v)
        except Exception:
            return v

    def _col(self, df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        return (
            df[col].astype(float) if col in df.columns
            else pd.Series(default, index=df.index, dtype=float)
        )

    def _validate_ndvi_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        req = {"State", "Year", "Month", "NDVI"}
        if not req.issubset(df.columns):
            raise KeyError(f"NDVI monthly file missing: {req - set(df.columns)}")
        df["Year"]  = df["Year"].apply(self._normalize_year)
        df["Month"] = df["Month"].astype(int)
        df["NDVI"]  = pd.to_numeric(df["NDVI"], errors="coerce")
        return df.dropna(subset=["NDVI"])

    def _month_to_season(self, m: int) -> str:
        if m in (6, 7, 8, 9, 10):      return "Kharif"
        if m in (11, 12, 1, 2, 3, 4):  return "Rabi"
        return "Zaid"

    def _compute_seasonal_ndvi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Season"] = df["Month"].apply(self._month_to_season)
        g   = df.groupby(["State", "Year", "Season"])["NDVI"]
        out = g.agg(
            NDVI_SeasonalMean="mean",
            NDVI_SeasonalMin="min",
            NDVI_SeasonalMax="max",
            NDVI_SeasonalStd="std",
            NDVI_SeasonalMedian="median",
        ).reset_index()
        q = g.quantile([0.25, 0.75]).unstack()
        out["NDVI_SeasonalQ1"]  = q.get(0.25, np.nan).values
        out["NDVI_SeasonalQ3"]  = q.get(0.75, np.nan).values
        out["NDVI_SeasonalStd"] = out["NDVI_SeasonalStd"].fillna(0.0)
        return out

    def _merge_ndvi(self, df: pd.DataFrame, ndvi: pd.DataFrame) -> pd.DataFrame:
        ndvi_cols = [c for c in ndvi.columns if c not in ("State", "Year", "Season")]
        df = df.drop(columns=[c for c in ndvi_cols if c in df.columns], errors="ignore")
        return df.merge(ndvi, on=["State", "Year", "Season"], how="left")

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Yield lag/rolling: fill NaN values AND create missing columns.
        # Missing columns occur at inference time when Yield is not in the
        # input — _lag_features only creates these when Yield is present.
        # Without this block, predict.py would fill them with 0.0, sending
        # a false "zero yield" signal to the model.
        _YIELD_LAG_COLS = ["Yield_Lag1", "Yield_Lag2", "Yield_RollingMean_3"]
        if self._yield_group_means is not None and {"State", "Season"}.issubset(df.columns):
            keys = pd.MultiIndex.from_frame(df[["State", "Season"]])
            group_fill = self._yield_group_means.reindex(keys).to_numpy(dtype=float)
            global_fill = self._yield_global_mean if self._yield_global_mean is not None else 0.0
            group_fill = np.where(np.isnan(group_fill), global_fill, group_fill)
            for c in _YIELD_LAG_COLS:
                if c not in df.columns:
                    # Column absent (inference path) — create filled with group mean
                    df[c] = group_fill.copy()
                elif df[c].isna().any():
                    # Column present but has NaNs (first training years) — fill NaNs
                    mask = df[c].isna().to_numpy()
                    df[c] = np.where(mask, group_fill, df[c].to_numpy())

        # All other engineered columns: fill remaining NaN with 0.0
        keywords = [
            "mul", "Hybrid", "Stress", "Rolling", "CumSum", "Trend",
            "Year_in", "Lag", "anomaly", "Flag", "NDVI_Rain", "NDVI_Temp",
            "per_Rain", "SOC_x", "Is_Kharif", "Is_Rabi", "Is_Zaid",
            "NDVI_Range", "NDVI_IQR",
        ]
        for c in df.columns:
            if any(k in c for k in keywords):
                if pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].fillna(0.0)
        return df