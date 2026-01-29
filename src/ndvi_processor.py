import os
from typing import Tuple
import pandas as pd
import numpy as np

from .utils_agri import season_months, preseason_month_map

class NDVIProcessor:
    """
    Loads monthly NDVI time series per State and computes:
      - seasonal mean (NDVI_SeasonalMean)
      - seasonal lag-1 mean (NDVI_SeasonalMean_lag1) -> previous year's same season
      - preseason mean (NDVI_preseason_mean) -> month(s) before season start
    """

    def __init__(self, config: dict):
        self.config = config or {}
        ndvi_cfg = self.config.get("data_sources", {}).get("ndvi", {}) if self.config else {}
        self.ndvi_path = ndvi_cfg.get("path", None)
        # default months mapping from utils_agri
        self.season_months = season_months()
        self.preseason_map = preseason_month_map()

    def load_monthly_ndvi(self) -> pd.DataFrame:
        """
        Expect CSV with at least: State, Year, Month, NDVI
        Year: int, Month: int (1-12), NDVI: float
        """
        if not self.ndvi_path:
            raise ValueError("NDVI path not set in config['data_sources']['ndvi']['path']")
        if not os.path.exists(self.ndvi_path):
            raise FileNotFoundError(f"NDVI file not found: {self.ndvi_path}")

        df = pd.read_csv(self.ndvi_path)
        # normalize column names
        df.columns = [c.strip() for c in df.columns]
        required = {"State", "Year", "Month", "NDVI"}
        if not required.issubset(set(df.columns)):
            raise KeyError(f"NDVI table missing required columns: {required - set(df.columns)}")

        # ensure types
        df = df.copy()
        df["State"] = df["State"].astype(str).str.strip()
        df["Year"] = df["Year"].astype(int)
        df["Month"] = df["Month"].astype(int)
        df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")
        return df

    def compute_seasonal_ndvi(self, ndvi_monthly: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
           State, Year, Season, NDVI_SeasonalMean, NDVI_SeasonalMean_lag1, NDVI_preseason_mean
        """
        df = ndvi_monthly.copy()

        results = []

        # iterate unique states and years
        states = df["State"].unique()
        years = df["Year"].unique()

        for state in states:
            state_df = df[df["State"] == state]
            for year in sorted(state_df["Year"].unique()):
                for season, months in self.season_months.items():
                    # special handling for seasons crossing year boundary (Rabi: Oct-Mar)
                    months_mask = None
                    if len(months) == 0:
                        continue

                    if min(months) <= max(months):
                        # normal season fully within year
                        months_mask = (state_df["Year"] == year) & (state_df["Month"].isin(months))
                    else:
                        # crossing year (e.g., months = [10,11,12,1,2,3])
                        # include months where (Year==year and Month in months >=10)
                        # and (Year==year+1 and Month in months <=3)
                        months_hi = [m for m in months if m >= 10]
                        months_lo = [m for m in months if m < 10]
                        mask_hi = (state_df["Year"] == year) & (state_df["Month"].isin(months_hi))
                        mask_lo = (state_df["Year"] == year + 1) & (state_df["Month"].isin(months_lo))
                        months_mask = mask_hi | mask_lo

                    seasonal_values = state_df[months_mask]["NDVI"].dropna()
                    seasonal_mean = seasonal_values.mean() if not seasonal_values.empty else np.nan

                    # preseason months: map season -> month(s) before start
                    premonths = self.preseason_map.get(season, [])
                    preseason_mask = (state_df["Year"] == year) & (state_df["Month"].isin(premonths))
                    preseason_mean = state_df[preseason_mask]["NDVI"].dropna().mean()
                    # lag1: previous year's same season mean
                    prev_year = year - 1
                    if min(months) <= max(months):
                        prev_mask = (state_df["Year"] == prev_year) & (state_df["Month"].isin(months))
                    else:
                        months_hi = [m for m in months if m >= 10]
                        months_lo = [m for m in months if m < 10]
                        prev_mask_hi = (state_df["Year"] == prev_year) & (state_df["Month"].isin(months_hi))
                        prev_mask_lo = (state_df["Year"] == prev_year + 1) & (state_df["Month"].isin(months_lo))
                        prev_mask = prev_mask_hi | prev_mask_lo

                    prev_vals = state_df[prev_mask]["NDVI"].dropna()
                    prev_mean = prev_vals.mean() if not prev_vals.empty else np.nan

                    results.append({
                        "State": state,
                        "Year": year,
                        "Season": season,
                        "NDVI_SeasonalMean": seasonal_mean,
                        "NDVI_SeasonalMean_lag1": prev_mean,
                        "NDVI_preseason_mean": preseason_mean
                    })

        out = pd.DataFrame(results)
        # drop duplicates if any and sort
        out = out.drop_duplicates(subset=["State", "Year", "Season"]).reset_index(drop=True)
        return out

    def run_full(self) -> pd.DataFrame:
        """Load monthly NDVI and compute seasonal & lag features."""
        ndvi_month = self.load_monthly_ndvi()
        seasonal = self.compute_seasonal_ndvi(ndvi_month)
        return seasonal

# Example usage:
# proc = NDVIProcessor(config)
# seasonal = proc.run_full()
