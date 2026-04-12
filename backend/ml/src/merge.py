import warnings
import pandas as pd
import numpy as np

# ============================================================
# Utilities
# ============================================================

def normalize_state(s):
    return (
        s.astype(str)
         .str.upper()
         .str.strip()
    )


def extract_year(s):
    return (
        s.astype(str)
         .str.extract(r"(\d{4})")[0]
         .astype("Int64")
    )


def month_to_season(m):
    try:
        m = int(m)
    except Exception:
        return pd.NA
    if 6 <= m <= 9:
        return "Kharif"
    if m >= 10 or m <= 4:
        return "Rabi"
    return pd.NA


# ============================================================
# Yield Processor
# ============================================================

class YieldProcessor:

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "State" not in df.columns or "Year" not in df.columns:
            raise KeyError("Yield data must contain State and Year")

        df["State"] = normalize_state(df["State"])
        df["Year"] = extract_year(df["Year"])

        if "Season" not in df.columns:
            df["Season"] = pd.NA

        df["Season"] = (
            df["Season"]
            .astype(str)
            .str.title()
            .replace({"Nan": pd.NA})
        )

        df = df.dropna(subset=["State", "Year", "Yield"])

        return df


# ============================================================
# Weather Processor
# ============================================================

class WeatherProcessor:

    DEFAULT_AGG = {
        "PRECTOTCORR": "sum",
        "T2M": "mean",
        "T2M_MAX": "mean",
        "T2M_MIN": "mean",
        "ALLSKY_SFC_SW_DWN": "sum",
        "RH2M": "mean",
        "WS2M": "mean",
    }

    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["State"] = normalize_state(df["State"])
        df["Year"] = extract_year(df["Year"])

        if "Month" in df.columns:
            df["Season"] = df["Month"].apply(month_to_season)

        df = df.dropna(subset=["State", "Year", "Season"])

        return (
            df.groupby(["State", "Year", "Season"], as_index=False)
              .agg(WeatherProcessor.DEFAULT_AGG)
        )


# ============================================================
# NDVI Processor
# ============================================================

class NDVIProcessor:

    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["State"] = normalize_state(df["State"])
        df["Year"] = extract_year(df["Year"])

        if "Month" not in df.columns:
            raise KeyError("NDVI data must contain Month")

        df["Season"] = df["Month"].apply(month_to_season)

        df = df.dropna(subset=["State", "Year", "Season", "NDVI"])

        return (
            df.groupby(["State", "Year", "Season"], as_index=False)
              .agg(NDVI_SeasonalMean=("NDVI", "mean"))
        )


# ============================================================
# SOC Processor
# ============================================================

class SOCProcessor:

    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["State"] = normalize_state(df["State"])

        numeric_cols = [c for c in df.columns if c != "State"]

        return df.groupby("State", as_index=False)[numeric_cols].mean()


# ============================================================
# FINAL DATASET MERGER (LEAKAGE-FREE)
# ============================================================

class DatasetMerger:
    """
    Clean merger — NO statistical operations here.
    """

    def __init__(self, config=None):
        self.config = config

    def merge(
        self,
        weather_seasonal: pd.DataFrame,
        ndvi: pd.DataFrame,
        soc: pd.DataFrame,
        yield_df: pd.DataFrame,
    ) -> pd.DataFrame:

        # ---- Clean inputs
        yield_clean = YieldProcessor.clean(yield_df)
        weather_clean = WeatherProcessor.prepare(weather_seasonal)
        ndvi_clean = NDVIProcessor.prepare(ndvi)
        soc_clean = SOCProcessor.prepare(soc)

        # ---- Weather season lookup
        weather_seasons = (
            weather_clean[["State", "Year", "Season"]]
            .drop_duplicates()
        )

        # ---- Expand yield to seasons
        yield_expanded = yield_clean.merge(
            weather_seasons,
            on=["State", "Year"],
            how="left",
            suffixes=("_yield", "_weather")
        )

        yield_expanded["Season"] = (
            yield_expanded["Season_yield"]
            .combine_first(yield_expanded["Season_weather"])
        )

        yield_expanded = yield_expanded.drop(
            columns=[c for c in yield_expanded.columns if c.endswith("_yield") or c.endswith("_weather")]
        )

        missing = yield_expanded["Season"].isna().sum()
        if missing > 0:
            warnings.warn(
                f"{missing} yield rows have no matching weather season; "
                "weather/NDVI will be NaN for those rows."
            )

        yield_expanded["Season"] = yield_expanded["Season"].astype(str).str.title()

        # ---- Merge all datasets (yield is anchor)
        merged = (
            yield_expanded
            .merge(weather_clean, on=["State", "Year", "Season"], how="left")
            .merge(ndvi_clean, on=["State", "Year", "Season"], how="left")
            .merge(soc_clean, on="State", how="left")
        )

        # 🚫 IMPORTANT: NO IMPUTATION HERE (avoids leakage)

        merged = merged.drop_duplicates().reset_index(drop=True)

        if merged.empty:
            raise RuntimeError(
                "Final merged dataset is empty. "
                "Check State normalization and Year overlap."
            )

        return merged