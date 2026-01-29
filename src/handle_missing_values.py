import pandas as pd
import yaml

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)


class MissingValueHandler:

    def __init__(self, config=None):
        cfg = config or CONFIG

        if "preprocessing" not in cfg:
            raise KeyError("Missing 'preprocessing' in config.yaml")
        if "missing_values" not in cfg["preprocessing"]:
            raise KeyError("Missing 'missing_values' in config.yaml")

        self.cfg = cfg["preprocessing"]["missing_values"]

        self.weather_cfg = self.cfg.get("weather", {})
        self.ndvi_cfg = self.cfg.get("ndvi", {})
        self.soc_cfg = self.cfg.get("soc", {})
        self.yield_cfg = self.cfg.get("yield", {})

    # ======================================================
    # WEATHER
    # ======================================================
    def _handle_weather(self, df):
        cfg = self.weather_cfg

        if "columns" not in cfg:
            return df

        cols = [c for c in cfg["columns"] if c in df.columns]

        if not cols:
            return df

        if cfg.get("method") == "interpolate":
            df[cols] = df[cols].interpolate(method="linear")
            df[cols] = df[cols].fillna(df[cols].median())

        return df

    # ======================================================
    # NDVI
    # ======================================================
    def _handle_ndvi(self, df):

        # choose correct NDVI column
        ndvi_col = "NDVI_SeasonalMean" if "NDVI_SeasonalMean" in df.columns else \
                   "NDVI" if "NDVI" in df.columns else None

        if not ndvi_col:
            return df

        # fill grouped missing values
        if "State" in df.columns and "Season" in df.columns:
            df[ndvi_col] = (
                df.groupby(["State", "Season"])[ndvi_col]
                  .transform(lambda x: x.fillna(x.median()))
            )

        # global fallback
        df[ndvi_col] = df[ndvi_col].fillna(df[ndvi_col].median())

        # NDVI interactions
        eps = 1e-9

        if "PRECTOTCORR" in df.columns:
            df["NDVI_Rain_Ratio"] = df[ndvi_col] / (df["PRECTOTCORR"] + eps)

        if "T2M" in df.columns:
            df["NDVI_Temp_Interaction"] = df[ndvi_col] * df["T2M"]
            df["Temp_NDVI"] = df[ndvi_col] * df["T2M"]

        # NDVI BINS
        if "ndvi_bins" in self.ndvi_cfg and "ndvi_labels" in self.ndvi_cfg:
            df["NDVI_Level"] = pd.cut(
                df[ndvi_col],
                bins=self.ndvi_cfg["ndvi_bins"],
                labels=self.ndvi_cfg["ndvi_labels"],
                include_lowest=True,
            )

        return df

    # ======================================================
    # SOC (SOIL ORGANIC CARBON)
    # ======================================================
    def _handle_soc(self, df):
        cfg = self.soc_cfg

        if "columns" not in cfg:
            return df

        cols = [c for c in cfg["columns"] if c in df.columns]

        if not cols:
            return df

        if cfg.get("method") == "median":
            for col in cols:
                df[col] = df[col].fillna(df[col].median())

        return df

    # ======================================================
    # YIELD
    # ======================================================
    def _handle_yield(self, df):
        cfg = self.yield_cfg

        if "columns" not in cfg:
            return df

        cols = [c for c in cfg["columns"] if c in df.columns]

        if not cols:
            return df

        if cfg.get("method") == "drop":
            df = df.dropna(subset=cols)

        return df

    # ======================================================
    # MASTER PROCESSOR
    # ======================================================
    def process(self, df):
        df = df.copy()

        df = self._handle_weather(df)
        df = self._handle_ndvi(df)
        df = self._handle_soc(df)
        df = self._handle_yield(df)

        # NO GLOBAL FILL — avoids silent corruption

        return df.reset_index(drop=True)
