import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """
    Fit statistics on training data only, then transform any split.
    """

    def __init__(self, config=None):
        if config is None:
            raise ValueError("MissingValueHandler requires a config dict")

        if "preprocessing" not in config:
            raise KeyError("Missing 'preprocessing' in config")
        if "missing_values" not in config["preprocessing"]:
            raise KeyError("Missing 'missing_values' in config['preprocessing']")

        mv = config["preprocessing"]["missing_values"]
        self.weather_cfg = mv.get("weather", {})
        self.ndvi_cfg = mv.get("ndvi", {})
        self.soc_cfg = mv.get("soc", {})
        self.yield_cfg = mv.get("yield", {})

        self._weather_medians: dict = {}
        self._soc_medians: dict = {}
        self._ndvi_group_medians: pd.Series | None = None
        self._ndvi_global_median: float | None = None
        self._fitted = False

    @staticmethod
    def _detect_ndvi_col(df: pd.DataFrame) -> str | None:
        for c in ("NDVI_SeasonalMean", "NDVI"):
            if c in df.columns:
                return c
        return None

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        df = df.copy()

        weather_cols = [c for c in self.weather_cfg.get("columns", []) if c in df.columns]
        for col in weather_cols:
            self._weather_medians[col] = df[col].median()

        soc_cols = [c for c in self.soc_cfg.get("columns", []) if c in df.columns]
        for col in soc_cols:
            self._soc_medians[col] = df[col].median()

        ndvi_col = self._detect_ndvi_col(df)
        if ndvi_col and "State" in df.columns and "Season" in df.columns:
            self._ndvi_group_medians = df.groupby(["State", "Season"])[ndvi_col].median()
        if ndvi_col:
            self._ndvi_global_median = float(df[ndvi_col].median())

        self._fitted = True
        logger.info("MissingValueHandler fitted on training data.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit(train_df) before .transform().")

        df = df.copy()
        df = self._handle_weather(df)
        df = self._handle_ndvi(df)
        df = self._handle_soc(df)
        df = self._handle_yield(df)
        return df.reset_index(drop=True)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _handle_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.weather_cfg
        if "columns" not in cfg:
            return df

        cols = [c for c in cfg["columns"] if c in df.columns]
        if not cols:
            return df

        if cfg.get("method") == "interpolate":
            df[cols] = df[cols].interpolate(method="linear")
            for col in cols:
                med = self._weather_medians.get(col)
                if med is not None:
                    df[col] = df[col].fillna(med)

        return df

    def _handle_ndvi(self, df: pd.DataFrame) -> pd.DataFrame:
        ndvi_col = self._detect_ndvi_col(df)
        if not ndvi_col:
            return df

        if (
            self._ndvi_group_medians is not None
            and "State" in df.columns
            and "Season" in df.columns
        ):
            gm = self._ndvi_group_medians.reset_index(name="_ndvi_grp_med")
            df = df.merge(gm, on=["State", "Season"], how="left")
            df[ndvi_col] = df[ndvi_col].fillna(df["_ndvi_grp_med"])
            df = df.drop(columns=["_ndvi_grp_med"])

        if self._ndvi_global_median is not None:
            df[ndvi_col] = df[ndvi_col].fillna(self._ndvi_global_median)

        # NOTE: NDVI_Rain_Ratio, NDVI_Temp_Interaction, Temp_NDVI, and NDVI_Level
        # are intentionally NOT created here. Feature engineering belongs exclusively
        # in FeatureEngineering.transform(), which is fitted on training data only.
        # Creating these columns here caused ~25 duplicate/leaky features and
        # inflated R² to 0.94+.

        return df

    def _handle_soc(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.soc_cfg.get("columns", []) if c in df.columns]
        if not cols:
            return df
        if self.soc_cfg.get("method") == "median":
            for col in cols:
                med = self._soc_medians.get(col)
                if med is not None:
                    df[col] = df[col].fillna(med)
        return df

    def _handle_yield(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.yield_cfg.get("columns", []) if c in df.columns]
        if not cols:
            return df
        if self.yield_cfg.get("method") in ("drop", "drop_missing"):
            df = df.dropna(subset=cols)
        return df