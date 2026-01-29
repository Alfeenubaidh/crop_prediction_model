import pandas as pd
import numpy as np
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


# ============================================================
# BASE
# ============================================================

class OutlierDetectorBase:
    def bounds(self, df, columns):
        raise NotImplementedError


# ============================================================
# Z-SCORE METHOD (CLIPPING)
# ============================================================

class ZScoreOutlierDetector(OutlierDetectorBase):
    def __init__(self, threshold):
        self.threshold = threshold

    def bounds(self, df, columns):
        df_num = df[columns].astype(float)

        means = df_num.mean()
        stds = df_num.std(ddof=0).replace(0, np.nan)

        lower = means - self.threshold * stds
        upper = means + self.threshold * stds

        return lower, upper


# ============================================================
# IQR METHOD (CLIPPING)
# ============================================================

class IQROutlierDetector(OutlierDetectorBase):
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def bounds(self, df, columns):
        df_num = df[columns].astype(float)

        Q1 = df_num.quantile(0.25)
        Q3 = df_num.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - self.multiplier * IQR
        upper = Q3 + self.multiplier * IQR

        return lower, upper


# ============================================================
# GROUPWISE IQR (CLIPPING)
# ============================================================

class GroupWiseIQROutlierDetector(OutlierDetectorBase):
    def __init__(self, multiplier=1.5, group_columns=None):
        if not group_columns:
            raise ValueError("group_columns must be specified")
        self.multiplier = multiplier
        self.group_columns = group_columns

    def clip(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()

        for _, group_idx in df.groupby(self.group_columns).groups.items():
            g = df.loc[group_idx, columns]

            Q1 = g.quantile(0.25)
            Q3 = g.quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - self.multiplier * IQR
            upper = Q3 + self.multiplier * IQR

            df.loc[group_idx, columns] = g.clip(lower, upper, axis=1)

        return df


# ============================================================
# FACTORY
# ============================================================

class OutlierDetectorFactory:
    @staticmethod
    def create(config):
        method = config["method"].lower()

        if method == "iqr_groupwise":
            group_cols = config.get("groupby") or [config.get("group_column")]
            return GroupWiseIQROutlierDetector(
                multiplier=config.get("iqr_multiplier", 1.5),
                group_columns=group_cols
            )

        if method == "iqr":
            return IQROutlierDetector(config.get("iqr_multiplier", 1.5))

        if method in ("zscore", "z_score"):
            return ZScoreOutlierDetector(config.get("zscore_threshold", 3.0))

        if method == "none":
            return None

        raise ValueError(f"Unknown method: {method}")


# ============================================================
# MASTER HANDLER (SAFE)
# ============================================================

class OutlierHandler:
    def __init__(self, config=None):
        self.config = config or CONFIG["preprocessing"]["outliers"]
        self.columns = self.config.get("numeric_columns", [])
        self.detector = OutlierDetectorFactory.create(self.config)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.detector is None or df.empty:
            return df

        cols = [c for c in self.columns if c in df.columns]
        if not cols:
            return df

        df = df.copy()

        # ---- Groupwise IQR (safe)
        if isinstance(self.detector, GroupWiseIQROutlierDetector):
            df = self.detector.clip(df, cols)

        # ---- Global IQR / Z-score
        else:
            lower, upper = self.detector.bounds(df, cols)
            df[cols] = df[cols].clip(lower, upper, axis=1)

        return df.reset_index(drop=True)
