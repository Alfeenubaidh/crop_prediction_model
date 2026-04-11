import pandas as pd
import numpy as np
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f) or {}
except FileNotFoundError:
    CONFIG = {}


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

_FALLBACK_KEY = "__fallback__"


class OutlierHandler:
    """
    Learn clipping bounds on training data only (fit), then apply to any split (transform).
    """

    def __init__(self, config=None):
        self.config = config or CONFIG.get("preprocessing", {}).get("outliers", {})
        if not self.config:
            raise ValueError("OutlierHandler requires outlier config or a valid config.yaml")
        self.columns = self.config.get("numeric_columns", [])
        self.detector = OutlierDetectorFactory.create(self.config)
        # (group_key_tuple, col) -> (lower, upper); global fallback (_FALLBACK_KEY, col)
        self._bounds: dict = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "OutlierHandler":
        if self.detector is None or df.empty:
            self._fitted = True
            return self

        cols = [c for c in self.columns if c in df.columns]
        if not cols:
            self._fitted = True
            return self

        self._bounds = {}

        if isinstance(self.detector, GroupWiseIQROutlierDetector):
            group_cols = self.detector.group_columns
            group_cols = [c for c in group_cols if c in df.columns]
            if not group_cols:
                self._fit_global_bounds(df, cols)
            else:
                for keys, group in df.groupby(group_cols):
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    for col in cols:
                        if col not in group.columns:
                            continue
                        q1, q3 = group[col].quantile([0.25, 0.75])
                        iqr = q3 - q1
                        lo = q1 - self.detector.multiplier * iqr
                        hi = q3 + self.detector.multiplier * iqr
                        self._bounds[(keys, col)] = (lo, hi)
                self._fit_global_bounds(df, cols)
        elif isinstance(self.detector, IQROutlierDetector):
            for col in cols:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lo = q1 - self.detector.multiplier * iqr
                hi = q3 + self.detector.multiplier * iqr
                self._bounds[(_FALLBACK_KEY, col)] = (lo, hi)
        elif isinstance(self.detector, ZScoreOutlierDetector):
            for col in cols:
                m = df[col].mean()
                s = df[col].std(ddof=0)
                if pd.isna(s) or s == 0:
                    self._bounds[(_FALLBACK_KEY, col)] = (-np.inf, np.inf)
                else:
                    lo = m - self.detector.threshold * s
                    hi = m + self.detector.threshold * s
                    self._bounds[(_FALLBACK_KEY, col)] = (lo, hi)

        self._fitted = True
        return self

    def _fit_global_bounds(self, df: pd.DataFrame, cols: list[str]) -> None:
        """Per-column train-wide bounds for groups missing at transform time."""
        for col in cols:
            if col not in df.columns:
                continue
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mult = (
                self.detector.multiplier
                if isinstance(self.detector, GroupWiseIQROutlierDetector)
                else self.config.get("iqr_multiplier", 1.5)
            )
            lo = q1 - mult * iqr
            hi = q3 + mult * iqr
            self._bounds[(_FALLBACK_KEY, col)] = (lo, hi)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.detector is None or df.empty:
            return df.reset_index(drop=True)
        if not self._fitted:
            raise RuntimeError("Call .fit(train_df) before .transform().")

        cols = [c for c in self.columns if c in df.columns]
        if not cols:
            return df.reset_index(drop=True)

        df = df.copy()

        if isinstance(self.detector, GroupWiseIQROutlierDetector):
            group_cols = self.detector.group_columns
            group_cols = [c for c in group_cols if c in df.columns]
            if not group_cols:
                self._clip_global(df, cols)
            else:
                for keys, idx in df.groupby(group_cols).groups.items():
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    for col in cols:
                        bounds = self._bounds.get((keys, col)) or self._bounds.get(
                            (_FALLBACK_KEY, col)
                        )
                        if bounds:
                            df.loc[idx, col] = df.loc[idx, col].clip(bounds[0], bounds[1])
        else:
            self._clip_global(df, cols)

        return df.reset_index(drop=True)

    def _clip_global(self, df: pd.DataFrame, cols: list[str]) -> None:
        for col in cols:
            bounds = self._bounds.get((_FALLBACK_KEY, col))
            if bounds:
                df[col] = df[col].clip(bounds[0], bounds[1])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DEPRECATED — do not use in the training pipeline.
        This method fits AND transforms on the same df. If df contains val/test rows,
        clipping bounds will be computed from all splits, leaking test-set statistics
        into the training process.

        Use the proper pattern instead:
            handler.fit(train_df)
            train_df = handler.transform(train_df)
            val_df   = handler.transform(val_df)
            test_df  = handler.transform(test_df)
        """
        raise RuntimeError(
            "OutlierHandler.apply() is disabled — it fits on whatever df is passed, "
            "which leaks val/test statistics when called on the full dataset. "
            "Use .fit(train_df) then .transform(split_df) for each split separately."
        )