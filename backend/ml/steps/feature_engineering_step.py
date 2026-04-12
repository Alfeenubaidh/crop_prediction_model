from typing import Tuple

from zenml import step
import pandas as pd

from ml.src.feature_engineering import FeatureEngineering
from ml.src.utils.config_loader import load_config


@step
def feature_engineering_step(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict | None = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Fit feature engineering on training data; transform train, val, and test.
    Validation/test use prior rows as lag_history so lags do not reset at split boundaries.
    Returns X_*, y_* with target column removed from features.
    """
    if train_df is None or train_df.empty:
        raise ValueError("feature_engineering_step received EMPTY train_df")

    cfg = config or load_config("conifgs/config.yaml")
    target = cfg["data_sources"]["yield"].get("target_column", "Yield")

    if target not in train_df.columns:
        raise KeyError(f"Target column {target!r} not found on train_df for feature engineering.")

    fe = FeatureEngineering(cfg)
    train_fe = fe.fit_transform(train_df, lag_history=None)
    val_fe = fe.transform(val_df, lag_history=train_df)
    test_fe = fe.transform(test_df, lag_history=pd.concat([train_df, val_df], ignore_index=True))

    for name, part in ("train", train_fe), ("val", val_fe), ("test", test_fe):
        if part is None or part.empty:
            raise ValueError(f"feature_engineering_step produced EMPTY {name} features")

    X_train = train_fe.drop(columns=[target])
    y_train = train_fe[target]
    X_val = val_fe.drop(columns=[target])
    y_val = val_fe[target]
    X_test = test_fe.drop(columns=[target])
    y_test = test_fe[target]

    # Drop columns that are all-NaN on training (same as previous single-table behavior)
    keep = X_train.columns[X_train.notna().any(axis=0)]
    X_train = X_train.loc[:, keep]
    X_val = X_val.reindex(columns=keep)
    X_test = X_test.reindex(columns=keep)

    print(
        f"[feature_engineering_step] feature columns={len(keep)} "
        f"rows train={len(X_train)} val={len(X_val)} test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
