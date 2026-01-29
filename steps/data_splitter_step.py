from zenml import step
from typing import Tuple
import pandas as pd

from src.data_splitter import DataSplitter, DataSplitterConfig


@step
def data_split_step(
    cleaned_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series
]:
    """
    Splits merged + cleaned dataframe into train/val/test sets.
    """

    # ---------- SAFETY CHECK (CRITICAL) ----------
    if cleaned_df is None or cleaned_df.empty:
        raise ValueError(
            "data_split_step received an EMPTY dataframe. "
            "Check merge / missing-values / outlier steps."
        )

    if "Yield" not in cleaned_df.columns:
        raise KeyError("Expected target column 'Yield' not found in cleaned_df.")

    # ---------- SPLIT ----------
    config = DataSplitterConfig()
    splitter = DataSplitter(config)
    split_data = splitter.split(cleaned_df)

    return (
        split_data["X_train"],
        split_data["X_val"],
        split_data["X_test"],
        split_data["y_train"],
        split_data["y_val"],
        split_data["y_test"],
    )
