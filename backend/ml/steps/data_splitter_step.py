from zenml import step
from typing import Tuple
import pandas as pd

from ml.src.data_splitter import DataSplitter, DataSplitterConfig
from ml.src.utils.config_loader import load_config


@step
def data_split_step(
    merged_df: pd.DataFrame,
    config: dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split merged data into train / val / test immediately after merge.
    Each dataframe includes the target column (Yield) for lag features and imputation.
    """
    if merged_df is None or merged_df.empty:
        raise ValueError(
            "data_split_step received an EMPTY dataframe. Check merge / ingestion."
        )

    cfg = config or load_config("configs/config.yaml")
    split_cfg = DataSplitterConfig(cfg=cfg)
    splitter = DataSplitter(split_cfg)

    train_df, val_df, test_df = splitter.split_full(merged_df)

    print(
        f"[data_split_step] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        "(rows include target for preprocessing)"
    )
    return train_df, val_df, test_df
