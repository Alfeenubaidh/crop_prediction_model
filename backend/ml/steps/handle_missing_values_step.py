# steps/handle_missing_values_step.py
from typing import Tuple

from zenml import step
import pandas as pd

from ml.src.handle_missing_values import MissingValueHandler
from ml.src.utils.config_loader import load_config


@step
def handle_missing_values_step(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit imputation on training data only; transform train, validation, and test.
    """
    cfg = config or load_config("configs/config.yaml")
    handler = MissingValueHandler(cfg)
    handler.fit(train_df)

    train_out = handler.transform(train_df)
    val_out = handler.transform(val_df)
    test_out = handler.transform(test_df)

    print(
        f"[handle_missing_values_step] rows: train={len(train_out)} val={len(val_out)} "
        f"test={len(test_out)}"
    )
    return train_out, val_out, test_out
