# steps/outlier_detection_step.py

from typing import Dict, Any, Tuple

from zenml import step
import pandas as pd

from ml.src.outlier_detection import OutlierHandler


@step
def handle_outliers_step(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit outlier bounds on training data only; clip train, validation, and test.
    """
    outlier_cfg = config["preprocessing"]["outliers"]

    handler = OutlierHandler(outlier_cfg)
    handler.fit(train_df)

    train_out = handler.transform(train_df)
    val_out = handler.transform(val_df)
    test_out = handler.transform(test_df)

    print(
        f"[handle_outliers_step] rows: train={len(train_out)} val={len(val_out)} "
        f"test={len(test_out)}"
    )

    return train_out, val_out, test_out
