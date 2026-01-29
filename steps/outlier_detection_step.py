# steps/outlier_detection_step.py

from typing import Dict, Any
from zenml import step
import pandas as pd

from src.outlier_detection import OutlierHandler


@step
def handle_outliers_step(
    data: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Detect and handle outliers in the dataset.

    Args:
        data: Input dataframe after missing-value handling.
        config: Global pipeline configuration dictionary.

    Returns:
        Cleaned dataframe with outliers handled.
    """
    outlier_cfg = config["preprocessing"]["outliers"]

    handler = OutlierHandler(outlier_cfg)
    cleaned = handler.apply(data)

    print(f"[handle_outliers_step] rows out: {len(cleaned)}")

    return cleaned
