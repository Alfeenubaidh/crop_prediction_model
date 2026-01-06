# steps/outlier_detection_step.py
from zenml import step
import pandas as pd

from src.utils.config_loader import load_config
from src.outlier_detection import OutlierHandler

@step
def handle_outliers_step(data, config):
    outlier_cfg = config["preprocessing"]["outliers"]
    handler = OutlierHandler(outlier_cfg)
    cleaned = handler.apply(data)
    print(f"[handle_outliers_step] rows out: {len(cleaned)}")
    return cleaned
