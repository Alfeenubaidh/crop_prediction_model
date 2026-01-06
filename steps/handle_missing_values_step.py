# steps/handle_missing_values_step.py
from zenml import step
import pandas as pd

from src.handle_missing_values import MissingValueHandler
from src.utils.config_loader import load_config


@step
def handle_missing_values_step(data: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    cfg = config or load_config("config.yaml")
    handler = MissingValueHandler(cfg)
    cleaned = handler.process(data)
    print(f"[handle_missing_values_step] rows out: {len(cleaned)}")
    return cleaned
