# steps/dataset_merger_step.py
from zenml import step
import pandas as pd

from src.utils.config_loader import load_config
from src.merge import DatasetMerger  # adjust path if your merge class is elsewhere


@step
def merge_data(
    weather_seasonal: pd.DataFrame,
    ndvi: pd.DataFrame,
    soc: pd.DataFrame,
    yield_df: pd.DataFrame,
) -> pd.DataFrame:
    cfg = load_config("config.yaml")
    print("[merge_data] weather_seasonal shape:", weather_seasonal.shape)
    print("[merge_data] ndvi shape:", ndvi.shape)
    print("[merge_data] soc shape:", soc.shape)
    print("[merge_data] yield_df shape:", yield_df.shape)

    print("[merge_data] weather_seasonal columns:", weather_seasonal.columns.tolist())
    print("[merge_data] ndvi columns:", ndvi.columns.tolist())
    print("[merge_data] soc columns:", soc.columns.tolist())
    print("[merge_data] yield_df columns:", yield_df.columns.tolist())

    merger = DatasetMerger(config=cfg)
    merged = merger.merge(
        weather_seasonal=weather_seasonal,
        ndvi=ndvi,
        soc=soc,
        yield_df=yield_df,
    )
    print(f"[merge_data] rows out: {len(merged)}")
    return merged
    

