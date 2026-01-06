# steps/feature_engineering_step.py
from zenml import step
import pandas as pd

from src.feature_engineering import FeatureEngineering
from src.utils.config_loader import load_config


@step
def feature_engineering_step(data: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    if data is None or data.empty:
        raise ValueError("feature_engineering_step received EMPTY dataframe")

    cfg = config or load_config("config.yaml")
    fe = FeatureEngineering(cfg)

    features = fe.transform(data)

    if features is None or features.empty:
        raise ValueError("FeatureEngineering.transform returned EMPTY dataframe")

    features = features.loc[:, features.notna().any()]

    return features

