from typing import Tuple
from zenml import step
import pandas as pd
import os

from src.utils.config_loader import load_config
from src.ingest import NasaWeatherIngester, NDVIIngestor
from src.ingest import SoilOrganicCarbonIngestor, YieldIngestor


def _fix_year(df: pd.DataFrame) -> pd.DataFrame:
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df


@step
def data_ingestion(config: dict) -> Tuple[
    pd.DataFrame,  # weather_daily
    pd.DataFrame,  # weather_seasonal
    pd.DataFrame,  # ndvi
    pd.DataFrame,  # soc
    pd.DataFrame,  # yield_df
    pd.DataFrame,  # simulated_df
]:
    cfg = config or load_config()

    weather_ingestor = NasaWeatherIngester(cfg["data_sources"]["weather"])
    yield_ingestor = YieldIngestor(cfg["data_sources"]["yield"])

    weather_daily, weather_seasonal = weather_ingestor.ingest()

    ndvi_df = pd.read_csv(cfg["data_sources"]["ndvi"]["local_file"])
    soc_df = pd.read_csv(cfg["data_sources"]["soc"]["local_file"])
    yield_df = yield_ingestor.ingest()

    weather_daily = _fix_year(weather_daily)
    weather_seasonal = _fix_year(weather_seasonal)
    ndvi_df = _fix_year(ndvi_df)
    soc_df = _fix_year(soc_df)
    yield_df = _fix_year(yield_df)

    # Optional simulated data
    simulated_df = pd.DataFrame()
    sim_cfg = cfg.get("data_sources", {}).get("simulated", {})
    sim_path = sim_cfg.get("path")

    if sim_path and os.path.exists(sim_path):
        simulated_df = _fix_year(pd.read_csv(sim_path))

    # ✅ RETURN A TUPLE — NOT A DICT
    return (
        weather_daily,
        weather_seasonal,
        ndvi_df,
        soc_df,
        yield_df,
        simulated_df,
    )
