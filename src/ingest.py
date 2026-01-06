from __future__ import annotations
import os
import logging
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

DEFAULT_RAW_DIR = os.environ.get("RAW_DIR", "data/raw")


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_get(d: dict, key: str, default=None):
    return d.get(key, default)


# ============================================================
# NASA POWER Weather Ingester
# ============================================================
class NasaWeatherIngester:
    def __init__(self, cfg: Dict, raw_dir: Optional[str] = None):
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict (weather config block)")
        self.start_date = cfg["start_date"]
        self.end_date = cfg["end_date"]
        self.states = cfg["states"]
        self.raw_dir = raw_dir or DEFAULT_RAW_DIR
        ensure_dir(self.raw_dir)

    def fetch_data_point(self, lat: float, lon: float) -> pd.DataFrame:
        import requests
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"

        params = {
            "latitude": lat,
            "longitude": lon,
            "start": self.start_date,
            "end": self.end_date,
            "parameters":
                "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,"
                "ALLSKY_SFC_SW_DWN,RH2M,WS2M",
            "community": "AG",
            "format": "JSON",
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        js = resp.json()

        if "properties" not in js or "parameter" not in js["properties"]:
            raise RuntimeError("Unexpected NASA POWER response format")

        df = pd.DataFrame(js["properties"]["parameter"])
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def add_season(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Month"] = df.index.month
        df["Year"] = df.index.year
        df["Season"] = None
        df.loc[(df["Month"] >= 10) | (df["Month"] <= 4), "Season"] = "Rabi"
        df.loc[df["Month"].between(6, 9), "Season"] = "Kharif"
        return df

    @staticmethod
    def compute_seasonal(df: pd.DataFrame) -> pd.DataFrame:
        agg_map = {
            "T2M": "mean",
            "T2M_MAX": "mean",
            "T2M_MIN": "mean",
            "PRECTOTCORR": "sum",
            "ALLSKY_SFC_SW_DWN": "sum",
            "RH2M": "mean",
            "WS2M": "mean",
        }
        agg_map = {k: v for k, v in agg_map.items() if k in df.columns}
        return df.groupby(["Year", "Season"]).agg(agg_map).reset_index()

    def ingest(self):
        all_daily = []
        logger.info("Starting NASA POWER ingestion for %d states", len(self.states))

        for state, coords in self.states.items():
            lat, lon = coords
            logger.info("Fetching weather for %s (%.3f, %.3f)", state, lat, lon)

            df = self.fetch_data_point(lat, lon)
            df = self.add_season(df)
            df["State"] = state
            all_daily.append(df)

        if not all_daily:
            raise RuntimeError("No weather data fetched")

        daily_df = pd.concat(all_daily).reset_index().rename(columns={"index": "Date"})
        daily_df.to_csv(os.path.join(self.raw_dir, "weather_daily.csv"), index=False)

        seasonal_rows = []
        for state in daily_df["State"].unique():
            seasonal = self.compute_seasonal(daily_df[daily_df["State"] == state])
            seasonal["State"] = state
            seasonal_rows.append(seasonal)

        seasonal_df = pd.concat(seasonal_rows).reset_index(drop=True)
        seasonal_df.to_csv(os.path.join(self.raw_dir, "weather_seasonal.csv"), index=False)

        return daily_df, seasonal_df


# ============================================================
# NDVI Ingestor (Pure Python Earth Engine Authentication)
# ============================================================
class NDVIIngestor:
    def __init__(self, cfg: Dict, raw_dir: Optional[str] = None, ee_project="ee-alfeenubaidh2006"):
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict (ndvi config block)")

        self.start_year = int(cfg["start_year"])
        self.end_year = int(cfg["end_year"])
        self.states = list(cfg["states"])
        self.raw_dir = raw_dir or DEFAULT_RAW_DIR
        ensure_dir(self.raw_dir)
        self.ee_project = ee_project

    def _init_ee(self):
        import ee

        # Try normal init
        try:
            ee.Initialize(project=self.ee_project)
            return ee
        except Exception:
            pass

        # Fall back to Python authentication (no CLI needed)
        try:
            ee.Authenticate()
            ee.Initialize(project=self.ee_project)
            return ee
        except Exception as exc:
            raise RuntimeError(
                "Google Earth Engine authentication failed using Python. "
                "Make sure your Google account has GEE access."
            ) from exc

    def ingest(self) -> pd.DataFrame:
        ee = self._init_ee()

        gaul = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(
            ee.Filter.eq("ADM0_NAME", "India")
        )
        roi = gaul.filter(ee.Filter.inList("ADM1_NAME", self.states))

        modis = (
            ee.ImageCollection("MODIS/061/MOD13A1")
            .select("NDVI")
            .map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
        )

        images = []
        for y in range(self.start_year, self.end_year + 1):
            for m in range(1, 12 + 1):
                subset = (
                    modis.filter(ee.Filter.calendarRange(y, y, "year"))
                         .filter(ee.Filter.calendarRange(m, m, "month"))
                )
                if subset.size().getInfo() == 0:
                    continue
                images.append(subset.mean().set({"year": y, "month": m}))

        if not images:
            raise RuntimeError("No MODIS images found")

        monthly = ee.ImageCollection.fromImages(images)

        def reducer(img):
            reduced = img.reduceRegions(
                collection=roi,
                reducer=ee.Reducer.mean(),
                scale=500,
            )
            return reduced.map(lambda f: f.set({
                "year": img.get("year"),
                "month": img.get("month")
            }))

        stats = monthly.map(reducer).flatten()

        df = pd.DataFrame({
            "State": stats.aggregate_array("ADM1_NAME").getInfo(),
            "Year": stats.aggregate_array("year").getInfo(),
            "Month": stats.aggregate_array("month").getInfo(),
            "NDVI": stats.aggregate_array("mean").getInfo(),
        })

        df.to_csv(
            os.path.join(self.raw_dir, f"NDVI_states_{self.start_year}_{self.end_year}.csv"),
            index=False
        )

        return df


# ============================================================
# Soil Organic Carbon Ingester
# ============================================================
class SoilOrganicCarbonIngestor:
    def __init__(self, cfg: Dict, raw_dir: Optional[str] = None):
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict (soc config)")

        self.shapefile = cfg["shapefile"]
        self.raster = cfg["raster"]
        self.states = list(cfg["states_of_interest"])
        self.raw_dir = raw_dir or DEFAULT_RAW_DIR
        ensure_dir(self.raw_dir)

    @staticmethod
    def _create_grid(geom, cell=0.1):
        from shapely.geometry import box
        minx, miny, maxx, maxy = geom.bounds
        grid = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                cell_geom = box(x, y, x + cell, y + cell)
                if geom.intersects(cell_geom):
                    grid.append(cell_geom.intersection(geom))
                y += cell
            x += cell
        return grid

    def ingest(self):
        import geopandas as gpd
        from rasterstats import zonal_stats

        gdf = gpd.read_file(self.shapefile)
        gdf = gdf[gdf["NAME_1"].isin(self.states)]
        if gdf.empty:
            raise RuntimeError("States not found in shapefile")

        cells = []
        for row in gdf.itertuples():
            for c in self._create_grid(row.geometry):
                cells.append({"State": row.NAME_1, "geometry": c})

        grid = gpd.GeoDataFrame(cells, crs=gdf.crs)

        stats = zonal_stats(grid, self.raster, stats=["mean", "min", "max", "median", "std"])

        df = pd.DataFrame([{
            "State": r.State,
            "Mean_SOC": s["mean"],
            "Median_SOC": s["median"],
            "Min_SOC": s["min"],
            "Max_SOC": s["max"],
            "Std_SOC": s["std"],
        } for s, r in zip(stats, grid.itertuples())])

        df.to_csv(os.path.join(self.raw_dir, "india_states_soc_stats.csv"), index=False)
        return df


# ============================================================
# Yield Ingester
# ============================================================
class YieldIngestor:
    def __init__(self, cfg: Dict, raw_dir: Optional[str] = None):
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict (yield config)")
        self.path = cfg["path"]
        self.raw_dir = raw_dir or DEFAULT_RAW_DIR
        ensure_dir(self.raw_dir)

    @staticmethod
    def _extract_year_and_season(x):
        try:
            year = int(x.split("-")[0].strip())
            return year, "Rabi"
        except Exception:
            return None, None

    def ingest(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)

        df = pd.read_csv(self.path)
        df.columns = [c.strip() for c in df.columns]

        df["Year"], df["Season"] = zip(*df["Year"].apply(self._extract_year_and_season))
        df.rename(columns={"Yield (Tonne/Hectare)": "Yield"}, inplace=True)

        df = df.dropna(subset=["Year"])
        df = df[["State", "Year", "Season", "Yield"]]

        return df
