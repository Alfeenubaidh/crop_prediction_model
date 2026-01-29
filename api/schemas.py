from pydantic import BaseModel
from typing import Optional, Dict, List


# =========================
# BUSINESS INPUT SCHEMA
# =========================
class ClimateInputs(BaseModel):
    avg_temperature: float
    max_temperature: float
    min_temperature: float
    total_rainfall: float
    solar_radiation: float
    relative_humidity: float
    wind_speed: float


class SoilInputs(BaseModel):
    soil_organic_carbon: float


class VegetationInputs(BaseModel):
    ndvi_early: float



class BusinessInputRequest(BaseModel):
    location_id: str
    agro_climatic_zone: str
    crop: str
    season: str

    climate: ClimateInputs
    soil: SoilInputs
    vegetation: VegetationInputs


# =========================
# BUSINESS OUTPUT
# =========================
class YieldPredictionResponse(BaseModel):
    expected_yield: float
    yield_lower: float
    yield_upper: float
    risk_level: str
    confidence: str
    key_drivers: List[str]
