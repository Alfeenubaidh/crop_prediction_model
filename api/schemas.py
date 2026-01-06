from pydantic import BaseModel
from typing import Optional


class PredictionRequest(BaseModel):
    State: str
    Year: int
    Season: str

    T2M: float
    T2M_MAX: float
    T2M_MIN: float
    PRECTOTCORR: float
    ALLSKY_SFC_SW_DWN: float
    RH2M: float
    WS2M: float

    NDVI_SeasonalMean: float

    Mean_SOC: float
    Median_SOC: float
    Min_SOC: float
    Max_SOC: float
    Std_SOC: float


class PredictionResponse(BaseModel):
    predicted_yield: float
