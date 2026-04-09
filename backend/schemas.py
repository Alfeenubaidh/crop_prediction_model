from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, Dict


# ============================================================
# INTERNAL SCHEMA (MODEL-READY FEATURES)
# ============================================================

class ModelFeatureRequest(BaseModel):
    """
    Internal schema.
    Used ONLY when features are already engineered
    (e.g., batch inference, offline experiments).
    """

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


# ============================================================
# USER-FACING SCHEMA (API / DASHBOARD INPUT)
# ============================================================

class UserInputRequest(BaseModel):
    """
    Human-facing input schema.
    Used by FastAPI and Streamlit dashboard.
    """

    state: str = Field(..., min_length=2, max_length=50)
    district: str = Field(..., min_length=2, max_length=50)

    crop: Literal[
        "Wheat",
        "Rice",
        "Maize",
        "Millet",
        "Barley",
    ]

    season: Literal[
        "Kharif",
        "Rabi",
        "Zaid",
    ]

    year: int = Field(..., ge=2000, le=2030)

    # ----------------------------
    # Normalization
    # ----------------------------
    @validator("state", "district")
    def normalize_text(cls, v: str) -> str:
        return v.strip().title()


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class PredictionResponse(BaseModel):
    predicted_yield: float
    explanation: Optional[Dict[str, float]] = None
