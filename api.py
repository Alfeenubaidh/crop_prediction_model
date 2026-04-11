"""
api.py — Crop Yield Prediction API
Run: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── ensure project root is importable ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.inference.predict import run_prediction

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
MODELS_DIR       = ROOT / "models"
MODEL_PATH       = str(MODELS_DIR / "student_lightgbm.joblib")
ENCODER_PATH     = str(MODELS_DIR / "preprocessor.joblib")
FE_PATH          = str(MODELS_DIR / "feature_engineering.joblib")
CONFORMAL_PATH   = str(MODELS_DIR / "conformal_quantiles.json")
EVAL_REPORT_PATH = ROOT / "evaluation" / "evaluation_report.json"
CONFIG_PATH      = ROOT / "config.yaml"

# ── minimal runtime config (populated on startup) ───────────────────────────
_cfg: Dict = {}

# ── lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg

    # Load config.yaml if present, else use minimal defaults
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            _cfg = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", CONFIG_PATH)
    else:
        logger.warning("config.yaml not found — using default model paths")

    # Ensure inference paths are set
    _cfg.setdefault("paths", {})
    _cfg["paths"].setdefault("fe_path", FE_PATH)
    _cfg["paths"].setdefault("conformal_quantiles_path", CONFORMAL_PATH)

    # Verify artifacts exist at startup
    missing = [p for p in (MODEL_PATH, ENCODER_PATH, FE_PATH) if not Path(p).exists()]
    if missing:
        logger.error("Missing model artifacts: %s", missing)
    else:
        logger.info("All model artifacts found — API ready")

    yield


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crop Yield Prediction API",
    description=(
        "Leakage-safe crop yield prediction for 5 Indian states "
        "(Punjab, Haryana, Rajasthan, Uttar Pradesh, Chandigarh). "
        "Teacher: Stacking Regressor — Student: LightGBM (deployed)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class PredictRequest(BaseModel):
    # Required location / time fields
    state:        str           = Field(..., description="State name (e.g. PUNJAB, HARYANA)")
    year:         int           = Field(..., description="Crop year (e.g. 2022)")
    season:       str           = Field(..., description="Season: Kharif | Rabi | Zaid")

    # Optional climate fields — sent by the form via buildRequest()
    T2M:          Optional[float] = Field(None, description="Mean temperature (°C)")
    T2M_MAX:      Optional[float] = Field(None, description="Max temperature (°C)")
    T2M_MIN:      Optional[float] = Field(None, description="Min temperature (°C)")
    PRECTOTCORR:  Optional[float] = Field(None, description="Total rainfall (mm)")
    ALLSKY_SFC_SW_DWN: Optional[float] = Field(None, description="Solar radiation (MJ/m²/day)")
    RH2M:         Optional[float] = Field(None, description="Relative humidity (%)")
    WS2M:         Optional[float] = Field(None, description="Wind speed (m/s)")

    # Optional lag fields
    yield_lag1:   Optional[float] = Field(None, description="Prior-year yield (t/ha)")
    yield_lag2:   Optional[float] = Field(None, description="Two-years-prior yield (t/ha)")


class ConformalInterval(BaseModel):
    coverage:   float
    lower:      float
    upper:      float
    half_width: float


class PredictResponse(BaseModel):
    state:           str
    year:            int
    season:          str
    predicted_yield: float               = Field(..., description="Predicted yield")
    intervals:       List[ConformalInterval] = Field(..., description="Conformal prediction intervals")
    risk_level:      str                 = Field(..., description="Low | Medium | High")
    model:           str                 = "student_lightgbm"


# ============================================================
# HELPERS
# ============================================================

def _to_dataframe(req: PredictRequest) -> pd.DataFrame:
    """Map flat API request fields to the raw feature DataFrame expected by run_prediction."""
    row: Dict = {
        "State":  req.state.upper(),
        "Year":   req.year,
        "Season": req.season,
    }
    # Include whichever climate fields were provided; missing ones become NaN
    # and are handled gracefully by FeatureEngineering / run_prediction (filled → 0).
    for api_col, df_col in (
        ("T2M",               "T2M"),
        ("T2M_MAX",           "T2M_MAX"),
        ("T2M_MIN",           "T2M_MIN"),
        ("PRECTOTCORR",       "PRECTOTCORR"),
        ("ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DWN"),
        ("RH2M",              "RH2M"),
        ("WS2M",              "WS2M"),
    ):
        val = getattr(req, api_col)
        if val is not None:
            row[df_col] = val
    return pd.DataFrame([row])


def _risk_level(req: PredictRequest) -> str:
    t_max = req.T2M_MAX or req.T2M or 0.0
    rain  = req.PRECTOTCORR or 0.0
    if t_max > 40:
        return "High"
    if t_max > 35 or rain < 300:
        return "Medium"
    return "Low"


# Conformal quantiles keyed by integer label → coverage fraction
_COVERAGE_MAP = {"80": 0.8, "90": 0.9, "95": 0.95}


def _format_response(output_row: pd.Series, req: PredictRequest) -> PredictResponse:
    """Build a PredictResponse from the output row returned by run_prediction."""
    predicted_yield = round(float(output_row["Predicted_Yield"]), 4)

    # Load half-widths from conformal_quantiles.json so we can include them
    half_widths: Dict[str, float] = {}
    if Path(CONFORMAL_PATH).exists():
        with open(CONFORMAL_PATH) as f:
            raw: Dict = json.load(f)
        # Keys in the file are coverage fractions as strings: "0.8", "0.9", "0.95"
        for k, v in raw.items():
            label = str(int(float(k) * 100))   # "0.8" → "80"
            half_widths[label] = float(v)

    intervals: List[ConformalInterval] = []
    for label, coverage in _COVERAGE_MAP.items():
        lower_col = f"Yield_Lower_{label}"
        upper_col = f"Yield_Upper_{label}"
        if lower_col in output_row.index and upper_col in output_row.index:
            intervals.append(ConformalInterval(
                coverage=coverage,
                lower=round(float(output_row[lower_col]), 4),
                upper=round(float(output_row[upper_col]), 4),
                half_width=round(half_widths.get(label, 0.0), 4),
            ))

    return PredictResponse(
        state=req.state.upper(),
        year=req.year,
        season=req.season,
        predicted_yield=predicted_yield,
        intervals=intervals,
        risk_level=_risk_level(req),
    )


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", tags=["System"])
def health():
    """Liveness probe."""
    artifacts = {
        "model":    Path(MODEL_PATH).exists(),
        "encoder":  Path(ENCODER_PATH).exists(),
        "feature_engineering": Path(FE_PATH).exists(),
    }
    status = "ok" if all(artifacts.values()) else "degraded"
    return {"status": status, "artifacts": artifacts}


@app.get("/metrics", tags=["System"])
def metrics():
    """Return test-set evaluation metrics from the last training run."""
    if not EVAL_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Evaluation report not found. Run the training pipeline first.")
    with open(EVAL_REPORT_PATH) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """
    Single-row crop yield prediction.

    Lag features (Yield_Lag1/2, Yield_RollingMean_3) are filled with
    training group means when no lag history is provided — the model
    degrades gracefully for cold-start inputs.
    """
    try:
        features = _to_dataframe(request)
        output_df, _, _ = run_prediction(
            features=features,
            model_path=MODEL_PATH,
            encoder_path=ENCODER_PATH,
            config=_cfg,
            lag_history=None,
        )
        return _format_response(output_df.iloc[0], request)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifact missing: {exc}")
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=List[PredictResponse], tags=["Inference"])
def predict_batch(requests: List[PredictRequest]):
    """
    Batch crop yield prediction (up to 50 rows).

    Rows are sent as a JSON array. Each row is predicted independently
    (no shared lag history). For time-series forecasting with proper
    lag propagation, submit rows in chronological order via sequential
    /predict calls with lag_history from prior results.
    """
    if len(requests) > 50:
        raise HTTPException(status_code=422, detail="Batch size must not exceed 50 rows.")

    results: List[PredictResponse] = []
    errors: List[str] = []

    for i, req in enumerate(requests):
        try:
            features = _to_dataframe(req)
            output_df, _, _ = run_prediction(
                features=features,
                model_path=MODEL_PATH,
                encoder_path=ENCODER_PATH,
                config=_cfg,
                lag_history=None,
            )
            results.append(_format_response(output_df.iloc[0], req))
        except Exception as exc:
            errors.append(f"Row {i}: {exc}")

    if errors and not results:
        raise HTTPException(status_code=422, detail=errors)
    if errors:
        logger.warning("Batch: %d rows failed — %s", len(errors), errors)

    return results
