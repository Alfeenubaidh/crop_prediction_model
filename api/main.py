from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import UserInputRequest, PredictionResponse
from api.predict import predict_yield

app = FastAPI(
    title="Crop Yield Prediction API",
    description="Inference API for crop yield prediction with optional SHAP explainability",
    version="1.0.0",
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MAX_DATA_YEAR = 2022  # last year with available NDVI/weather data

# ------------------------------------------------------------
# CORS (safe default for dashboards)
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: UserInputRequest,
    explain: bool = Query(
        default=False,
        description="Set true to include SHAP explanation",
    ),
):
    try:
        # ----------------------------------------------------
        # Restrict predictions to available data years
        # ----------------------------------------------------
        if request.year > MAX_DATA_YEAR:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Prediction beyond {MAX_DATA_YEAR} is not supported. "
                    "Future-year prediction requires forecast NDVI/weather data."
                ),
            )

        # predict_yield already returns the final response dict
        return predict_yield(
            payload=request.dict(),
            explain=explain,
        )

    except HTTPException:
        # keep FastAPI errors unchanged
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
