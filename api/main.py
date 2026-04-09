from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import traceback

from api.schemas import BusinessInputRequest, YieldPredictionResponse
from api.predict import predict_yield


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Crop Yield Intelligence API",
    description=(
        "Business-grade, scenario-based crop yield estimation API. "
        "Predictions are independent of calendar year and support "
        "forecasted climate, soil, and vegetation inputs."
    ),
    version="2.0.0",
)

# ============================================================
# CORS (safe default for dashboards & SaaS)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Health check
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# Prediction endpoint (BUSINESS)
# ============================================================
@app.post("/predict", response_model=dict)
def predict(
    request: BusinessInputRequest,
    explain: bool = Query(
        default=False,
        description="Set true to include SHAP-based explanation",
    ),
):
    """
    Scenario-based yield estimation.
    Accepts forecast or assumed climate, soil, and vegetation inputs.
    """

    try:
        # ----------------------------------------------------
        # CALL CORE BUSINESS INFERENCE
        # ----------------------------------------------------
        result = predict_yield(
            payload=request.dict(),
            explain=explain,
        )

        return result

    except HTTPException:
        # Let FastAPI-native errors pass through unchanged
        raise

    except Exception as e:
        # ----------------------------------------------------
        # 🔥 DO NOT HIDE THE REAL ERROR (CRITICAL FOR DEBUGGING)
        # ----------------------------------------------------
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
