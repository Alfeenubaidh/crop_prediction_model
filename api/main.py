from fastapi import FastAPI
from api.schemas import PredictionRequest, PredictionResponse
from api.model_loader import ModelLoader
from api.predict import Predictor
from api.feature_builder import FeatureBuilder

MODEL_DIR = "models"

app = FastAPI(
    title="Crop Yield Prediction API",
    version="1.0.0"
)

# ---------------- LOAD ARTIFACTS ----------------
model = ModelLoader().load()

feature_builder = FeatureBuilder(
    preprocessor_path=f"{MODEL_DIR}/preprocessor.joblib",
    feature_schema_path=f"{MODEL_DIR}/feature_schema.json",
)

predictor = Predictor(model, feature_builder)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_yield(request: PredictionRequest):
    prediction = predictor.predict(request.dict())
    return PredictionResponse(predicted_yield=prediction)
