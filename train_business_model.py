import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Processed" / "versions" / "merged_output.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)

FEATURE_MAP = {
    "avg_temperature": "T2M",
    "max_temperature": "T2M_MAX",
    "min_temperature": "T2M_MIN",
    "total_rainfall": "PRECTOTCORR",
    "solar_radiation": "ALLSKY_SFC_SW_DWN",
    "relative_humidity": "RH2M",
    "wind_speed": "WS2M",
    "soil_organic_carbon": "Mean_SOC",
    "ndvi_early": "NDVI_SeasonalMean",
}

TARGET = "Yield (Tonne/Hectare)"


# Build business feature matrix
X = pd.DataFrame()
for k, v in FEATURE_MAP.items():
    if v not in df.columns:
        raise ValueError(f"Missing column in CSV: {v}")
    X[k] = df[v]

y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

print("Train R²:", pipeline.score(X_train, y_train))
print("Test  R²:", pipeline.score(X_test, y_test))

joblib.dump(pipeline, MODEL_DIR / "student_lightgbm.joblib")

print("✅ Business model trained with 9 real features")
