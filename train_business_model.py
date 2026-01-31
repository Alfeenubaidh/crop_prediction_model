import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data/Processed/versions/merged_output.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# Load data
# ============================================================
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "avg_temperature",
    "max_temperature",
    "min_temperature",
    "total_rainfall",
    "solar_radiation",
    "evapotranspiration",
    "soil_moisture",
    "soil_organic_carbon",
    "soil_nitrogen",
    "soil_ph",
    "ndvi_early",
    "evi_early",
]

TARGET = "yield_t_ha"


# 🔴 HARD CHECK
missing = set(FEATURES) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# Train / test split (NO TIME SPLIT)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# Pipeline
# ============================================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ))
])


# ============================================================
# Train
# ============================================================
pipeline.fit(X_train, y_train)

print("Train R²:", pipeline.score(X_train, y_train))
print("Test  R²:", pipeline.score(X_test, y_test))


# ============================================================
# Save (OVERWRITE OLD MODEL)
# ============================================================
joblib.dump(pipeline, MODEL_DIR / "student_lightgbm.joblib")

print("✅ Business model trained with 12 features")
