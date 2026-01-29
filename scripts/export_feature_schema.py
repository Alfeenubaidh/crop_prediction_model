import joblib
import json
from pathlib import Path
import numpy as np

# Absolute project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load feature columns
feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")

# Convert numpy → list
if isinstance(feature_cols, np.ndarray):
    feature_cols = feature_cols.tolist()

# Save encoded schema
output_path = MODEL_DIR / "encoded_feature_schema.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=2)

print(f"Saved encoded feature schema to: {output_path}")
print(f"Total features: {len(feature_cols)}")
