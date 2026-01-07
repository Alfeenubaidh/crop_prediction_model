import joblib
import json
from pathlib import Path
import numpy as np

# Load existing feature columns
feature_cols = joblib.load("models/feature_columns.joblib")

# Convert numpy array → Python list (CRITICAL FIX)
if isinstance(feature_cols, np.ndarray):
    feature_cols = feature_cols.tolist()

# Save as JSON
output_path = Path("models/encoded_feature_schema.json")
with output_path.open("w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"Saved encoded feature schema to {output_path}")
print(f"Total features: {len(feature_cols)}")
