# steps/model_training_step.py

import os
import yaml
import json
import joblib
from typing import Tuple

import pandas as pd
import numpy as np

from zenml import step
from zenml.logger import get_logger

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logger = get_logger(__name__)
CONFIG_PATH = "config.yaml"


def _make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _get_feature_names(ct: ColumnTransformer, input_features):
    try:
        return ct.get_feature_names_out(input_features)
    except Exception:
        names = []
        for name, transformer, cols in ct.transformers_:
            if transformer == "drop":
                continue
            if transformer == "passthrough":
                names.extend(cols)
                continue
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out(cols))
            else:
                names.extend(cols)
        return names


@step(enable_cache=False)
def training_run(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[str, str, str, pd.DataFrame, pd.Series]:
    """
    RETURNS (STRICT ORDER):
    1. teacher_model_path
    2. student_model_path
    3. encoder_path
    4. X_test
    5. y_test
    """

    logger.info("========== TRAINING STEP STARTED ==========")

    # ---------------- CONFIG ----------------
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    model_dir = paths.get("model_dir", "models")
    os.makedirs(model_dir, exist_ok=True)

    TEACHER_NAME = paths.get("teacher_model_name", "teacher_stacking.joblib")
    STUDENT_NAME = paths.get("student_model_name", "student_lightgbm.joblib")
    ENCODER_NAME = "preprocessor.joblib"
    FEATURE_SCHEMA_NAME = "feature_schema.json"

    # ---------------- FEATURES ----------------
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_onehot_encoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0
    )

    # ---------------- TEACHER MODEL ----------------
    base_learners = [
        ("xgb", XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, verbosity=0)),
        ("lgbm", LGBMRegressor(n_estimators=500, learning_rate=0.05)),
        ("rf", RandomForestRegressor(n_estimators=400, n_jobs=-1)),
        ("ridge", Ridge(alpha=1.0)),
    ]

    meta = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, verbosity=0)

    stack = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta,
        passthrough=True,
        cv=5,
        n_jobs=-1,
    )

    teacher_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("stack", stack)
    ])

    logger.info("Training TEACHER model...")
    teacher_pipeline.fit(X_train, y_train)

    teacher_path = os.path.join(model_dir, TEACHER_NAME)
    joblib.dump(teacher_pipeline, teacher_path)

    fitted_preprocessor = teacher_pipeline.named_steps["preprocessor"]
    encoder_path = os.path.join(model_dir, ENCODER_NAME)
    joblib.dump(fitted_preprocessor, encoder_path)

    # ---------------- ENCODE DATA ----------------
    X_train_enc = fitted_preprocessor.transform(X_train)
    X_test_enc = fitted_preprocessor.transform(X_test)

    feature_names = _get_feature_names(fitted_preprocessor, X_train.columns)

    X_train_enc_df = pd.DataFrame(X_train_enc, columns=feature_names)
    X_test_enc_df = pd.DataFrame(X_test_enc, columns=feature_names)

    # ---------------- 🔒 FEATURE SCHEMA LOCK ----------------
    schema_path = os.path.join(model_dir, FEATURE_SCHEMA_NAME)
    with open(schema_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    # Also keep joblib version (optional but useful)
    joblib.dump(feature_names, os.path.join(model_dir, "feature_columns.joblib"))

    # ---------------- STUDENT MODEL ----------------
    student_cfg = cfg.get("model", {}).get("student", {}).get("params", {})
    student = LGBMRegressor(
        n_estimators=student_cfg.get("n_estimators", 300),
        learning_rate=student_cfg.get("learning_rate", 0.05),
        num_leaves=student_cfg.get("num_leaves", 31),
        n_jobs=-1,
    )

    logger.info("Training STUDENT model...")
    student.fit(X_train_enc_df, y_train)

    student_path = os.path.join(model_dir, STUDENT_NAME)
    joblib.dump(student, student_path)

    logger.info("========== TRAINING STEP FINISHED ==========")

    
    return (
        teacher_path,
        student_path,
        encoder_path,
        X_test,
        y_test,
    )
