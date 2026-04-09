# steps/model_training_step.py

import os
import yaml
import json
import joblib
from typing import Tuple
from pathlib import Path

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


def _build_feature_names(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer
) -> list[str]:
    """
    Deterministic feature name builder.
    NO sklearn introspection.
    """

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    feature_names = []

    # Numeric features
    feature_names.extend(num_cols)

    # Categorical expanded features
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_expanded = ohe.get_feature_names_out(cat_cols)
    feature_names.extend(cat_expanded.tolist())

    return feature_names


@step(enable_cache=False)
def training_run(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[str, str, str, pd.DataFrame, pd.Series]:

    logger.info("========== TRAINING STEP STARTED ==========")

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    model_dir = paths.get("model_dir", "models")
    os.makedirs(model_dir, exist_ok=True)

    TEACHER_NAME = paths.get("teacher_model_name", "teacher_stacking.joblib")
    STUDENT_NAME = paths.get("student_model_name", "student_lightgbm.joblib")

    PREPROCESSOR_NAME = "preprocessor.joblib"
    FEATURE_SCHEMA_JSON = "feature_schema.json"
    FEATURE_COLUMNS_JOBLIB = "feature_columns.joblib"

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

    # ---------------- TEACHER ----------------
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

    # Save teacher
    teacher_path = os.path.join(model_dir, TEACHER_NAME)
    joblib.dump(teacher_pipeline, teacher_path)

    # Save fitted preprocessor
    fitted_preprocessor = teacher_pipeline.named_steps["preprocessor"]
    joblib.dump(fitted_preprocessor, os.path.join(model_dir, PREPROCESSOR_NAME))

    # ---------------- ENCODE ----------------
    X_train_enc = fitted_preprocessor.transform(X_train)
    X_test_enc = fitted_preprocessor.transform(X_test)

    feature_names = _build_feature_names(X_train, fitted_preprocessor)

    X_train_enc_df = pd.DataFrame(X_train_enc, columns=feature_names)
    X_test_enc_df = pd.DataFrame(X_test_enc, columns=feature_names)

    # ---------------- SCHEMA LOCK ----------------
    with open(os.path.join(model_dir, FEATURE_SCHEMA_JSON), "w") as f:
        json.dump(feature_names, f, indent=2)

    joblib.dump(
        feature_names,
        os.path.join(model_dir, FEATURE_COLUMNS_JOBLIB)
    )

    # ---------------- STUDENT ----------------
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
        os.path.join(model_dir, PREPROCESSOR_NAME),
        X_test,
        y_test,
    )
