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

from ml.src.feature_engineering import FeatureEngineering

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logger = get_logger(__name__)
CONFIG_PATH = "configs/config.yaml"


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
    train_df_raw: pd.DataFrame | None = None,   # full train split (with Yield) to refit FE for saving
) -> Tuple[str, str, str, pd.DataFrame, pd.Series]:

    logger.info("========== TRAINING STEP STARTED ==========")

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    model_dir = paths.get("model_dir", "models")
    os.makedirs(model_dir, exist_ok=True)

    TEACHER_NAME = paths.get("teacher_model_name", "teacher_stacking.joblib")
    STUDENT_NAME = paths.get("student_model_name", "student_lightgbm.joblib")

    PREPROCESSOR_NAME        = "preprocessor.joblib"
    FE_NAME                  = "feature_engineering.joblib"   # fitted FeatureEngineering
    FEATURE_SCHEMA_JSON      = "feature_schema.json"
    FEATURE_COLUMNS_JOBLIB   = "feature_columns.joblib"

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
    # KFold with shuffle=False preserves temporal order in cross-validation.
    # Default cv=5 uses shuffled KFold — on a time-series dataset this lets
    # future years appear in training folds, which is the primary leak source
    # inflating R² to 0.94. TimeSeriesSplit is the correct alternative but
    # KFold(shuffle=False) is a safe minimum fix that respects row order.
    temporal_cv = KFold(n_splits=5, shuffle=False)

    # Base learner complexity reduced to match dataset size (~120 train rows).
    # 600-tree XGB + 500-tree LGBM + 400-tree RF on 120 rows = guaranteed overfit.
    base_learners = [
        ("xgb", XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, verbosity=0
        )),
        ("lgbm", LGBMRegressor(
            n_estimators=100, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, subsample=0.8, verbose=-1
        )),
        ("rf", RandomForestRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=3, n_jobs=-1
        )),
        ("ridge", Ridge(alpha=1.0)),
    ]

    meta = Ridge(alpha=1.0)   # linear meta-learner — prevents meta overfitting

    stack = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta,
        passthrough=False,     # passthrough=True doubles feature space → overfits on small data
        cv=temporal_cv,
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

    # ---- Save fitted FeatureEngineering object for inference ----
    # Re-fit on train_df_raw (full rows including Yield) so inference can call
    # fe.transform(new_data) with the exact same group medians and statistics
    # that were used during training. Without this, predict.py would create a
    # fresh unfitted FE instance — producing wrong anomaly/interaction features.
    fe_path = os.path.join(model_dir, FE_NAME)
    if train_df_raw is not None and not train_df_raw.empty:
        cfg_for_fe = cfg  # already loaded above
        fe_for_save = FeatureEngineering(cfg_for_fe)
        fe_for_save.fit(train_df_raw)
        joblib.dump(fe_for_save, fe_path)
        logger.info("Fitted FeatureEngineering saved -> %s", fe_path)
    else:
        logger.warning(
            "train_df_raw not passed to training_run — "
            "FeatureEngineering object NOT saved. Inference will use wrong feature values."
        )

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
        n_estimators=student_cfg.get("n_estimators", 200),
        learning_rate=student_cfg.get("learning_rate", 0.05),
        num_leaves=student_cfg.get("num_leaves", 15),
        min_child_samples=student_cfg.get("min_child_samples", 5),
        reg_alpha=student_cfg.get("reg_alpha", 0.1),
        reg_lambda=student_cfg.get("reg_lambda", 0.5),
        subsample=student_cfg.get("subsample", 0.8),
        colsample_bytree=student_cfg.get("colsample_bytree", 0.8),
        n_jobs=-1,
        verbose=-1,
    )

    logger.info("Training STUDENT model...")
    student.fit(X_train_enc_df, y_train)

    student_path = os.path.join(model_dir, STUDENT_NAME)
    joblib.dump(student, student_path)

    # ---------------- CONFORMAL CALIBRATION ----------------
    # Split conformal prediction: calibrate on held-out val residuals so that
    # P(y_true in [y_pred - q, y_pred + q]) >= coverage for exchangeable data.
    # Coverage is approximate for time-series (no strict exchangeability), but
    # empirically valid and far better than a naive ±sigma heuristic.
    X_val_enc = fitted_preprocessor.transform(X_val)
    X_val_enc_df = pd.DataFrame(X_val_enc, columns=feature_names)
    val_preds = student.predict(X_val_enc_df)
    residuals = np.sort(np.abs(y_val.to_numpy() - val_preds))
    n_cal = len(residuals)

    conformal_quantiles: dict[str, float] = {}
    for coverage in [0.80, 0.90, 0.95]:
        # Finite-sample conformal quantile: ceil((n+1)(1-alpha)) / n
        idx = int(np.ceil((n_cal + 1) * coverage)) - 1
        idx = min(idx, n_cal - 1)
        conformal_quantiles[str(coverage)] = float(residuals[idx])
        logger.info("Conformal q(%.0f%%): %.4f", coverage * 100, residuals[idx])

    conformal_path = os.path.join(model_dir, "conformal_quantiles.json")
    with open(conformal_path, "w") as f:
        json.dump(conformal_quantiles, f, indent=2)
    logger.info("Conformal quantiles saved -> %s", conformal_path)

    logger.info("========== TRAINING STEP FINISHED ==========")

    return (
        teacher_path,
        student_path,
        os.path.join(model_dir, PREPROCESSOR_NAME),
        X_test,
        y_test,
    )
    # fe_path is saved to disk at: os.path.join(model_dir, FE_NAME)
    # Load it in predict.py with: joblib.load(os.path.join(model_dir, "feature_engineering.joblib"))