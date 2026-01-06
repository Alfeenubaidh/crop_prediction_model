# src/model_building.py

import joblib
import numpy as np
import lightgbm as lgb

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression


# ================================================================
#  UNIVERSAL EVALUATION FUNCTION
# ================================================================
def evaluate_regression(model, X, y):
    """Return RMSE, R², MAE as a tuple."""
    preds = model.predict(X)

    rmse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)

    return rmse, r2, mae


# ================================================================
#  TEACHER MODEL BUILDER (STACKING REGRESSOR)
#  NOTE: Teacher ONLY uses X_train, y_train — no validation needed.
# ================================================================
class TeacherModelBuilder:

    def __init__(self, config):
        teacher_cfg = config["model"]["teacher"]
        selected = teacher_cfg["base_models"]

        # Base model mapping
        base_map = {
            "RandomForestRegressor": RandomForestRegressor(n_estimators=250, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "Lasso": Lasso(alpha=0.001),
            "ElasticNet": ElasticNet(alpha=0.001),
        }

        # Build tuple list for StackingRegressor
        estimators = [(name, base_map[name]) for name in selected]

        # Final stacking regressor
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            n_jobs=-1
        )

    # TRAINING METHOD
    def train(self, X_train, y_train):
        """Train teacher model using only training data."""
        self.model.fit(X_train, y_train)
        return self.model

    # EVALUATION
    def evaluate(self, X, y):
        return evaluate_regression(self.model, X, y)


# ================================================================
#  STUDENT MODEL BUILDER (LIGHTGBM)
#  NOTE: Student *requires* validation data for early stopping.
# ================================================================
class StudentModelBuilder:

    def __init__(self, config):
        params = config["model"]["student"]["params"]

        # Initialize LightGBM model
        self.model = lgb.LGBMRegressor(
            **params,
            verbose=-1
        )

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train with validation to tune metrics.
        """
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            verbose=False
        )
        return self.model

    def evaluate(self, X, y):
        return evaluate_regression(self.model, X, y)


# ================================================================
#  SAVE / LOAD HELPERS
# ================================================================
def save_model(model, path):
    joblib.dump(model, path)
    return path


def load_model(path):
    return joblib.load(path)
