# src/stacking_sklearn.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class SklearnStackingEnsembler:
    """
    Balanced-version Stacking model.
    This class behaves like a clean sklearn estimator.

    - No Yield column required in __init__
    - X, y passed ONLY during fit()
    """

    def __init__(self):
        self.pipeline = None
        self.cat_cols = None
        self.num_cols = None

    def _build_preprocessor(self, X: pd.DataFrame):
        """Detect columns and build preprocessing pipeline."""

        self.cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # IMPORTANT: sklearn 1.2+ uses 'sparse_output', not 'sparse'
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric, self.num_cols),
                ("cat", categorical, self.cat_cols),
            ],
            remainder="drop"
        )

        return preprocessor

    def _build_stacking_model(self):
        """Stacking model (balanced version)."""

        base_models = [
            ("xgb", XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )),
            ("lgbm", LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42
            )),
            ("rf", RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )),
            ("ridge", Ridge(alpha=1.0))
        ]

        meta_model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )

        stack = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            passthrough=True,
            cv=5
        )

        return stack

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train teacher/stacking model."""

        preprocessor = self._build_preprocessor(X)
        stack_model = self._build_stacking_model()

        self.pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("stack", stack_model)
        ])

        self.pipeline.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame):
        return self.pipeline.predict(X)

    def evaluate(self, X_test, y_test):
        preds = self.pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        return rmse, r2

    @property
    def model(self):
        return self.pipeline
