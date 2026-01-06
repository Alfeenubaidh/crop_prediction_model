# src/model_evaluation.py
import os
import json
import joblib
import yaml
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any, Dict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class ModelEvaluator:
    """
    Loads model + preprocessor (if required), runs predictions, computes metrics,
    and writes predictions + JSON report to disk. Config-driven.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # paths config
        paths = self.config.get("paths", {})
        self.project_root = paths.get("project_root", ".")
        self.model_dir = os.path.join(self.project_root, paths.get("model_dir", "models"))
        self.eval_dir = os.path.join(self.model_dir, paths.get("evaluation_dir", "evaluation"))
        os.makedirs(self.eval_dir, exist_ok=True)

        # model filenames from config (fall back to common defaults)
        model_cfg = self.config.get("model", {})
        teacher_cfg = model_cfg.get("teacher", {})
        student_cfg = model_cfg.get("student", {})

        self.teacher_name = teacher_cfg.get("save_name", teacher_cfg.get("save_name", "teacher_stacking.joblib"))
        self.student_name = student_cfg.get("save_name", student_cfg.get("save_name", "student_lightgbm.joblib"))

        # allow explicit overrides for filenames under top-level paths
        cfg_paths = self.config.get("paths", {})
        if "teacher_model_name" in cfg_paths:
            self.teacher_name = cfg_paths["teacher_model_name"]
        if "student_model_name" in cfg_paths:
            self.student_name = cfg_paths["student_model_name"]

        # default output filenames
        self.predictions_filename = self.config.get("evaluation", {}).get("predictions_filename", "evaluation_predictions.csv")
        self.report_filename = self.config.get("evaluation", {}).get("report_filename", "evaluation_report.json")

    # -------------------------
    # Loading utilities
    # -------------------------
    def _model_path(self, model_type: str = "student") -> str:
        if model_type == "teacher":
            return os.path.join(self.model_dir, self.teacher_name)
        return os.path.join(self.model_dir, self.student_name)

    def load_model(self, model_type: str = "student") -> Any:
        """
        Load a model from disk. Returns raw object (pipeline or estimator).
        """
        model_path = self._model_path(model_type)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)

    def load_preprocessor(self, explicit_preprocessor_path: Optional[str] = None) -> Optional[Any]:
        """
        Try to load a saved preprocessor:
          1. If explicit_preprocessor_path provided -> load it.
          2. Otherwise, look for preprocessor.joblib in model_dir.
          3. Otherwise, attempt to load teacher model and extract preprocessor/pipeline.
        Returns: preprocessor object (ColumnTransformer / Pipeline step) or None.
        """
        # explicit path
        if explicit_preprocessor_path:
            if os.path.exists(explicit_preprocessor_path):
                return joblib.load(explicit_preprocessor_path)
            else:
                raise FileNotFoundError(f"Explicit preprocessor not found: {explicit_preprocessor_path}")

        # common filename
        candidate = os.path.join(self.model_dir, "preprocessor.joblib")
        if os.path.exists(candidate):
            return joblib.load(candidate)

        # try to extract from teacher pipeline / object
        teacher_path = os.path.join(self.model_dir, self.teacher_name)
        if os.path.exists(teacher_path):
            obj = joblib.load(teacher_path)
            # common attribute names to try
            for attr in ("pipeline", "preprocess", "preprocessor", "preprocessor_step", "preprocessor__transformer_list"):
                if hasattr(obj, attr):
                    # If it's a full pipeline, return the preprocessing step
                    val = getattr(obj, attr)
                    # if pipeline -> try to return 'preprocess' step or the ColumnTransformer
                    # Several possible shapes; we return something with .transform
                    if hasattr(val, "transform"):
                        return val
            # Sometimes the teacher is a custom wrapper with property .preprocessor_step
            if hasattr(obj, "preprocessor_step"):
                return getattr(obj, "preprocessor_step")
            # fallback: if it's a sklearn Pipeline itself, try to extract named_steps
            if hasattr(obj, "named_steps"):
                # try to find a step that looks like preprocessing
                steps = getattr(obj, "named_steps")
                for name in ("preprocess", "preprocessor", "transformer"):
                    if name in steps:
                        return steps[name]
        # nothing found
        return None

    # -------------------------
    # Prediction / evaluation
    # -------------------------
    def prepare_input_for_model(self, model: Any, X: pd.DataFrame, preprocessor: Optional[Any] = None) -> Tuple[Any, bool]:
        """
        Returns (X_for_model, model_accepts_raw_df_flag)
        - If the model is a Pipeline, it can handle raw DataFrame -> return (X, True)
        - If not, and a preprocessor is provided, return (preprocessor.transform(X), False)
        - If no preprocessor and model is not a pipeline, raise error.
        """
        # If model is a sklearn Pipeline (has predict and will do preprocessing)
        if hasattr(model, "named_steps") or (hasattr(model, "predict") and hasattr(model, "fit") and hasattr(model, "transform") and hasattr(model, "steps")):
            # Some pipelines expose these attributes; treat as pipeline and pass raw X
            return X, True

        # If model is estimator that expects numeric numpy arrays
        if preprocessor is not None:
            # preprocessor must have transform
            if not hasattr(preprocessor, "transform"):
                raise ValueError("Provided preprocessor object does not implement transform().")
            X_num = preprocessor.transform(X)
            return X_num, False

        # If model itself can take a DataFrame (rare), assume it can; otherwise throw
        # We detect LightGBM / XGBoost scikit-learn wrappers that accept arrays only -> require preprocessor
        raise ValueError("Model appears to be a raw estimator that requires a numeric matrix. "
                         "Provide a preprocessor or save a pipeline that handles preprocessing.")

    def evaluate(self,
                 model: Any,
                 X: pd.DataFrame,
                 y: pd.Series,
                 preprocessor: Optional[Any] = None,
                 predictions_path: Optional[str] = None,
                 report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute predictions and compute metrics. Save outputs to disk and return metrics + paths.
        """
        # prepare X for model
        X_for_model, model_accepts_raw = self.prepare_input_for_model(model, X, preprocessor)

        # predict
        preds = model.predict(X_for_model)

        # ensure numpy array / pandas Series
        preds = np.asarray(preds).ravel()
        y_arr = np.asarray(y).ravel()

        # metrics
        rmse = float(np.sqrt(mean_squared_error(y_arr, preds)))
        r2 = float(r2_score(y_arr, preds))
        mae = float(mean_absolute_error(y_arr, preds))

        # prepare output files
        if predictions_path is None:
            predictions_path = os.path.join(self.eval_dir, self.predictions_filename)
        if report_path is None:
            report_path = os.path.join(self.eval_dir, self.report_filename)

        # save predictions CSV: include original X_test index if available
        df_out = X.copy().reset_index(drop=True)
        df_out["y_true"] = y_arr
        df_out["y_pred"] = preds
        df_out.to_csv(predictions_path, index=False)

        # save JSON report
        report = {
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "n_samples": int(len(y_arr)),
            "predictions_file": os.path.abspath(predictions_path)
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return {
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "predictions_path": os.path.abspath(predictions_path),
            "report_path": os.path.abspath(report_path)
        }

    # -------------------------
    # Convenience runner
    # -------------------------
    def run_from_files(self,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       model_type: str = "student",
                       explicit_preprocessor_path: Optional[str] = None) -> Dict[str, Any]:
        """
        High-level convenience function: load model + preprocessor (if needed),
        run evaluate() and return results.
        """
        model = self.load_model(model_type=model_type)

        # If model is a pipeline that already includes preprocessing, preprocessor can be None.
        # Otherwise try to find preprocessor
        preprocessor = None
        try:
            # attempt to discover a preprocessor only when the model is not a pipeline
            if not hasattr(model, "named_steps"):
                preprocessor = self.load_preprocessor(explicit_preprocessor_path)
        except Exception:
            preprocessor = None

        results = self.evaluate(model=model,
                                X=X_test,
                                y=y_test,
                                preprocessor=preprocessor)
        return results
