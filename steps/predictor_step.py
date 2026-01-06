from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple, List

from src.inference.predict import run_prediction


@step(enable_cache=False)
def predictor_step(
    features: pd.DataFrame,
    model_path_override: str | None = None,
    config: dict | None = None,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    List[str],
]:

    cfg = config or {}
    paths = cfg.get("paths", {})

    model_path = model_path_override or paths.get(
        "student_model_path", "models/student_lightgbm.joblib"
    )
    encoder_path = paths.get(
        "encoder_path", "models/preprocessor.joblib"
    )

    output, X_enc, feature_names = run_prediction(
        features=features,
        model_path=model_path,
        encoder_path=encoder_path,
        config=cfg,
    )

    # File writing is allowed here (edge of pipeline)
    output.to_csv(
        "data/predictions/deployment_predictions.csv",
        index=False
    )

    return output, X_enc, feature_names
