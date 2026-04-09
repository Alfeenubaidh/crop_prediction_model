from zenml import step
from pathlib import Path
import pandas as pd

from src.utils.config_loader import load_config


@step(enable_cache=False)
def export_test_set_step(
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Explicitly export feature-engineered test set for experiments.
    Resolves project root from paths.project_root in config.yaml.
    """

    # --------------------------------------------------
    # Load config inside step (ZenML-safe)
    # --------------------------------------------------
    config = load_config("config.yaml")

    try:
        root_dir = config["paths"]["project_root"]
    except KeyError as e:
        raise KeyError(
            "Missing 'paths.project_root' in config.yaml. "
            "Required for exporting experiment artifacts."
        ) from e

    ROOT = Path(root_dir).resolve()
    MODELS_DIR = ROOT / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Export artifacts
    # --------------------------------------------------
    X_path = MODELS_DIR / "X_test_features.csv"
    y_path = MODELS_DIR / "y_test.csv"

    X_test.to_csv(X_path, index=False)
    y_test.to_csv(y_path, index=False)

    print(f"[EXPORT] X_test_features saved to {X_path}")
    print(f"[EXPORT] y_test saved to {y_path}")
