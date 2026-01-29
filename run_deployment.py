import yaml
from pathlib import Path

from pipelines.deployment_pipeline import deployment_pipeline


def load_config():
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError("config.yaml not found.")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    # 1. Path to inference data (STRING ONLY)
    inference_input_path = cfg["paths"].get(
        "inference_input", "data/new_input.csv"
    )

    if not Path(inference_input_path).exists():
        raise FileNotFoundError(f"Input file not found: {inference_input_path}")

    # 2. Model path
    model_path = cfg["paths"].get(
        "deployment_model", "models/student_lightgbm.joblib"
    )

    # 3. Run pipeline
    deployment_pipeline(
        inference_input_path=inference_input_path,
        model_path_override=model_path,
        config=cfg,
    )

    print("\n========== DEPLOYMENT PIPELINE COMPLETED ==========\n")


if __name__ == "__main__":
    main()
