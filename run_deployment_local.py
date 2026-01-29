from steps.load_inference_data_step import load_inference_data_step
from steps.predictor_step import predictor_step
from steps.shap_explainer_step import shap_explainer_step
from src.utils.config_loader import load_config


def main():
    cfg = load_config("config.yaml")

    inference_input_path = cfg["paths"].get("inference_input")
    if not inference_input_path:
        raise ValueError("Missing 'paths.inference_input' in config.yaml")

    model_path = cfg["paths"]["deployment_model"]

    # 1. Load inference data (PURE PYTHON)
    features = load_inference_data_step.entrypoint(
        inference_input_path
    )

    # 2. Predict (PURE PYTHON)
    output, X_enc, feature_names = predictor_step.entrypoint(
        features=features,
        model_path_override=model_path,
        config=cfg,
    )

    # 3. SHAP explainability (PURE PYTHON)
    shap_explainer_step.entrypoint(
        features_encoded=X_enc,
        model_path=model_path,
        feature_names=feature_names,
    )

    print("Local deployment completed successfully.")


if __name__ == "__main__":
    main()
