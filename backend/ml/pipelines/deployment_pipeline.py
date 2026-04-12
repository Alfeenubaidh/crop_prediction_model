from zenml import pipeline
from ml.steps.predictor_step import predictor_step
from ml.steps.load_inference_data_step import load_inference_data_step
from steps.shap_explainer_step import shap_explainer_step


@pipeline(enable_cache=False)
def deployment_pipeline(
    inference_input_path: str,
    model_path_override: str | None = None,
    config: dict | None = None,
):
    # 1. Load inference CSV
    features = load_inference_data_step(inference_input_path)

    # 2. Predict
    predictions, X_enc, feature_names = predictor_step(
        features=features,
        model_path_override=model_path_override,
        config=config,
    )

    # 3. Explain predictions (SHAP)
    shap_df = shap_explainer_step(
        features_encoded=X_enc,
        model_path=model_path_override or "models/student_lightgbm.joblib",
        feature_names=feature_names,
    )

    return predictions, shap_df
