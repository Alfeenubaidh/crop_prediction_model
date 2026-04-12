# pipelines/training_pipeline.py
# Kept in sync with ml/pipelines/training_pipeline.py (canonical ZenML entry).

from zenml import pipeline

from ml.steps.data_ingestion_step import data_ingestion
from ml.steps.dataset_merger_step import merge_data
from ml.steps.data_splitter_step import data_split_step
from ml.steps.handle_missing_values_step import handle_missing_values_step
from ml.steps.outlier_detection_step import handle_outliers_step
from ml.steps.feature_engineering_step import feature_engineering_step
from ml.steps.model_training_step import training_run
from ml.steps.export_test_set_step import export_test_set_step
from ml.steps.model_evaluate_step import evaluation_run


@pipeline(enable_cache=False)
def training_pipeline(config: dict):
    (
        weather_daily,
        weather_seasonal,
        ndvi,
        soc,
        yield_df,
        simulated_df,
    ) = data_ingestion(config=config)

    merged = merge_data(
        weather_seasonal=weather_seasonal,
        ndvi=ndvi,
        soc=soc,
        yield_df=yield_df,
        config=config,
    )

    train_df, val_df, test_df = data_split_step(merged_df=merged, config=config)

    train_mv, val_mv, test_mv = handle_missing_values_step(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
    )

    train_ol, val_ol, test_ol = handle_outliers_step(
        train_df=train_mv,
        val_df=val_mv,
        test_df=test_mv,
        config=config,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = feature_engineering_step(
        train_df=train_ol,
        val_df=val_ol,
        test_df=test_ol,
        config=config,
    )

    (
        teacher_model_path,
        student_model_path,
        encoder_path,
        X_test_out,
        y_test_out,
    ) = training_run(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_df_raw=train_ol,   # full train rows (with Yield) — used to save fitted FE for inference
    )

    export_test_set_step(
        X_test=X_test_out,
        y_test=y_test_out,
    )

    evaluation_results = evaluation_run(
        model_path=student_model_path,
        encoder_path=encoder_path,
        X_test=X_test_out,
        y_test=y_test_out,
    )

    return evaluation_results