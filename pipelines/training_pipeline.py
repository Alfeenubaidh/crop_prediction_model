# pipelines/training_pipeline.py

from zenml import pipeline

from steps.data_ingestion_step import data_ingestion
from steps.dataset_merger_step import merge_data
from steps.handle_missing_values_step import handle_missing_values_step
from steps.outlier_detection_step import handle_outliers_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import data_split_step
from steps.model_training_step import training_run
from steps.model_evaluate_step import evaluation_run


@pipeline(enable_cache=False)
def training_pipeline(config: dict):

    # ======================================================
    # 1. DATA INGESTION
    # ======================================================
    (
        weather_daily,
        weather_seasonal,
        ndvi,
        soc,
        yield_df,
        simulated_df,
    ) = data_ingestion(config=config)

    # ======================================================
    # 2. MERGE
    # ======================================================
    merged = merge_data(
        weather_seasonal=weather_seasonal,
        ndvi=ndvi,
        soc=soc,
        yield_df=yield_df,
    )

    # ======================================================
    # 3. MISSING VALUES
    # ======================================================
    cleaned = handle_missing_values_step(
        data=merged,
        config=config,
    )

    # ======================================================
    # 4. OUTLIERS
    # ======================================================
    no_outliers = handle_outliers_step(
        data=cleaned,
        config=config,
    )

    # ======================================================
    # 5. FEATURE ENGINEERING
    # ======================================================
    features = feature_engineering_step(
        data=no_outliers,
        config=config,
    )

    # ======================================================
    # 6. TRAIN / VAL / TEST SPLIT
    # ======================================================
    X_train, X_val, X_test, y_train, y_val, y_test = data_split_step(
        cleaned_df=features
    )

    # ======================================================
    # 7. MODEL TRAINING
    # ======================================================
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
    )

    # ======================================================
    # 8. MODEL EVALUATION (STUDENT MODEL)
    # ======================================================
    evaluation_results = evaluation_run(
        model_path=student_model_path,
        encoder_path=encoder_path,
        X_test=X_test_out,
        y_test=y_test_out,
    )

    return evaluation_results
