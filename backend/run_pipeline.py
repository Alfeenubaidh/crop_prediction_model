# run_pipeline.py
from ml.src.utils.config_loader import load_config
from ml.pipelines.training_pipeline import training_pipeline


def main():
    config = load_config("conifgs/config.yaml")
    training_pipeline(config=config)


if __name__ == "__main__":
    main()
