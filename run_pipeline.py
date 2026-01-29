# run_pipeline.py

from src.utils.config_loader import load_config
from pipelines.training_pipeline import training_pipeline


def main():
    config = load_config("config.yaml")

    # ZenML pipelines execute on call
    training_pipeline(config=config)


if __name__ == "__main__":
    main()
