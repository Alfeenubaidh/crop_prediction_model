from pipelines.training_pipeline import training_pipeline
from src.utils.config_loader import load_config

if __name__ == "__main__":
    # Load full config ONCE
    cfg = load_config("config.yaml")

    # Instantiate pipeline with SINGLE argument
    pipe = training_pipeline(config=cfg)

