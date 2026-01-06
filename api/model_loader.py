import os
import joblib
import yaml


class ModelLoader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_path = os.path.join(
            self.config["paths"]["project_root"],
            "models",
            "saved_model.pkl"
        )

        self.model = None

    def load(self):
        if self.model is None:
            self.model = joblib.load(self.model_path)
        return self.model
