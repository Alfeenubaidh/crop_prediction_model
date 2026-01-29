import os
import joblib


class ModelLoader:
    """
    Loads the trained student model for inference.
    """

    def __init__(self, model_path: str = "models/student_lightgbm.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self._model = None

    def load_model(self):
        """
        Load model once and cache it.
        """
        if self._model is None:
            self._model = joblib.load(self.model_path)
        return self._model
