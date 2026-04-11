import os
import joblib


class ModelLoader:
    """
    Loads the trained student model for inference.
    Automatically detects whether the model is a full sklearn Pipeline.
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

            # 🔍 CRITICAL DEBUG INFO
            if hasattr(self._model, "named_steps"):
                print(
                    "[ModelLoader] Loaded FULL sklearn Pipeline "
                    "(preprocessor + estimator)"
                )
            else:
                print("[ModelLoader] Loaded raw estimator (no preprocessing)")

        return self._model
