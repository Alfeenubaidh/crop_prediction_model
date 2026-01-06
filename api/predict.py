class Predictor:
    def __init__(self, model, feature_builder):
        self.model = model
        self.feature_builder = feature_builder

    def predict(self, payload: dict) -> float:
        X = self.feature_builder.build(payload)
        prediction = self.model.predict(X)[0]
        return float(prediction)
