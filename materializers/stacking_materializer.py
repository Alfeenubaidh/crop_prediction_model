from typing import Type
from zenml.materializers.base_materializer import BaseMaterializer
from src.stacking_sklearn import SklearnStackingEnsembler
import joblib
import os

class StackingMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (SklearnStackingEnsembler,)
    ASSOCIATED_ARTIFACT_TYPES = None

    def load(self, data_type: Type[SklearnStackingEnsembler]):
        return joblib.load(os.path.join(self.uri, "teacher_stacking.joblib"))

    def save(self, model: SklearnStackingEnsembler):
        joblib.dump(model, os.path.join(self.uri, "teacher_stacking.joblib"))
