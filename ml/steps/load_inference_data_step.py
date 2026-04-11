from zenml import step
from ml.src.inference.load import load_inference_data
import pandas as pd

@step
def load_inference_data_step(inference_input_path: str) -> pd.DataFrame:
    return load_inference_data(inference_input_path)
