import pandas as pd

def load_inference_data(inference_input_path: str) -> pd.DataFrame:
    return pd.read_csv(inference_input_path)
