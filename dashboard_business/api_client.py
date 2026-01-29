# dashboard_business/api_client.py
import requests

API_URL = "http://127.0.0.1:8000/predict"


def get_prediction(payload: dict, explain: bool = False) -> dict:
    """
    Calls the business prediction API.

    Parameters:
    - payload: business input payload
    - explain: whether to request SHAP explanation

    Returns:
    - JSON response from API
    """

    params = {
        "explain": explain
    }

    response = requests.post(
        API_URL,
        json=payload,
        params=params,
        timeout=30,
    )

    response.raise_for_status()
    return response.json()
