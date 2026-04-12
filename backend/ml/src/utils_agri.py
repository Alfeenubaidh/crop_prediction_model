# src/utils_agri.py
from typing import Dict, List

def season_months() -> Dict[str, List[int]]:
    """
    Mapping of agricultural seasons to months (1-12).
    Kharif: June (6) - Sep (9)
    Rabi: Oct (10) - Mar (3) -- crosses year boundary
    Zaid: Apr (4) - May (5)
    """
    return {
        "Kharif": [6, 7, 8, 9],
        # Rabi crosses year boundary -> months listed across boundary
        "Rabi": [10, 11, 12, 1, 2, 3],
        "Zaid": [4, 5],
    }

def preseason_month_map() -> Dict[str, List[int]]:
    """
    Month(s) before season start used as 'preseason' indicator.
    Kharif preseason: May (5)
    Rabi preseason: Sep (9)
    Zaid preseason: Mar (3)
    """
    return {
        "Kharif": [5],
        "Rabi": [9],
        "Zaid": [3],
    }

def safe_str_strip(x):
    try:
        return str(x).strip()
    except Exception:
        return x
