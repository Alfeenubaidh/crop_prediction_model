def risk_label(value: float) -> str:
    if value < 0.7:
        return "High"
    if value < 1.0:
        return "Medium"
    return "Low"


def yield_trend(predicted: float, baseline: float = 1.0) -> str:
    if predicted > baseline * 1.1:
        return "Above Average"
    if predicted < baseline * 0.9:
        return "Below Average"
    return "Normal"
