"""Pure threshold logic for the AI application.""" 

import math


def is_high_consumption(predicted_wh: float, threshold_wh: float) -> bool:
    """Return True only when the prediction strictly exceeds a non-negative threshold."""
    try:
        predicted = float(predicted_wh)
        threshold = float(threshold_wh)
    except (TypeError, ValueError) as error:
        raise ValueError("Tahmin ve eşik sayısal değer olmalı.") from error

    if not math.isfinite(predicted) or not math.isfinite(threshold):
        raise ValueError("Tahmin ve eşik sonlu sayılar olmalı.")
    if threshold < 0:
        raise ValueError("Uyarı eşiği negatif olamaz.")

    return predicted > threshold
