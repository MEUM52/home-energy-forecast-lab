"""Regression evaluation metrics in the target unit (Wh)."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred):
    """Return MAE and RMSE, both in Wh for Appliances targets."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Gerçek ve tahmin dizilerinin boyutları aynı olmalı.")
    if actual.size == 0:
        raise ValueError("Değerlendirme için en az bir örnek gerekli.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Gerçek veya tahmin değerlerinde sonlu olmayan sayı var.")
    return {
        "mae_wh": float(mean_absolute_error(actual, predicted)),
        "rmse_wh": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
