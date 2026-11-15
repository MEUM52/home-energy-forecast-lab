"""Persistence baseline: use the latest Appliances measurement as prediction."""


def predict_last_measurement(data):
    """Return y_hat(t+10) = Appliances(t), aligned to each input row."""
    column = "appliances_lag_0"
    if column not in data.columns:
        raise ValueError(f"Baseline için '{column}' özelliği gerekli.")
    return data[column].to_numpy(dtype=float)
