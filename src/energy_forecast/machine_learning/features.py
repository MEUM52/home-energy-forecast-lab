"""Leakage-safe supervised feature engineering for 10-minute forecasts."""

import pandas as pd

EXPECTED_INTERVAL = pd.Timedelta(minutes=10)
META_COLUMNS = ["origin_time", "target_time", "target_appliances_wh"]


def build_supervised_dataset(data, lag_steps=(0, 1, 2, 3, 6, 12), rolling_windows=(3, 6, 12)):
    """Predict Appliances at t+10 using only measurements available at/before t."""
    required = {"date", "Appliances"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Gerekli sütunlar eksik: {', '.join(missing)}")
    if not lag_steps or any(not isinstance(x, int) or x < 0 for x in lag_steps):
        raise ValueError("lag_steps sıfır veya pozitif tam sayılardan oluşmalı.")
    if not rolling_windows or any(not isinstance(x, int) or x < 1 for x in rolling_windows):
        raise ValueError("rolling_windows pozitif tam sayılardan oluşmalı.")
    if len(set(lag_steps)) != len(lag_steps) or len(set(rolling_windows)) != len(rolling_windows):
        raise ValueError("Lag ve rolling pencerelerinde tekrar eden değer olmamalı.")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("'date' sütununda geçersiz tarih var.")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Yinelenen zaman damgaları bulundu.")
    if len(frame) < 2:
        raise ValueError("Hedef oluşturmak için en az iki satır gerekli.")
    if not (frame["date"].diff().dropna() == EXPECTED_INTERVAL).all():
        raise ValueError("Veri satırları arasında tam 10 dakika olmalı.")

    numeric_columns = [c for c in frame.columns if c != "date"]
    non_numeric = [c for c in numeric_columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError("Girdi sütunları sayısal olmalı: " + ", ".join(non_numeric))
    if frame[numeric_columns].isna().any().any():
        raise ValueError("Girdi sütunlarında eksik değer bulundu.")

    result = pd.DataFrame(index=frame.index)
    result["origin_time"] = frame["date"]
    result["target_time"] = frame["date"].shift(-1)

    # Contemporaneous exogenous measurements at t only.
    for column in numeric_columns:
        if column != "Appliances":
            result[f"current__{column}"] = frame[column]

    # Lag 0 is the latest Appliances observation at t.
    for step in lag_steps:
        result[f"appliances_lag_{step}"] = frame["Appliances"].shift(step)

    # Right-aligned rolling windows include t and earlier observations only.
    for window in rolling_windows:
        result[f"appliances_rolling_mean_{window}"] = (
            frame["Appliances"].rolling(window=window, min_periods=window).mean()
        )

    # Next row is used only as the label, never as an input feature.
    result["target_appliances_wh"] = frame["Appliances"].shift(-1)
    result = result.dropna().reset_index(drop=True)
    if result.empty:
        raise ValueError("Özellik pencereleri için yeterli geçmiş veri bulunamadı.")
    return result


def chronological_split(supervised_data, train_ratio=0.70, validation_ratio=0.15, test_ratio=0.15):
    """Split chronologically by target_time without shuffling."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(not 0 < ratio < 1 for ratio in ratios):
        raise ValueError("Bölüm oranlarının her biri 0 ile 1 arasında olmalı.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Train, validation ve test oranlarının toplamı 1 olmalı.")
    if "target_time" not in supervised_data.columns:
        raise ValueError("'target_time' sütunu bulunamadı.")

    ordered = supervised_data.sort_values("target_time").reset_index(drop=True)
    if ordered["target_time"].duplicated().any():
        raise ValueError("target_time değerleri benzersiz olmalı.")
    count = len(ordered)
    train_end = int(count * train_ratio)
    validation_end = int(count * (train_ratio + validation_ratio))
    if min(train_end, validation_end - train_end, count - validation_end) < 1:
        raise ValueError("Her bölümde en az bir satır olması için daha fazla veri gerekli.")
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def get_feature_columns(supervised_data):
    """Return predictors, excluding target and timestamp metadata."""
    excluded = set(META_COLUMNS)
    return [column for column in supervised_data.columns if column not in excluded]
