"""Tests for leakage-safe feature engineering and chronological splitting."""

import numpy as np
import pandas as pd
import pytest

from energy_forecast.machine_learning.features import (
    build_supervised_dataset, chronological_split, get_feature_columns,
)


def make_data(row_count=40):
    dates = pd.date_range("2024-01-01", periods=row_count, freq="10min")
    return pd.DataFrame({
        "date": dates,
        "Appliances": np.arange(row_count, dtype=float) * 10,
        "T1": 20 + np.arange(row_count, dtype=float),
        "lights": np.arange(row_count, dtype=float) % 4,
    })


def test_target_is_next_ten_minute_measurement():
    data = make_data()
    prepared = build_supervised_dataset(data, lag_steps=(0, 1), rolling_windows=(2,))
    first = prepared.iloc[0]

    assert first["origin_time"] == data.loc[1, "date"]
    assert first["target_time"] == data.loc[2, "date"]
    assert first["target_time"] - first["origin_time"] == pd.Timedelta(minutes=10)
    assert first["target_appliances_wh"] == data.loc[2, "Appliances"]

    # Every prepared row must map its target to the raw measurement at t + 10.
    raw_values = data.set_index("date")["Appliances"]
    for row in prepared.itertuples():
        assert row.target_time - row.origin_time == pd.Timedelta(minutes=10)
        assert row.target_appliances_wh == raw_values.loc[row.target_time]


def test_features_do_not_use_future_sensor_values():
    original = make_data()
    changed = original.copy()
    changed.loc[20, ["T1", "lights"]] = [99999, 88888]
    kwargs = {"lag_steps": (0, 1, 2), "rolling_windows": (2, 3)}
    left = build_supervised_dataset(original, **kwargs)
    right = build_supervised_dataset(changed, **kwargs)
    timestamp = original.loc[19, "date"]
    left_row = left.loc[left["origin_time"] == timestamp].iloc[0]
    right_row = right.loc[right["origin_time"] == timestamp].iloc[0]
    columns = get_feature_columns(left)
    pd.testing.assert_series_equal(left_row[columns], right_row[columns])


def test_rolling_mean_uses_current_and_past_only():
    data = make_data(8)
    prepared = build_supervised_dataset(data, lag_steps=(0,), rolling_windows=(3,))
    row = prepared.loc[prepared["origin_time"] == data.loc[2, "date"]].iloc[0]
    assert row["appliances_rolling_mean_3"] == data.loc[0:2, "Appliances"].mean()


def test_target_and_timestamps_are_not_features():
    prepared = build_supervised_dataset(make_data(), lag_steps=(0, 1), rolling_windows=(2,))
    columns = get_feature_columns(prepared)
    assert "target_time" not in columns
    assert "target_appliances_wh" not in columns
    assert "current__T1" in columns
    assert "appliances_lag_0" in columns


def test_chronological_split_is_ordered_and_disjoint():
    prepared = build_supervised_dataset(make_data(60), lag_steps=(0, 1), rolling_windows=(2,))
    train, validation, test = chronological_split(prepared, 0.6, 0.2, 0.2)

    assert train["target_time"].is_monotonic_increasing
    assert validation["target_time"].is_monotonic_increasing
    assert test["target_time"].is_monotonic_increasing
    assert train["target_time"].max() < validation["target_time"].min()
    assert validation["target_time"].max() < test["target_time"].min()
    assert len(train) + len(validation) + len(test) == len(prepared)


def test_split_rejects_invalid_ratios():
    prepared = build_supervised_dataset(make_data(), lag_steps=(0, 1), rolling_windows=(2,))
    with pytest.raises(ValueError, match="toplamı 1"):
        chronological_split(prepared, 0.7, 0.2, 0.2)


def test_irregular_timestamps_are_rejected():
    data = make_data()
    data.loc[10, "date"] += pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="tam 10 dakika"):
        build_supervised_dataset(data)
