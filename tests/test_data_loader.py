"""Tests for UCI CSV loading and validation."""

import pandas as pd
import pytest

from energy_forecast.data_analysis.data_loader import (
    EXPECTED_COLUMNS,
    NUMERIC_COLUMNS,
    load_energy_data,
)


def make_valid_frame(dates=None):
    if dates is None:
        dates = pd.date_range("2024-01-01 00:00", periods=3, freq="10min")

    values = {
        column: [float(index + 1) for index in range(len(dates))]
        for column in NUMERIC_COLUMNS
    }
    values["Appliances"] = [10.0 * (index + 1) for index in range(len(dates))]
    values["date"] = dates
    return pd.DataFrame(values)[EXPECTED_COLUMNS]


def write_csv(tmp_path, frame):
    path = tmp_path / "energydata_complete.csv"
    frame.to_csv(path, index=False)
    return path


def test_loader_sorts_rows_and_parses_numeric_columns(tmp_path):
    dates = pd.to_datetime([
        "2024-01-01 00:20",
        "2024-01-01 00:00",
        "2024-01-01 00:10",
    ])
    frame = make_valid_frame(dates)
    path = write_csv(tmp_path, frame)

    loaded = load_energy_data(path)

    assert loaded["date"].is_monotonic_increasing
    assert loaded["date"].tolist() == sorted(dates.tolist())
    assert loaded["Appliances"].tolist() == [20.0, 30.0, 10.0]
    assert pd.api.types.is_numeric_dtype(loaded["T1"])


def test_loader_rejects_missing_required_column(tmp_path):
    frame = make_valid_frame().drop(columns=["rv2"])
    path = write_csv(tmp_path, frame)

    with pytest.raises(ValueError, match="rv2"):
        load_energy_data(path)


def test_loader_rejects_duplicate_timestamps(tmp_path):
    dates = pd.to_datetime([
        "2024-01-01 00:00",
        "2024-01-01 00:10",
        "2024-01-01 00:10",
    ])
    path = write_csv(tmp_path, make_valid_frame(dates))

    with pytest.raises(ValueError, match="yinelenen"):
        load_energy_data(path)


def test_loader_rejects_irregular_ten_minute_interval(tmp_path):
    dates = pd.to_datetime([
        "2024-01-01 00:00",
        "2024-01-01 00:10",
        "2024-01-01 00:25",
    ])
    path = write_csv(tmp_path, make_valid_frame(dates))

    with pytest.raises(ValueError, match="tam 10 dakikalık"):
        load_energy_data(path)


def test_loader_rejects_non_finite_numeric_values(tmp_path):
    frame = make_valid_frame()
    frame.loc[1, "T1"] = float("inf")
    path = write_csv(tmp_path, frame)

    with pytest.raises(ValueError, match="sonlu olmayan"):
        load_energy_data(path)


def test_loader_reports_missing_file(tmp_path):
    missing_path = tmp_path / "does-not-exist.csv"

    with pytest.raises(FileNotFoundError, match="Enerji veri seti bulunamadı"):
        load_energy_data(missing_path)
