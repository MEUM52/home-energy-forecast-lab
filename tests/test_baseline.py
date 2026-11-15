"""Tests for the persistence baseline and regression metrics."""

import numpy as np
import pandas as pd
import pytest

from energy_forecast.machine_learning.baseline import predict_last_measurement
from energy_forecast.machine_learning.evaluation import regression_metrics


def test_baseline_uses_latest_observed_appliances():
    frame = pd.DataFrame({"appliances_lag_0": [30.0, 42.0, 18.0]})
    np.testing.assert_array_equal(predict_last_measurement(frame), [30.0, 42.0, 18.0])


def test_baseline_requires_latest_measurement():
    with pytest.raises(ValueError, match="appliances_lag_0"):
        predict_last_measurement(pd.DataFrame({"other": [1]}))


def test_metrics_are_reported_in_target_units():
    result = regression_metrics([10, 20], [13, 16])
    assert result["mae_wh"] == pytest.approx(3.5)
    assert result["rmse_wh"] == pytest.approx(np.sqrt(12.5))


def test_metrics_reject_different_shapes():
    with pytest.raises(ValueError, match="boyutları aynı"):
        regression_metrics([1, 2], [1])
