"""Tests for the user-configurable high-consumption warning threshold."""

import pytest

from energy_forecast.prediction.alert import is_high_consumption


def test_alert_is_true_when_prediction_exceeds_threshold():
    assert is_high_consumption(200.01, 200.0) is True


def test_alert_is_false_when_prediction_equals_threshold():
    assert is_high_consumption(200.0, 200.0) is False


def test_alert_is_false_when_prediction_is_below_threshold():
    assert is_high_consumption(199.99, 200.0) is False


@pytest.mark.parametrize("threshold", [-1, float("inf"), float("nan")])
def test_alert_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        is_high_consumption(100.0, threshold)


def test_alert_rejects_non_finite_prediction():
    with pytest.raises(ValueError):
        is_high_consumption(float("nan"), 200.0)
