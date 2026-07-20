import numpy as np
import pandas as pd

from src.forecasting import calculate_metrics, compare_models, create_forecast


def make_series(length=60):
    index = pd.date_range("2025-01-01", periods=length, freq="D")
    values = 4 + 0.03 * np.arange(length) + 1.2 * np.sin(np.arange(length) / 5)
    return pd.Series(np.maximum(0, values), index=index)


def test_calculate_metrics_returns_expected_values():
    metrics = calculate_metrics([1, 2, 3], [1, 2, 4])
    assert round(metrics["MAE"], 3) == 0.333
    assert round(metrics["RMSE"], 3) == 0.577
    assert metrics["sMAPE"] > 0


def test_manual_forecast_has_intervals_and_future_dates():
    series = make_series()
    bundle = create_forecast(
        series,
        steps=7,
        order=(1, 0, 0),
        automatic=False,
    )

    assert len(bundle.forecast) == 7
    assert bundle.forecast.index[0] == series.index[-1] + pd.Timedelta(days=1)
    assert (bundle.forecast >= 0).all()
    assert (bundle.upper >= bundle.lower).all()
    assert bundle.model_name == "ARIMA(1, 0, 0)"


def test_model_comparison_includes_naive_baseline():
    comparison = compare_models(
        make_series(),
        candidate_orders=[(1, 0, 0), (0, 1, 1)],
        test_size=8,
    )
    assert "Naive" in comparison["Model"].tolist()
    assert comparison.iloc[0]["Status"] == "OK"
