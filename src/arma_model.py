"""Backward-compatible ARMA wrappers."""

from __future__ import annotations

import pandas as pd

from src.forecasting import fit_arima


def fit_arma(ts: pd.Series, p: int = 1, q: int = 1):
    """Fit an ARMA(p, q) model using ARIMA with d=0."""
    return fit_arima(ts, order=(p, 0, q))


def forecast_steps(res, steps: int = 7):
    """Forecast future values from a fitted statsmodels result."""
    forecast = res.get_forecast(steps=steps)
    return forecast.predicted_mean.clip(lower=0), forecast.conf_int().clip(lower=0)
