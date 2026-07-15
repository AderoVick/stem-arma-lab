import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def fit_arma(ts: pd.Series, p: int = 1, q: int = 1):
    """Fit an ARMA(p, q) model using ARIMA with d=0."""
    model = ARIMA(ts, order=(p, 0, q))
    return model.fit()


def forecast_steps(res, steps: int = 7):
    """Forecast future values from the fitted ARMA model."""
    forecast = res.get_forecast(steps=steps)
    return forecast.predicted_mean, forecast.conf_int()
