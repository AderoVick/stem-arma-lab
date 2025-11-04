import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def fit_arma(ts: pd.Series, p: int = 1, q: int = 1):
    model = ARIMA(ts, order=(p, 0, q))
    return model.fit()

def forecast_steps(res, steps: int = 7):
    f = res.get_forecast(steps=steps)
    return f.predicted_mean, f.conf_int()
