"""Forecasting, backtesting, model comparison, and diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ForecastBundle:
    """Forecast output and model metadata."""

    model_name: str
    order: tuple[int, int, int] | None
    forecast: pd.Series
    lower: pd.Series
    upper: pd.Series
    fitted_model: Any | None
    metrics: dict[str, float]
    comparison: pd.DataFrame


def validate_series(series: pd.Series, minimum_points: int = 12) -> pd.Series:
    """Return a clean numeric series or raise a user-friendly error."""
    if series is None or len(series) == 0:
        raise ValueError("The selected topic produced an empty time series.")
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(clean) < minimum_points:
        raise ValueError(
            f"At least {minimum_points} time periods are required; only {len(clean)} were found."
        )
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("The time series contains non-finite values.")
    if clean.sum() <= 0:
        raise ValueError("The selected topic has no occurrences in the chosen period.")
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise ValueError("The time series must use a DatetimeIndex.")
    return clean


def calculate_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and symmetric MAPE."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted values must have the same shape.")
    error = actual_values - predicted_values
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = np.abs(actual_values) + np.abs(predicted_values)
    smape_terms = np.divide(
        2.0 * np.abs(error),
        denominator,
        out=np.zeros_like(error, dtype=float),
        where=denominator != 0,
    )
    return {"MAE": mae, "RMSE": rmse, "sMAPE": float(np.mean(smape_terms) * 100)}


def _future_index(series: pd.Series, steps: int) -> pd.DatetimeIndex:
    frequency = series.index.freq or pd.infer_freq(series.index)
    if frequency is None:
        frequency = "D"
    offset = pd.tseries.frequencies.to_offset(frequency)
    return pd.date_range(series.index[-1] + offset, periods=steps, freq=frequency)


def fit_arima(series: pd.Series, order: tuple[int, int, int]):
    """Fit an ARIMA-family model with guarded warnings and validation."""
    clean = validate_series(series)
    p, d, q = order
    if min(p, d, q) < 0:
        raise ValueError("ARIMA order values cannot be negative.")
    if p + q > max(1, len(clean) // 4):
        raise ValueError("The selected p and q values are too large for this dataset.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(
            clean,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(method_kwargs={"maxiter": 60})


def _forecast_arima(
    series: pd.Series,
    order: tuple[int, int, int],
    steps: int,
    confidence: float,
) -> tuple[pd.Series, pd.Series, pd.Series, Any]:
    fitted = fit_arima(series, order)
    result = fitted.get_forecast(steps=steps)
    alpha = 1.0 - confidence
    mean = pd.Series(result.predicted_mean.to_numpy(), index=_future_index(series, steps))
    interval = result.conf_int(alpha=alpha)
    lower = pd.Series(interval.iloc[:, 0].to_numpy(), index=mean.index)
    upper = pd.Series(interval.iloc[:, 1].to_numpy(), index=mean.index)
    return mean.clip(lower=0), lower.clip(lower=0), upper.clip(lower=0), fitted


def _forecast_naive(series: pd.Series, steps: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    clean = validate_series(series)
    future = _future_index(clean, steps)
    last_value = float(clean.iloc[-1])
    differences = clean.diff().dropna()
    spread = float(differences.std()) if len(differences) > 1 else 0.0
    horizon_scale = np.sqrt(np.arange(1, steps + 1))
    mean = pd.Series(np.repeat(last_value, steps), index=future)
    lower = pd.Series(np.maximum(0, last_value - 1.96 * spread * horizon_scale), index=future)
    upper = pd.Series(last_value + 1.96 * spread * horizon_scale, index=future)
    return mean, lower, upper


def _backtest(
    series: pd.Series,
    order: tuple[int, int, int] | None,
    test_size: int,
) -> dict[str, float]:
    train = series.iloc[:-test_size]
    actual = series.iloc[-test_size:]
    if order is None:
        predicted = np.repeat(float(train.iloc[-1]), test_size)
    else:
        fitted = fit_arima(train, order)
        predicted = fitted.forecast(steps=test_size).to_numpy()
    predicted = np.maximum(0, np.asarray(predicted, dtype=float))
    return calculate_metrics(actual.to_numpy(), predicted)


def compare_models(
    series: pd.Series,
    candidate_orders: list[tuple[int, int, int]] | None = None,
    test_size: int | None = None,
) -> pd.DataFrame:
    """Backtest a compact model set and rank successful candidates by RMSE."""
    clean = validate_series(series, minimum_points=20)
    # Keep model selection responsive on long daily datasets while preserving
    # the most recent behaviour that is most relevant to short-term forecasts.
    evaluation_series = clean.tail(365)
    if candidate_orders is None:
        candidate_orders = [
            (1, 0, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 1, 1),
        ]
    if test_size is None:
        test_size = min(14, max(4, len(evaluation_series) // 5))
    if len(evaluation_series) - test_size < 12:
        raise ValueError("Not enough training observations remain for backtesting.")

    rows: list[dict[str, object]] = []
    candidates: list[tuple[str, tuple[int, int, int] | None]] = [("Naive", None)]
    candidates.extend((f"ARIMA{order}", order) for order in candidate_orders)

    for name, order in candidates:
        try:
            metrics = _backtest(evaluation_series, order, test_size)
            rows.append(
                {
                    "Model": name,
                    "Order": "—" if order is None else str(order),
                    **metrics,
                    "Status": "OK",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Model": name,
                    "Order": "—" if order is None else str(order),
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "sMAPE": np.nan,
                    "Status": f"Skipped: {exc}",
                }
            )

    comparison = pd.DataFrame(rows)
    successful = comparison[comparison["Status"].eq("OK")]
    if successful.empty:
        raise ValueError("None of the candidate forecasting models could be fitted.")
    ranking = successful.sort_values(["RMSE", "MAE", "sMAPE"]).index.tolist()
    skipped = comparison[~comparison["Status"].eq("OK")].index.tolist()
    return comparison.loc[ranking + skipped].reset_index(drop=True)


def create_forecast(
    series: pd.Series,
    steps: int = 7,
    order: tuple[int, int, int] | None = None,
    automatic: bool = True,
    confidence: float = 0.95,
) -> ForecastBundle:
    """Create a manual or automatically selected forecast with diagnostics."""
    clean = validate_series(series)
    if steps < 1 or steps > 365:
        raise ValueError("Forecast steps must be between 1 and 365.")
    if not 0.5 < confidence < 1.0:
        raise ValueError("Confidence must be between 0.5 and 1.0.")

    comparison = pd.DataFrame()
    selected_order = order
    selected_name = ""
    metrics: dict[str, float] = {}

    if automatic:
        comparison = compare_models(clean)
        winner = comparison.iloc[0]
        selected_name = str(winner["Model"])
        metrics = {key: float(winner[key]) for key in ("MAE", "RMSE", "sMAPE")}
        if selected_name == "Naive":
            selected_order = None
        else:
            order_text = str(winner["Order"]).strip("()")
            selected_order = tuple(int(value.strip()) for value in order_text.split(","))
    else:
        if order is None:
            raise ValueError("Provide an ARIMA order for manual forecasting.")
        selected_name = f"ARIMA{order}"
        test_size = min(14, max(4, len(clean) // 5))
        metrics = _backtest(clean, order, test_size)
        comparison = pd.DataFrame(
            [{"Model": selected_name, "Order": str(order), **metrics, "Status": "OK"}]
        )

    if selected_order is None:
        forecast, lower, upper = _forecast_naive(clean, steps)
        fitted = None
    else:
        forecast, lower, upper, fitted = _forecast_arima(
            clean, selected_order, steps, confidence
        )

    return ForecastBundle(
        model_name=selected_name,
        order=selected_order,
        forecast=forecast,
        lower=lower,
        upper=upper,
        fitted_model=fitted,
        metrics=metrics,
        comparison=comparison,
    )
