# Version 2 Upgrade Summary

## Product improvements

- Rebranded the application as **TextPulse Forecast Lab**.
- Replaced the single-chart prototype with a four-section analytics dashboard.
- Added responsive metric cards, interactive Plotly charts, plain-language interpretation, and CSV downloads.
- Added a Streamlit Community Cloud entry point and theme configuration.

## Data and NLP improvements

- Uploads are read directly from memory and never written to a shared `_uploaded.csv` file.
- Added case-insensitive schema validation, invalid-date handling, blank-row handling, duplicate removal, and a data-quality report.
- Fixed inconsistent target normalization, so entries such as `delays`, `delayed`, and `delay` work together in stemming mode.
- Added multiple-topic comparison, exact matching, user-friendly topic suggestions, and daily/weekly/monthly aggregation.
- Expanded the demonstration dataset to 730 deterministic daily records and included a regeneration script.

## Forecasting improvements

- Added ARMA/ARIMA support through a common forecasting module.
- Added automatic backtested model selection and a naive baseline.
- Added MAE, RMSE, and sMAPE evaluation metrics.
- Added forecast confidence intervals and non-negative count handling.
- Added residual charts, AIC, BIC, and residual summary statistics.
- Added safeguards for empty, short, invalid, or all-zero time series and oversized model orders.

## Engineering improvements

- Split the project into data loading, text analysis, forecasting, UI, and CLI modules.
- Preserved backward compatibility for the original `preprocess.py` and `arma_model.py` functions.
- Added ten automated tests, CI checks for Python 3.10–3.12, critical linting, Docker support, and an MIT license.
- Rewrote the README with architecture, setup, testing, CLI, Streamlit deployment, and Docker instructions.
