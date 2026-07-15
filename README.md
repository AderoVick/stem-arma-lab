# Text Stemming + ARMA Forecasting Lab

This project is a lightweight text analytics and time-series forecasting tool. It helps users identify how often a selected word stem appears over time and then forecast future movement using an ARMA model.

The project is useful when text data has a date attached to it, such as news articles, customer feedback, support tickets, survey responses, reports, or social media posts. For example, a user can track how often words related to “traffic,” “price,” “delay,” “climate,” or “sales” appear each day, then generate a simple short-term forecast.

## Project Purpose

Many real-world datasets contain text, but text is often difficult to analyze over time. This project converts raw text into a daily numerical time series by counting selected word stems. Once the text has been converted into a time series, an ARMA model can be used to study patterns and make short-term forecasts.

For example, words such as “traffic,” “traffics,” and “trafficking” may share the same root stem. By applying stemming, the project groups related word forms together and produces a cleaner count of how often the topic appears.

## What the Project Does

The pipeline performs four main tasks:

1. Cleans and tokenizes text from a CSV file.
2. Applies Porter stemming to reduce words to their root form.
3. Aggregates the selected stem frequency by date.
4. Fits an ARMA(p, q) model and forecasts future daily counts.

The project can be used through both a command-line interface and a simple Streamlit dashboard.

## Example Use Cases

This project can be applied in several areas:

* **News trend analysis:** Track how often a topic appears in articles over time.
* **Customer feedback analysis:** Monitor complaints or repeated issues from customer text.
* **Social media monitoring:** Follow changes in public discussion around a keyword.
* **Academic research:** Study how important terms appear in dated text documents.
* **Business intelligence:** Forecast short-term attention around products, services, or market topics.

## Input Format

The project expects a CSV file with two columns:

```csv
date,text
2026-01-01,"Traffic delays increased after the road closure."
2026-01-02,"The city reported reduced traffic near the highway."
```

Required columns:

* `date`: the date of the text entry
* `text`: the raw text to analyze

## How the Workflow Works

The user selects a target stem, such as `traffic`. The system then searches each text record, counts how many times that stem appears, groups the counts by date, and creates a daily time series.

After the time series is created, the ARMA model estimates short-term movement based on past values and previous error patterns. The result is shown as a forecast chart in the dashboard.

## Dashboard Features

The Streamlit dashboard allows users to:

* upload a CSV file,
* select a target word stem,
* choose ARMA parameters `p` and `q`,
* select the number of forecast days,
* view historical stem counts,
* view forecasted future counts.

## Quickstart

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.\.venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run ui/App.py
```

Run a CLI example:

```bash
python -m src.main --csv data/sample_text.csv --stem traffic --p 1 --q 1 --steps 7
```

## Technologies Used

* Python
* pandas
* NLTK Porter Stemmer
* statsmodels
* Streamlit
* matplotlib

## Limitations

This project is designed as a learning and experimentation tool. Forecast quality depends on the amount and consistency of the data. Very rare words, short datasets, missing dates, or unstable text patterns may produce weak forecasts.

The ARMA model is best suited for relatively stable time series. If the data has strong trends, seasonality, or major structural changes, future versions could add ARIMA, SARIMA, or automated model selection.

## Future Improvements

Possible improvements include:

* adding model performance metrics such as MAE or RMSE,
* showing confidence intervals on the dashboard,
* allowing users to compare multiple stems,
* adding automatic stem suggestions,
* supporting lemmatization as an alternative to stemming,
* adding downloadable forecast results,
* improving visual design with summary cards and clearer charts.

## Summary

Text Stemming + ARMA Forecasting Lab connects natural language processing with time-series forecasting. It turns dated text into measurable trend data, making it easier to understand how topics rise, fall, and possibly move in the near future.
