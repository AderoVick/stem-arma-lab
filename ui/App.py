import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.arma_model import fit_arma, forecast_steps
from src.preprocess import build_stem_timeseries


st.set_page_config(page_title="Text Stemming + ARMA Lab", layout="wide")

st.title("Text Stemming + ARMA Forecasting Lab")

st.write(
    "Upload dated text data, select a target word stem, and forecast "
    "future stem frequency using an ARMA model."
)

uploaded = st.file_uploader("Upload CSV with columns: date,text", type=["csv"])

stem = st.text_input("Target stem", "traffic")
p = st.number_input("AR order p", min_value=1, max_value=5, value=1)
q = st.number_input("MA order q", min_value=0, max_value=5, value=1)
steps = st.number_input("Forecast steps in days", min_value=1, max_value=30, value=7)

if st.button("Run Forecast"):
    if uploaded is None:
        path = "data/sample_text.csv"
    else:
        path = "data/_uploaded.csv"
        with open(path, "wb") as file:
            file.write(uploaded.read())

    ts = build_stem_timeseries(path, stem)

    res = fit_arma(ts, p=int(p), q=int(q))
    pred, _ = forecast_steps(res, steps=int(steps))

    pred.index = pd.date_range(
        ts.index[-1] + pd.Timedelta(days=1),
        periods=int(steps),
        freq="D",
    )

    fig, ax = plt.subplots()
    ts.plot(ax=ax, label="Historical count")
    pred.plot(ax=ax, label="Forecast")

    ax.set_title(f"Forecast for stem: {stem}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stem frequency")
    ax.legend()

    st.pyplot(fig)
