import streamlit as st, pandas as pd, matplotlib.pyplot as plt
from src.preprocess import build_stem_timeseries
from src.arma_model import fit_arma, forecast_steps

st.set_page_config(page_title="STEM + ARMA Lab", layout="wide")
st.title("STEM (stemming) analysis + ARMA forecasting")

uploaded = st.file_uploader("Upload CSV (date,text)", type=["csv"])
stem = st.text_input("Target stem", "traffic")
p = st.number_input("p", 1, 5, 1)
q = st.number_input("q", 0, 5, 1)
steps = st.number_input("Forecast steps (days)", 1, 30, 7)

if st.button("Run"):
    if uploaded is None:
        path = "data/sample_text.csv"
    else:
        path = "data/_uploaded.csv"
        with open(path, "wb") as f:
            f.write(uploaded.read())
    ts = build_stem_timeseries(path, stem)
    res = fit_arma(ts, p=int(p), q=int(q))
    pred, _ = forecast_steps(res, steps=int(steps))
    pred.index = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=int(steps), freq="D")
    fig, ax = plt.subplots()
    ts.plot(ax=ax, label="count"); pred.plot(ax=ax, label="forecast")
    ax.legend(); st.pyplot(fig)
