import argparse, pandas as pd, matplotlib.pyplot as plt
from .preprocess import build_stem_timeseries
from .arma_model import fit_arma, forecast_steps

def run(csv_path, stem, p, q, steps):
    ts = build_stem_timeseries(csv_path, stem)
    res = fit_arma(ts, p=p, q=q)
    pred, _ = forecast_steps(res, steps=steps)
    ax = ts.plot(label="count")
    pred.index = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    pred.plot(ax=ax, label="forecast")
    ax.legend(); ax.set_title(f"Stem: {stem}  ARMA({p},{q})")
    plt.tight_layout(); plt.savefig("assets/forecast.png")
    print("Saved plot to assets/forecast.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_text.csv")
    ap.add_argument("--stem", default="traffic")
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--steps", type=int, default=7)
    a = ap.parse_args()
    run(a.csv, a.stem, a.p, a.q, a.steps)
