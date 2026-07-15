import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.arma_model import fit_arma, forecast_steps
from src.preprocess import build_stem_timeseries


def run(csv_path, stem, p, q, steps):
    """Run the full text-stem time-series forecasting pipeline."""
    ts = build_stem_timeseries(csv_path, stem)

    res = fit_arma(ts, p=p, q=q)
    pred, _ = forecast_steps(res, steps=steps)

    pred.index = pd.date_range(
        ts.index[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq="D",
    )

    ax = ts.plot(label="count")
    pred.plot(ax=ax, label="forecast")

    ax.legend()
    ax.set_title(f"Stem: {stem} ARMA({p}, {q})")

    plt.tight_layout()
    plt.savefig("forecast.png")

    print("Saved plot to forecast.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="data/sample_text.csv")
    parser.add_argument("--stem", default="traffic")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--steps", type=int, default=7)

    args = parser.parse_args()

    run(args.csv, args.stem, args.p, args.q, args.steps)
