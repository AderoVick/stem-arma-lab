"""Command-line interface for TextPulse Forecast Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import load_text_data
from src.forecasting import create_forecast
from src.text_analysis import build_topic_timeseries


def run(
    csv_path: str,
    topic: str,
    steps: int,
    frequency: str,
    mode: str,
    automatic: bool,
    order: tuple[int, int, int],
    output: str,
) -> None:
    """Run the full text-topic forecasting pipeline and save a CSV result."""
    data, report = load_text_data(csv_path)
    topic_frame = build_topic_timeseries(data, [topic], frequency=frequency, mode=mode)
    series = topic_frame.iloc[:, 0]
    bundle = create_forecast(
        series,
        steps=steps,
        order=order,
        automatic=automatic,
    )

    historical = series.rename("historical").to_frame()
    forecast = pd.DataFrame(
        {
            "forecast": bundle.forecast,
            "lower_95": bundle.lower,
            "upper_95": bundle.upper,
        }
    )
    result = historical.join(forecast, how="outer")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index_label="date")

    print(f"Rows analyzed: {report.output_rows}")
    print(f"Topic: {topic}")
    print(f"Selected model: {bundle.model_name}")
    print(f"RMSE: {bundle.metrics['RMSE']:.3f}")
    print(f"Saved results to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast topic frequency in dated text.")
    parser.add_argument("--csv", default="data/sample_text.csv")
    parser.add_argument("--topic", "--stem", dest="topic", default="traffic")
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--frequency", choices=["D", "W", "M", "ME"], default="D")
    parser.add_argument("--mode", choices=["stem", "exact"], default="stem")
    parser.add_argument("--manual", action="store_true", help="Use the supplied p,d,q order.")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--d", type=int, default=0)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--output", default="outputs/forecast.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        csv_path=args.csv,
        topic=args.topic,
        steps=args.steps,
        frequency=args.frequency,
        mode=args.mode,
        automatic=not args.manual,
        order=(args.p, args.d, args.q),
        output=args.output,
    )
