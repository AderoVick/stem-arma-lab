"""Generate the deterministic demonstration dataset used by the dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "sample_text.csv"


def generate_sample_data(days: int = 730, seed: int = 42) -> pd.DataFrame:
    """Create realistic dated text with topic trends, cycles, and incident spikes."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    rows: list[dict[str, str]] = []

    traffic_templates = [
        "Traffic congestion increased near the city centre.",
        "Road repairs caused traffic delays during the morning commute.",
        "New traffic signals improved vehicle flow downtown.",
        "Residents reported delayed buses and heavy traffic.",
    ]
    climate_templates = [
        "Climate adaptation projects expanded across the county.",
        "Heavy rainfall renewed discussion about climate resilience.",
        "The climate programme supported tree planting and clean energy.",
    ]
    outage_templates = [
        "A brief network outage affected online services.",
        "Engineers restored service after the reported outage.",
        "Customers reported slow connections but no major outage.",
    ]
    sales_templates = [
        "Retail sales improved after the weekend promotion.",
        "Monthly sales were supported by stronger customer demand.",
        "The sales team introduced a new product campaign.",
    ]
    general_templates = [
        "Community parks received positive feedback from residents.",
        "Public transport upgrades improved the commuter experience.",
        "The city opened a new health and education support centre.",
    ]

    for day_number in range(days):
        date = start + pd.Timedelta(days=day_number)
        texts: list[str] = []

        traffic_rate = 0.8 + 0.0015 * day_number + 0.5 * np.sin(day_number / 14)
        climate_rate = 0.3 + 0.0012 * day_number + 0.28 * np.sin(day_number / 28)
        outage_spike = any(
            start_day <= day_number <= start_day + 9
            for start_day in (80, 255, 430, 610)
        )
        outage_rate = 0.22 + (1.25 if outage_spike else 0.0)
        sales_rate = 0.4 + 0.48 * ((day_number % 30) > 22)

        if rng.random() < min(0.95, max(0.08, traffic_rate / 2.1)):
            texts.append(str(rng.choice(traffic_templates)))
        if rng.random() < min(0.9, max(0.05, climate_rate / 1.8)):
            texts.append(str(rng.choice(climate_templates)))
        if rng.random() < min(0.9, outage_rate / 1.6):
            texts.append(str(rng.choice(outage_templates)))
        if rng.random() < min(0.9, sales_rate / 1.4):
            texts.append(str(rng.choice(sales_templates)))
        if not texts or rng.random() < 0.35:
            texts.append(str(rng.choice(general_templates)))

        rows.append({"date": date.date().isoformat(), "text": " ".join(texts)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    frame = generate_sample_data()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(frame)} records to {OUTPUT}")
