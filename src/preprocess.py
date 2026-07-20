"""Backward-compatible preprocessing helpers.

New code should prefer :mod:`src.data_loader` and :mod:`src.text_analysis`.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from src.data_loader import load_text_data
from src.text_analysis import build_topic_timeseries, tokenize_normalized


def tokenize_stem(text: str) -> list[str]:
    """Tokenize text and return Porter-stemmed words."""
    return tokenize_normalized(text, mode="stem")


def build_stem_timeseries(
    csv_path: str | Path | IO[bytes] | IO[str] | pd.DataFrame,
    target_stem: str,
) -> pd.Series:
    """Build a daily series for one topic while normalizing user input."""
    data, _ = load_text_data(csv_path)
    frame = build_topic_timeseries(data, [target_stem], frequency="D", mode="stem")
    return frame.iloc[:, 0]
