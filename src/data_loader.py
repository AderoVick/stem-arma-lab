"""Data loading and validation utilities for dated text datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

import pandas as pd

REQUIRED_COLUMNS = {"date", "text"}


@dataclass(frozen=True)
class DataQualityReport:
    """Summary of the cleaning operations applied to an input dataset."""

    input_rows: int
    output_rows: int
    invalid_date_rows: int
    blank_text_rows: int
    duplicate_rows: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""
        report = asdict(self)
        report["start_date"] = self.start_date.date().isoformat()
        report["end_date"] = self.end_date.date().isoformat()
        return report


def _read_source(source: str | Path | IO[bytes] | IO[str] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    try:
        return pd.read_csv(source)
    except Exception as exc:  # pandas raises several parser/encoding exceptions
        raise ValueError(f"Unable to read the CSV file: {exc}") from exc


def load_text_data(
    source: str | Path | IO[bytes] | IO[str] | pd.DataFrame,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load, normalize, and validate a dated text dataset.

    Column names are matched case-insensitively. Invalid dates and blank text rows
    are removed, exact duplicate rows are dropped, and the final data is sorted.
    """
    df = _read_source(source)
    if df.empty:
        raise ValueError("The dataset is empty.")

    df.columns = [str(column).strip().lower() for column in df.columns]
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required column(s): {missing_list}.")

    df = df.loc[:, ["date", "text"]].copy()
    input_rows = len(df)

    parsed_dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    invalid_date_mask = parsed_dates.isna()
    invalid_date_rows = int(invalid_date_mask.sum())
    df = df.loc[~invalid_date_mask].copy()
    parsed_dates = parsed_dates.loc[~invalid_date_mask]

    # Convert to timezone-naive normalized dates for predictable aggregation.
    df["date"] = parsed_dates.dt.tz_convert(None).dt.normalize()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    blank_text_mask = df["text"].eq("")
    blank_text_rows = int(blank_text_mask.sum())
    df = df.loc[~blank_text_mask].copy()

    duplicate_rows = int(df.duplicated(subset=["date", "text"]).sum())
    df = df.drop_duplicates(subset=["date", "text"]).sort_values("date")
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after cleaning dates and text.")

    report = DataQualityReport(
        input_rows=input_rows,
        output_rows=len(df),
        invalid_date_rows=invalid_date_rows,
        blank_text_rows=blank_text_rows,
        duplicate_rows=duplicate_rows,
        start_date=df["date"].min(),
        end_date=df["date"].max(),
    )
    return df, report
