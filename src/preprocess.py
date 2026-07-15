import re

import pandas as pd
from nltk.stem import PorterStemmer


TOKEN = re.compile(r"[A-Za-z]+")
ps = PorterStemmer()


def tokenize_stem(text: str):
    """Tokenize text and return Porter-stemmed words."""
    tokens = TOKEN.findall(text.lower())
    return [ps.stem(token) for token in tokens]


def build_stem_timeseries(csv_path: str, target_stem: str) -> pd.Series:
    """Build a daily time series showing the count of a selected stem."""
    df = pd.read_csv(csv_path)

    if "date" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'date' and 'text' columns.")

    df["date"] = pd.to_datetime(df["date"])

    counts = {}

    for _, row in df.iterrows():
        stems = tokenize_stem(str(row["text"]))
        count = stems.count(target_stem)
        date = row["date"].date()
        counts[date] = counts.get(date, 0) + count

    series = pd.Series(counts).sort_index()
    series.index = pd.to_datetime(series.index)
    series = series.asfreq("D", fill_value=0)

    return series
