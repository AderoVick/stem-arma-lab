import re
from datetime import datetime
import pandas as pd
from nltk.stem import PorterStemmer

TOKEN = re.compile(r"[A-Za-z]+")
ps = PorterStemmer()

def tokenize_stem(text: str):
    tokens = TOKEN.findall(text.lower())
    return [ps.stem(t) for t in tokens]

def build_stem_timeseries(csv_path: str, target_stem: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    counts = {}
    for _, row in df.iterrows():
        stems = tokenize_stem(str(row["text"]))
        c = stems.count(target_stem)
        d = row["date"].date()
        counts[d] = counts.get(d, 0) + c
    s = pd.Series(counts).sort_index()
    s.index = pd.to_datetime(s.index)
    s = s.asfreq("D", fill_value=0)
    return s
