"""Tokenization, normalization, topic extraction, and time-series creation."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

import pandas as pd
from nltk.stem import PorterStemmer

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STEMMER = PorterStemmer()

# A compact built-in list avoids requiring an NLTK corpus download at runtime.
STOP_WORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}

NORMALIZATION_MODES = {"stem", "exact"}
FREQUENCIES = {"D": "Daily", "W": "Weekly", "ME": "Monthly"}


def tokenize(text: object) -> list[str]:
    """Extract lowercase alphabetic tokens from a value."""
    return TOKEN_PATTERN.findall(str(text).lower())


def normalize_token(token: str, mode: str = "stem") -> str:
    """Normalize one token using Porter stemming or exact lowercase matching."""
    mode = mode.lower().strip()
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unsupported normalization mode: {mode}.")
    clean_tokens = tokenize(token)
    if not clean_tokens:
        raise ValueError("The topic must contain one alphabetic word.")
    if len(clean_tokens) > 1:
        raise ValueError("Each topic must be a single word; separate topics with commas.")
    clean = clean_tokens[0]
    return STEMMER.stem(clean) if mode == "stem" else clean


def tokenize_normalized(text: object, mode: str = "stem") -> list[str]:
    """Tokenize text and normalize every token."""
    tokens = tokenize(text)
    if mode == "stem":
        return [STEMMER.stem(token) for token in tokens]
    if mode == "exact":
        return tokens
    raise ValueError(f"Unsupported normalization mode: {mode}.")


def suggest_topics(
    data: pd.DataFrame,
    top_n: int = 15,
    mode: str = "stem",
    minimum_length: int = 3,
) -> list[tuple[str, int]]:
    """Return frequently occurring non-stopword topics and their counts."""
    if "text" not in data.columns:
        raise ValueError("Data must contain a 'text' column.")
    counter: Counter[str] = Counter()
    surface_forms: dict[str, Counter[str]] = {}
    for text in data["text"]:
        for raw_token in tokenize(text):
            if len(raw_token) < minimum_length or raw_token in STOP_WORDS:
                continue
            normalized = STEMMER.stem(raw_token) if mode == "stem" else raw_token
            counter[normalized] += 1
            surface_forms.setdefault(normalized, Counter())[raw_token] += 1

    suggestions: list[tuple[str, int]] = []
    for normalized, count in counter.most_common(max(1, int(top_n))):
        display_word = surface_forms[normalized].most_common(1)[0][0]
        suggestions.append((display_word, count))
    return suggestions


def _validate_topics(topics: Iterable[str], mode: str) -> list[tuple[str, str]]:
    validated: list[tuple[str, str]] = []
    seen: set[str] = set()
    for topic in topics:
        label = str(topic).strip()
        if not label:
            continue
        normalized = normalize_token(label, mode=mode)
        if normalized not in seen:
            validated.append((label, normalized))
            seen.add(normalized)
    if not validated:
        raise ValueError("Enter at least one valid topic.")
    return validated


def build_topic_timeseries(
    data: pd.DataFrame,
    topics: Sequence[str],
    frequency: str = "D",
    mode: str = "stem",
) -> pd.DataFrame:
    """Build an evenly spaced count time series for one or more topics."""
    if not {"date", "text"}.issubset(data.columns):
        raise ValueError("Data must contain 'date' and 'text' columns.")
    if frequency == "M":
        frequency = "ME"
    if frequency not in FREQUENCIES:
        raise ValueError("Frequency must be one of: D, W, ME.")

    topic_pairs = _validate_topics(topics, mode)
    normalized_rows = data["text"].map(lambda value: tokenize_normalized(value, mode))

    output = pd.DataFrame(index=pd.DatetimeIndex(data["date"]))
    for label, normalized_topic in topic_pairs:
        output[label] = normalized_rows.map(lambda row: row.count(normalized_topic)).to_numpy()

    output = output.groupby(level=0).sum().sort_index()
    output = output.resample(frequency).sum().asfreq(frequency, fill_value=0)
    output.index.name = "date"
    return output.astype(int)
