import pandas as pd

from src.text_analysis import (
    build_topic_timeseries,
    normalize_token,
    suggest_topics,
    tokenize_normalized,
)


def sample_data():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-03"]),
            "text": [
                "Traffic delays delayed commuters.",
                "Traffic flows improved.",
                "Climate plans and traffic safety.",
            ],
        }
    )


def test_user_topic_is_normalized_consistently():
    assert normalize_token("delays", mode="stem") == "delay"
    tokens = tokenize_normalized("Delay delayed delays", mode="stem")
    assert tokens.count("delay") == 3


def test_build_topic_timeseries_fills_missing_dates():
    frame = build_topic_timeseries(sample_data(), ["traffic", "delays"], frequency="D")

    assert len(frame) == 3
    assert frame.loc[pd.Timestamp("2026-01-02")].sum() == 0
    assert frame["traffic"].sum() == 3
    assert frame["delays"].sum() == 2


def test_suggest_topics_excludes_common_stopwords():
    suggestions = dict(suggest_topics(sample_data(), top_n=10))
    assert "traffic" in suggestions
    assert "and" not in suggestions
