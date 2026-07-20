import pandas as pd

from src.preprocess import build_stem_timeseries, tokenize_stem


def test_tokenize_stem():
    result = tokenize_stem("Traffic delays and traffic flow")
    assert "traffic" in result
    assert "delay" in result
    assert result.count("traffic") == 2


def test_build_stem_timeseries_normalizes_plural_input():
    data = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "text": ["Several delays occurred", "The delayed bus arrived"],
        }
    )
    series = build_stem_timeseries(data, "delays")
    assert series.sum() == 2
