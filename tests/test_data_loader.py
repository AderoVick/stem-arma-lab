import pandas as pd
import pytest

from src.data_loader import load_text_data


def test_load_text_data_cleans_invalid_and_duplicate_rows():
    source = pd.DataFrame(
        {
            "Date": ["2026-01-01", "bad-date", "2026-01-01", "2026-01-02"],
            "Text": ["Traffic improved", "Invalid", "Traffic improved", "   "],
        }
    )
    data, report = load_text_data(source)

    assert len(data) == 1
    assert report.invalid_date_rows == 1
    assert report.blank_text_rows == 1
    assert report.duplicate_rows == 1
    assert data.iloc[0]["text"] == "Traffic improved"


def test_load_text_data_requires_date_and_text_columns():
    with pytest.raises(ValueError, match="missing required"):
        load_text_data(pd.DataFrame({"date": ["2026-01-01"], "message": ["hello"]}))
