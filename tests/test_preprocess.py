from src.preprocess import tokenize_stem


def test_tokenize_stem():
    result = tokenize_stem("Traffic delays and traffic flow")
    assert "traffic" in result
    assert "delay" in result
    assert result.count("traffic") == 2
