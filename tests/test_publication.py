import pytest

from marketpulse.publication import inline, latex, table


def test_manuscript_renderer_preserves_math_citations_and_special_characters():
    text = inline("S&P at 90%: $x_t$ [@holm1979] `path_a.py`")
    assert r"S\&P at 90\%" in text
    assert "$x_t$" in text
    assert r"\citep{holm1979}" in text
    assert r"\path{path_a.py}" in text


def test_manuscript_tables_keep_header_and_caption():
    data = table(1, "Measured results", ["Index", "RMSE"], [["S&P", ".01"]])
    rendered = latex("# Title\n\nAuthor\n\n" + data)
    assert r"\caption{Measured results}" in rendered
    assert r"S\&P & .01" in rendered
    assert "---" not in rendered.split(r"\begin{longtable}")[1]
    with pytest.raises(AssertionError):
        latex("# Title\n\n![broken]")
