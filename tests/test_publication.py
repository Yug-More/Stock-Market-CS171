import pytest

from marketpulse.publication import inline, latex, render, table


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


def test_display_math_is_preserved_as_typeset_latex():
    rendered = latex("# Title\n\nYug More\n\n## Methods\n\n$$\nx_t = y_t\n$$")
    assert r"\[" in rendered
    assert "x_t = y_t" in rendered
    assert r"\]" in rendered
    assert "$$" not in rendered


def test_reader_facing_manuscript_hides_tags_but_retains_manifest_bindings():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    markdown, tex, manifest = render(root)
    assert "<!-- evidence:" not in markdown
    assert "<!-- evidence:" not in tex
    assert "C01" not in markdown and "E:history" not in markdown
    assert len(manifest["claim_ids"]) == 37
    assert "history" in manifest["bound_evidence_groups"]
    assert len(manifest["citation_keys"]) == 12
