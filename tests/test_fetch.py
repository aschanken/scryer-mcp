"""Tests for fetch.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestTruncateWords:
    def test_under_limit(self):
        from scryer_mcp.fetch import _truncate_words
        result = _truncate_words("one two three", 10)
        assert result == "one two three"

    def test_over_limit(self):
        from scryer_mcp.fetch import _truncate_words
        result = _truncate_words("one two three four five six", 3)
        assert result == "one two three…"

    def test_empty(self):
        from scryer_mcp.fetch import _truncate_words
        result = _truncate_words("", 10)
        assert result == ""


class TestExtractTitle:
    def test_extracts_title(self):
        from scryer_mcp.fetch import _extract_title
        html = "<html><head><title>My Page Title</title></head><body></body></html>"
        assert _extract_title(html) == "My Page Title"

    def test_no_title(self):
        from scryer_mcp.fetch import _extract_title
        html = "<html><body>No title here</body></html>"
        assert _extract_title(html) is None
