"""Tests for fetch.py."""
import sys
from pathlib import Path
import pytest

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


class TestFetchUrls:
    @pytest.mark.asyncio
    async def test_fetch_urls_returns_results(self, monkeypatch):
        from scryer_mcp.fetch import fetch_urls

        async def mock_fetch(url, client, mode):
            return {"url": url, "content": "test content", "status": 200,
                    "error": None, "title": "Test Title"}
        monkeypatch.setattr("scryer_mcp.fetch.fetch_url", mock_fetch)

        results = await fetch_urls(["https://example.com/a", "https://example.com/b"])
        assert len(results) == 2
        assert all(r.get("error") is None for r in results)
        assert all(r.get("content") == "test content" for r in results)
