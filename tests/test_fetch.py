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

    def test_over_limit_hard_fallback(self):
        """No sentence boundary found — fall back to hard word cut (no ellipsis)."""
        from scryer_mcp.fetch import _truncate_words
        result = _truncate_words("one two three four five six", 3)
        assert result == "one two three"
        assert not result.endswith("…")

    def test_over_limit_preserves_sentence_boundary(self):
        """Must break at the nearest sentence boundary when one exists near the cutoff."""
        from scryer_mcp.fetch import _truncate_words
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        # max_words=5: "First sentence. Second sentence. Third" — walks back to
        # the nearest sentence boundary: "First sentence. Second sentence."
        result = _truncate_words(text, 5)
        assert result == "First sentence. Second sentence."

    def test_sentence_boundary_exclamation_and_question(self):
        """Must also recognise ! and ? as sentence boundaries."""
        from scryer_mcp.fetch import _truncate_words
        # Words: ["A.", "B!", "C", "D."] — at max_words=3 the cut creates
        # "A. B! C"; walks back to the "!" boundary.
        result = _truncate_words("A. B! C D.", 3)
        assert result == "A. B!"

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
                    "error": None, "title": "Test Title",
                    "content_truncated": False, "content_length": 12}
        monkeypatch.setattr("scryer_mcp.fetch.fetch_url", mock_fetch)

        results = await fetch_urls(["https://example.com/a", "https://example.com/b"])
        assert len(results) == 2
        assert all(r.get("error") is None for r in results)
        assert all(r.get("content") == "test content" for r in results)
