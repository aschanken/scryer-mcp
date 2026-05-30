"""Tests for llm_client.py."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scryer_mcp.llm_client import LLMClient, _extract_json, _format_results


class TestExtractJSON:
    def test_simple_object(self):
        text = 'prefix {"key": "value"} suffix'
        assert _extract_json(text) == '{"key": "value"}'

    def test_nested_object(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        assert _extract_json(text) == text

    def test_no_braces(self):
        text = "no json here"
        assert _extract_json(text) == text

    def test_unclosed_brace(self):
        text = '{"key": "value"'
        assert _extract_json(text) == text


class TestFormatResults:
    def test_basic(self):
        results = [
            {"title": "T", "url": "https://x.com", "snippet": "snippet text"},
            {"title": "T2", "url": "https://y.com", "highlights": "hl text"},
        ]
        formatted = _format_results(results)
        assert "T" in formatted
        assert "snippet text" in formatted
        assert "hl text" in formatted


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_unavailable(self, mock_llm_unavailable):
        client = LLMClient()
        result = await client.synthesize("query", [{"title": "T", "url": "x", "snippet": "s"}])
        assert "error" in result
        assert "unavailable" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_classify_passthrough_on_unavailable(self, mock_llm_unavailable):
        client = LLMClient()
        results = [{"url": "https://x.com"}, {"url": "https://y.com"}]
        urls = await client.classify_category(results, "news")
        assert urls == ["https://x.com", "https://y.com"]  # all pass through
