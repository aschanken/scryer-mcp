"""Tests for search.py."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAugmentQuery:
    def test_no_category(self):
        from scryer_mcp.search import _augment_query
        assert _augment_query("test", None) == "test"
        assert _augment_query("test", "") == "test"

    def test_company_category(self):
        from scryer_mcp.search import _augment_query
        result = _augment_query("stripe", "company")
        assert "company" in result
        assert "stripe" in result

    def test_news_category(self):
        from scryer_mcp.search import _augment_query
        result = _augment_query("election", "news")
        assert "news" in result

    def test_unknown_category(self):
        from scryer_mcp.search import _augment_query
        result = _augment_query("test", "unknown_cat")
        # Unknown category adds no augmentation
        assert result == "test"


class TestDDGSearch:
    @pytest.mark.asyncio
    async def test_mocked_search(self, mock_ddg):
        from scryer_mcp.search import ddg_search
        results = await ddg_search("test query", num_results=3)
        assert len(results) == 3
        assert results[0]["title"] == "Result 1"
        assert "url" in results[0]
        assert "snippet" in results[0]

    @pytest.mark.asyncio
    async def test_category_passthrough(self, mock_ddg):
        from scryer_mcp.search import ddg_search
        results = await ddg_search("test", num_results=2, category="news")
        assert len(results) == 2
