"""Shared test fixtures."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_ddg(monkeypatch):
    """Mock duckduckgo_search to avoid real HTTP calls.

    Must patch in BOTH search.py (for direct importers) AND tiers.py
    (which imports ddg_search into its own namespace via 'from .search import').
    """
    async def mock_search(*args, **kwargs):
        return [
            {"title": f"Result {i}", "url": f"https://example.com/{i}",
             "snippet": f"Snippet for result {i} about test query."}
            for i in range(1, min(kwargs.get("num_results", 10) + 1, 6))
        ]

    from scryer_mcp import search, tiers
    monkeypatch.setattr(search, "ddg_search", mock_search)
    monkeypatch.setattr(tiers, "ddg_search", mock_search)
    return mock_search


@pytest.fixture
def mock_llm_available(monkeypatch):
    """Make LLMClient appear available."""
    async def mock_available(self):
        return True

    from scryer_mcp import llm_client as lc
    monkeypatch.setattr(lc.LLMClient, "is_available", mock_available)
    return mock_available


@pytest.fixture
def mock_llm_unavailable(monkeypatch):
    """Make LLMClient appear unavailable."""
    async def mock_unavailable(self):
        return False

    from scryer_mcp import llm_client as lc
    monkeypatch.setattr(lc.LLMClient, "is_available", mock_unavailable)
    return mock_unavailable
