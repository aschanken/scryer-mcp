"""Tests for tier dispatch."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scryer_mcp.schema import ScryerRequest, ScryerResponse, Tier


class TestInstantTier:
    @pytest.mark.asyncio
    async def test_returns_results(self, mock_ddg):
        from scryer_mcp.tiers import _instant
        req = ScryerRequest(query="test", tier=Tier.INSTANT, num_results=3)
        resp = await _instant(req)
        assert isinstance(resp, ScryerResponse)
        assert resp.tier_used == "instant"
        assert len(resp.results) == 3
        assert resp.results[0].title == "Result 1"
        assert resp.results[0].snippet
        assert resp.results[0].highlights is None
        assert resp.grounded_answer is None

    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch):
        from scryer_mcp import tiers
        async def empty_search(*a, **kw):
            return []
        monkeypatch.setattr(tiers, "ddg_search", empty_search)

        req = ScryerRequest(query="nonexistent", tier=Tier.INSTANT, num_results=3)
        resp = await tiers._instant(req)
        assert len(resp.results) == 0
        assert resp.error == "No results found"


class TestFastTier:
    @pytest.mark.asyncio
    async def test_returns_highlights(self, mock_ddg, monkeypatch):
        from scryer_mcp import tiers
        # Mock fetch to return content
        async def mock_fetch(urls, mode, timeout_ms):
            return [{"url": u, "title": f"Title for {u}",
                     "content": f"Content for {u} with enough words to fill highlights.",
                     "status": 200, "error": None} for u in urls]
        monkeypatch.setattr(tiers, "fetch_urls", mock_fetch)
        # Skip cache
        monkeypatch.setattr(tiers, "cache_ttl", lambda *a: False)

        req = ScryerRequest(query="test", tier=Tier.FAST, num_results=3, livecrawl=True)
        resp = await tiers._fast(req)
        assert resp.tier_used == "fast"
        assert resp.trace.urls_fetched > 0


class TestAutoTier:
    @pytest.mark.asyncio
    async def test_synthesis_error_degraded(self, mock_ddg, mock_llm_unavailable, monkeypatch):
        from scryer_mcp import tiers
        async def mock_fetch(urls, mode, timeout_ms):
            return [{"url": u, "title": f"T: {u}",
                     "content": f"Highlights content for {u}.", "status": 200,
                     "error": None} for u in urls]
        monkeypatch.setattr(tiers, "fetch_urls", mock_fetch)
        monkeypatch.setattr(tiers, "cache_ttl", lambda *a: False)

        req = ScryerRequest(query="test", tier=Tier.AUTO, num_results=3, livecrawl=True)
        resp = await tiers._auto(req)
        assert resp.tier_used == "auto"
        # Should degrade gracefully — results still returned
        assert len(resp.results) > 0
        # Synthesis failed but search worked
        assert resp.error is not None or resp.grounded_answer is None


class TestDeepTier:
    @pytest.mark.asyncio
    async def test_returns_grounded_answer(self, mock_ddg, mock_llm_available, monkeypatch):
        from scryer_mcp import tiers
        async def mock_fetch(urls, mode, timeout_ms):
            return [{"url": u, "title": f"T: {u}",
                     "content": f"Detailed content for {u}.", "status": 200,
                     "error": None} for u in urls]
        monkeypatch.setattr(tiers, "fetch_urls", mock_fetch)
        monkeypatch.setattr(tiers, "cache_ttl", lambda *a: False)

        # Mock LLM _chat to return valid JSON (instance attribute — no 'self' binding)
        async def mock_chat(system, user, temperature=0.3, max_tokens=2000):
            if "synthesize" in system.lower() or "research synthesizer" in system.lower():
                return '{"grounded_answer": "This is a test synthesis. [cite: https://example.com/1]", "citations": ["https://example.com/1"]}'
            return '["follow-up query 1"]'
        monkeypatch.setattr(tiers.llm_client, "_chat", mock_chat)
        # Mock LLM synthesize to return valid answer (must be async/awaitable)
        async def mock_synthesize(q, r):
            return {"grounded_answer": "Test answer with [cite: https://x.com]", "citations": ["https://x.com"]}
        monkeypatch.setattr(tiers.llm_client, "synthesize", mock_synthesize)

        req = ScryerRequest(query="test", tier=Tier.DEEP, num_results=3, livecrawl=True)
        resp = await tiers._deep(req)
        assert resp.tier_used == "deep"
        assert resp.grounded_answer is not None
        assert len(resp.citations) > 0


class TestExecuteTier:
    @pytest.mark.asyncio
    async def test_dispatches_instant(self, mock_ddg):
        from scryer_mcp.tiers import execute_tier
        req = ScryerRequest(query="test", tier=Tier.INSTANT, num_results=2)
        resp = await execute_tier(req)
        assert resp.tier_used == "instant"
        assert len(resp.results) == 2

    @pytest.mark.asyncio
    async def test_dispatches_fast(self, mock_ddg, monkeypatch):
        from scryer_mcp import tiers
        from scryer_mcp.tiers import execute_tier
        async def mock_fetch(urls, mode, timeout_ms):
            return [{"url": u, "title": f"T: {u}",
                     "content": f"Content for {u}.", "status": 200,
                     "error": None} for u in urls]
        monkeypatch.setattr(tiers, "fetch_urls", mock_fetch)
        monkeypatch.setattr(tiers, "cache_ttl", lambda *a: False)

        req = ScryerRequest(query="test", tier=Tier.FAST, num_results=2, livecrawl=True)
        resp = await execute_tier(req)
        assert resp.tier_used == "fast"
