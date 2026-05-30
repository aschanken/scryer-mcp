"""Integration tests — requires real network access."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_instant_search_real():
    """Real DDG search — verifies the server actually works against the internet.

    Note: DDG may rate-limit silently (return 0 results without error).
    We validate response structure regardless of result count.
    """
    from scryer_mcp.schema import ScryerRequest, Tier, ScryerResponse
    from scryer_mcp.tiers import _instant

    req = ScryerRequest(query="capital of France", tier=Tier.INSTANT, num_results=3)
    resp = await _instant(req)

    # Response structure must be valid regardless of DDG rate-limiting
    assert isinstance(resp, ScryerResponse)
    assert resp.tier_used == "instant"
    assert resp.trace.searches_performed == 1

    # If we got results (no rate limit), validate them
    for r in resp.results:
        assert r.title, "Each result should have a title"
        assert r.url, "Each result should have a URL"
        assert r.url.startswith("http"), "URL should be valid"
        assert r.snippet, "Each result should have a snippet"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_checks():
    """Real health check — verifies DDG, HTTP, cache are reachable."""
    from scryer_mcp.server import scryer_health

    status = await scryer_health()

    assert "checks" in status
    # DDG may rate-limit: if it returns 0 results, check detail explains why
    assert "ddg_search" in status["checks"], f"Missing DDG check: {status}"
    assert status["checks"]["http_fetch"]["ok"], f"HTTP fetch should work: {status['checks']['http_fetch']}"
    assert status["checks"]["cache"]["ok"], f"Cache should be writable: {status['checks']['cache']}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_real_url():
    """Real URL fetch — verifies trafilatura extraction."""
    from scryer_mcp.fetch import fetch_urls

    results = await fetch_urls(["https://example.com"], "highlights", 8000)
    assert len(results) == 1
    r = results[0]
    assert r["error"] is None, f"Fetch should succeed: {r.get('error')}"
    assert r["content"], "Should extract some content"
    assert len(r["content"]) > 0
    assert r["status"] == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dead_link_buffer():
    """Fetching unreachable URLs should not crash — Karl's dead link buffer."""
    from scryer_mcp.fetch import fetch_urls

    results = await fetch_urls(
        ["https://nonexistent-domain-xyz123456.com", "https://example.com"],
        "highlights", 5000
    )
    assert len(results) == 2
    # First should fail
    assert results[0]["error"] is not None, "Dead link should report error"
    # Second should succeed
    assert results[1]["error"] is None or results[1]["content"], "Valid URL should succeed"
    # Partial results should be usable
    successful = [r for r in results if r["error"] is None]
    assert len(successful) >= 1
