"""Scryer MCP Server — local search via DuckDuckGo.

Tools:
  - scryer_search             Web search with quality tiers
  - scryer_fetch_content      Content extraction from URLs
  - scryer_extract_structured Structured data extraction via JSON Schema
  - scryer_synthesize         LLM synthesis with grounded answers
  - scryer_health             Connectivity verification
"""
from __future__ import annotations

import traceback
from mcp.server.fastmcp import FastMCP
from .schema import (
    ScryerRequest, ScryerResponse, Tier, Category,
    validate_extraction_schema,
)
from .search import ddg_search
from .fetch import fetch_urls
from .tiers import execute_tier, llm_client
from . import __version__

mcp = FastMCP("scryer")


@mcp.tool()
async def scryer_search(
    query: str,
    num_results: int = 10,
    category: str | None = None,
    highlights: bool = True,
    full_text: bool = False,
    livecrawl: bool = True,
    tier: str = "auto",
    max_tokens: int | None = None,
) -> dict:
    """Search the web using DuckDuckGo-style web search.

    Returns structured results with optional highlights, full text, and
    LLM synthesis depending on tier.

    Args:
        query: The search query string (required).
        num_results: Number of results to return (1-50, default 10).
        category: Optional filter — company, people, news, research, paper, code.
        highlights: Return token-efficient content extracts (default true).
        full_text: Return full page content instead of highlights (default false).
        livecrawl: Fetch fresh content; false = snippets only (default true).
        tier: Quality tier — instant, fast, auto, deep, deep-reasoning (default auto).
        max_tokens: Soft cap on total output tokens.
    """
    try:
        cat = Category(category) if category else None
    except ValueError:
        valid = [c.value for c in Category]
        return {"error": f"Invalid category: '{category}'. Valid values: {valid}"}

    try:
        t = Tier(tier)
    except ValueError:
        valid = [t.value for t in Tier]
        return {"error": f"Invalid tier: '{tier}'. Valid values: {valid}"}

    request = ScryerRequest(
        query=query, tier=t, num_results=num_results,
        category=cat, highlights=highlights, full_text=full_text,
        livecrawl=livecrawl, max_tokens=max_tokens,
    )
    try:
        response = await execute_tier(request)
        return response.model_dump()
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@mcp.tool()
async def scryer_fetch_content(
    urls: list[str],
    mode: str = "highlights",
    timeout_ms: int = 8000,
) -> dict:
    """Fetch and extract clean content from URLs.

    Uses trafilatura for content extraction with BeautifulSoup fallback.
    Returns clean markdown or highlights — no raw HTML.

    Args:
        urls: URLs to fetch (max 15).
        mode: "highlights" (150-200 words) or "full_text".
        timeout_ms: Per-URL timeout in milliseconds (default 8000).
    """
    if mode not in ("highlights", "full_text"):
        return {"error": f"Invalid mode: '{mode}'. Use 'highlights' or 'full_text'."}

    try:
        results = await fetch_urls(urls[:15], mode, timeout_ms)
        return {
            "results": results,
            "fetched": len([r for r in results if not r.get("error")]),
            "failed": len([r for r in results if r.get("error")]),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def scryer_extract_structured(
    urls: list[str],
    schema: dict,
    mode: str = "highlights",
) -> dict:
    """Extract structured data from URLs according to a JSON Schema.

    Uses map-reduce pattern: extracts from each page independently via LLM,
    then merges and deduplicates. Requires LLM_ENDPOINT to be configured.

    Args:
        urls: URLs to extract data from (max 10).
        schema: JSON Schema defining the extraction shape.
               Example: {"type": "object", "properties": {"name": {"type": "string"}}}
        mode: "highlights" or "full_text" for content extraction.
    """
    err = validate_extraction_schema(schema)
    if err:
        return {"error": f"Invalid schema: {err}"}

    results = await fetch_urls(urls[:10], mode, 10000)

    extractions = []
    errors = 0
    for r in results:
        if r.get("error") or not r.get("content"):
            continue
        # Retry once on failure
        extracted = await llm_client.extract_structured(r["content"], schema)
        if "error" in extracted:
            extracted = await llm_client.extract_structured(r["content"], schema)
        if "error" in extracted:
            errors += 1
        else:
            extractions.append(extracted)

    from .extract import merge_extractions
    merged = merge_extractions(extractions, schema)

    return {
        "items": [merged] if merged else [],
        "schema_used": schema,
        "extraction_errors": errors,
        "pages_processed": len([r for r in results if not r.get("error")]),
    }


@mcp.tool()
async def scryer_synthesize(
    query: str,
    results: list[dict],
) -> dict:
    """Synthesize a grounded, cited answer from search results.

    Uses an LLM to produce 2-3 paragraphs with inline citations.
    Requires LLM_ENDPOINT to be configured.

    Args:
        query: The original search query.
        results: List of dicts with title, url, and highlights/snippet fields.
    """
    return await llm_client.synthesize(query, results)


@mcp.tool()
async def scryer_health() -> dict:
    """Check connectivity of all backend dependencies.

    Returns status for: DDG search, HTTP fetch, LLM endpoint, cache directory.
    Use this to verify the server is production-ready.
    """
    import os
    from .cache import DEFAULT_ROOT

    status = {"version": __version__, "checks": {}, "all_ok": True}

    # DDG availability
    try:
        test_results = await ddg_search("test", num_results=1)
        status["checks"]["ddg_search"] = {
            "ok": True, "detail": f"returned {len(test_results)} results"
        }
    except Exception as e:
        status["checks"]["ddg_search"] = {"ok": False, "detail": str(e)}
        status["all_ok"] = False

    # HTTP fetch
    try:
        r = await fetch_urls(["https://example.com"], "highlights", 8000)
        ok = r[0].get("error") is None and r[0].get("status") == 200
        status["checks"]["http_fetch"] = {"ok": ok, "detail": f"status {r[0].get('status', '?')}"}
    except Exception as e:
        status["checks"]["http_fetch"] = {"ok": False, "detail": str(e)}
        status["all_ok"] = False

    # LLM endpoint (optional — doesn't set all_ok=False)
    llm_ok = await llm_client.is_available()
    status["checks"]["llm_endpoint"] = {
        "ok": llm_ok,
        "detail": "reachable" if llm_ok else "unavailable (synthesis/extraction disabled)"
    }

    # Cache directory
    try:
        DEFAULT_ROOT.mkdir(parents=True, exist_ok=True)
        test_file = DEFAULT_ROOT / ".health_check"
        test_file.write_text("ok")
        test_file.unlink()
        status["checks"]["cache"] = {"ok": True, "detail": str(DEFAULT_ROOT)}
    except Exception as e:
        status["checks"]["cache"] = {"ok": False, "detail": str(e)}
        status["all_ok"] = False

    return status


def main():
    """Entry point for stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
