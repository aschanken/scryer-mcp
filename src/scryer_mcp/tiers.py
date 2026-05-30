"""Tier dispatch: instant → fast → auto → deep → deep-reasoning."""
from __future__ import annotations

import asyncio
import math
import time
from .schema import (
    ScryerRequest, ScryerResponse, SearchResult, TierTrace, Tier
)
from .search import ddg_search
from .fetch import fetch_urls
from .cache import get as cache_get, put as cache_put, ttl as cache_ttl
from .llm_client import LLMClient, _extract_json

llm_client = LLMClient()


def _estimate_tokens(text: str | None) -> int:
    """Rough token estimate (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


async def execute_tier(request: ScryerRequest) -> ScryerResponse:
    """Dispatch to the appropriate tier handler."""
    t0 = time.monotonic()

    match request.tier:
        case Tier.INSTANT:
            resp = await _instant(request)
        case Tier.FAST:
            resp = await _fast(request)
        case Tier.AUTO:
            resp = await _auto(request)
        case Tier.DEEP:
            resp = await _deep(request)
        case Tier.DEEP_REASONING:
            resp = await _deep_reasoning(request)
        case _:
            resp = await _auto(request)

    resp.search_time_ms = int((time.monotonic() - t0) * 1000)
    resp.trace.tier = request.tier.value
    resp.trace.wall_time_ms = resp.search_time_ms
    return resp


def _to_search_result(raw: dict, fetched: dict | None = None) -> SearchResult:
    return SearchResult(
        title=raw.get("title", ""),
        url=raw.get("url", ""),
        snippet=raw.get("snippet", ""),
        highlights=fetched.get("content") if fetched else None,
    )


# === instant ===
async def _instant(req: ScryerRequest) -> ScryerResponse:
    results = await ddg_search(req.query, min(math.ceil(req.num_results * 1.2), 20),
                               req.category.value if req.category else None)
    trace = TierTrace(
        tier="instant", searches_performed=1,
        urls_fetched=0, urls_skipped_cache=0, urls_failed=0,
        wall_time_ms=0, tokens_consumed=len(results) * 50,
    )
    sr = [_to_search_result(r) for r in results[:req.num_results]]
    error = "No results found" if not sr else None
    return ScryerResponse(
        results=sr, search_time_ms=0, tier_used="instant",
        trace=trace, error=error,
    )


# === fast ===
async def _fast(req: ScryerRequest) -> ScryerResponse:
    fetch_count = min(math.ceil(req.num_results * 1.3), 20)
    results = await ddg_search(req.query, fetch_count,
                               req.category.value if req.category else None)
    trace = TierTrace(
        tier="fast", searches_performed=1, urls_fetched=0,
        urls_skipped_cache=0, urls_failed=0, wall_time_ms=0,
        tokens_consumed=0,
    )

    to_fetch = []
    cached = []
    import json as _json
    for r in results[:fetch_count]:
        if req.livecrawl and not cache_ttl("fetch", r["url"], 86400):
            to_fetch.append(r)
        else:
            entry = cache_get("fetch", r["url"])
            if entry:
                data = _json.loads(entry)
                cached.append(SearchResult(
                    title=data.get("title") or r.get("title", ""),
                    url=data.get("url", ""),
                    snippet=r.get("snippet", ""),
                    highlights=data.get("content"),
                ))
                trace.urls_skipped_cache += 1
            elif req.livecrawl:
                to_fetch.append(r)

    if to_fetch and (req.highlights or req.full_text) and req.livecrawl:
        mode = "full_text" if req.full_text else "highlights"
        fetched = await fetch_urls([r["url"] for r in to_fetch], mode, 8000)
        trace.urls_fetched = len(to_fetch)
        fetched_map = {}
        for f in fetched:
            if f["error"]:
                trace.urls_failed += 1
            else:
                fetched_map[f["url"]] = f
                if req.livecrawl:
                    cache_put("fetch", f["url"], _json.dumps(f))

        all_results = cached + [
            _to_search_result(r, fetched_map.get(r["url"]))
            for r in to_fetch
        ]
    else:
        all_results = cached + [_to_search_result(r) for r in to_fetch]

    valid = [r for r in all_results if r.highlights or r.snippet]
    if not valid:
        return await _instant(req)

    trace.tokens_consumed = sum(_estimate_tokens(r.highlights) + _estimate_tokens(r.snippet) for r in valid)
    return ScryerResponse(
        results=valid[:req.num_results], search_time_ms=0,
        tier_used="fast", trace=trace,
    )


# === auto ===
async def _auto(req: ScryerRequest) -> ScryerResponse:
    # Steps 1-6: identical to fast
    fast_resp = await _fast(req)
    if not fast_resp.results:
        return fast_resp

    # Step 7: Synthesis
    result_dicts = [
        {"title": r.title, "url": r.url, "snippet": r.snippet, "highlights": r.highlights}
        for r in fast_resp.results
    ]
    synthesis = await llm_client.synthesize(req.query, result_dicts)

    if "error" in synthesis:
        fast_resp.trace.tier = "auto"
        fast_resp.tier_used = "auto"
        fast_resp.error = synthesis.get("error")
        return fast_resp

    fast_resp.grounded_answer = synthesis.get("grounded_answer")
    fast_resp.citations = synthesis.get("citations", [])
    fast_resp.trace.tier = "auto"
    fast_resp.tier_used = "auto"
    fast_resp.trace.tokens_consumed += _estimate_tokens(fast_resp.grounded_answer) + 500
    return fast_resp


# === deep ===
async def _deep(req: ScryerRequest) -> ScryerResponse:
    # First pass: auto tier
    auto_resp = await _auto(req)

    # Second pass: gap-filling search
    gaps = []
    if auto_resp.grounded_answer:
        gaps = await _identify_gaps(req.query, auto_resp)
        if gaps:
            for gap_query in gaps[:3]:
                gap_results = await ddg_search(gap_query, 5)
                if gap_results:
                    fetched = await fetch_urls(
                        [r["url"] for r in gap_results[:3]], "highlights", 8000
                    )
                    for f in fetched:
                        if not f["error"] and f.get("content"):
                            # Deduplicate against existing results
                            if any(r.url == f["url"] for r in auto_resp.results):
                                continue
                            auto_resp.results.append(SearchResult(
                                title=f.get("title") or f["url"],
                                url=f["url"],
                                snippet="",
                                highlights=f["content"],
                            ))
                            auto_resp.trace.urls_fetched += 1

    # Re-synthesize with gap results
    result_dicts = [
        {"title": r.title, "url": r.url, "snippet": r.snippet, "highlights": r.highlights}
        for r in auto_resp.results
    ]
    synthesis = await llm_client.synthesize(req.query, result_dicts)
    if "error" not in synthesis:
        auto_resp.grounded_answer = synthesis.get("grounded_answer")
        auto_resp.citations = synthesis.get("citations", [])

    auto_resp.trace.tier = "deep"
    auto_resp.tier_used = "deep"
    auto_resp.trace.searches_performed += len(gaps) if gaps else 0
    auto_resp.trace.tokens_consumed += _estimate_tokens(auto_resp.grounded_answer) + 1000
    return auto_resp


async def _identify_gaps(query: str, response: ScryerResponse) -> list[str]:
    """Identify knowledge gaps for follow-up search queries."""
    if not response.grounded_answer:
        return []
    text = (
        f"Query: {query}\n\n"
        f"Current answer:\n{response.grounded_answer}\n\n"
        "Identify 1-3 specific follow-up search queries that would fill "
        "gaps or resolve uncertainties in this answer. Return ONLY a JSON "
        'array of query strings: ["query1", "query2"]'
    )
    result = await llm_client._chat(
        "You identify knowledge gaps in research.",
        text, temperature=0.3, max_tokens=300,
    )
    if not result:
        return []
    try:
        import json as _json
        parsed = _json.loads(_extract_json(result))
        if isinstance(parsed, list):
            return parsed[:3]
    except Exception:
        pass
    return []


# === deep-reasoning ===
async def _deep_reasoning(req: ScryerRequest) -> ScryerResponse:
    deep_resp = await _deep(req)
    if not deep_resp.grounded_answer:
        return deep_resp

    # Chain-of-thought analysis
    cot = await llm_client._chat(
        "You are a rigorous research analyst. Perform step-by-step reasoning "
        "on research findings. Assess consensus, minority views, recency, "
        "assumptions, and evidence quality.",
        f"Query: {req.query}\n\nFindings:\n{deep_resp.grounded_answer}",
        temperature=0.3, max_tokens=1000,
    )

    # Adversarial verification (3 independent lenses, concurrent)
    if cot:
        lenses = [
            ("accuracy", "Fact-check this answer for accuracy. Flag unsupported claims."),
            ("completeness", "Check for completeness. What important aspects are missing?"),
            ("bias", "Check for bias/balance. Does it over-rely on one source type?"),
        ]

        async def verify(lens_name: str, prompt: str) -> dict | None:
            text = await llm_client._chat(
                f"You are a {lens_name} reviewer.", prompt, temperature=0.2, max_tokens=500
            )
            return {"lens": lens_name, "verdict": text} if text else None

        verdicts = await asyncio.gather(*[
            verify(name, f"{prompt}\n\nQuery: {req.query}\n\nAnswer:\n{deep_resp.grounded_answer}")
            for name, prompt in lenses
        ])
        cot_tokens = _estimate_tokens(cot)
        deep_resp.trace.tokens_consumed += cot_tokens + 1500
        deep_resp.cot = cot
        deep_resp.verifications = [v for v in verdicts if v]

    deep_resp.trace.tier = "deep-reasoning"
    deep_resp.tier_used = "deep-reasoning"
    return deep_resp
