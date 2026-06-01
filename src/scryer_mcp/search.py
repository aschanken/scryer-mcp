"""DDG search adapter with jittered retry on rate limiting."""
from __future__ import annotations

import asyncio
import random
import warnings
from duckduckgo_search import DDGS


def _augment_query(query: str, category: str | None) -> str:
    """Append category-specific terms to improve result relevance."""
    if not category:
        return query
    augment = {
        "company": " company profile",
        "people": " biography profile",
        "news": " news",
        "research": " research paper study",
        "paper": " research paper study",
        "code": " github repository code",
    }
    return query + augment.get(category, "")


async def ddg_search(
    query: str,
    num_results: int = 10,
    category: str | None = None,
    max_retries: int = 3,
) -> list[dict]:
    """Run a DDG search with jittered retry on rate limiting.

    Args:
        query: The search query.
        num_results: Number of results to request (DDG may return fewer).
        category: Optional category filter for query augmentation.
        max_retries: Maximum retry attempts on rate-limiting (429).

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    augmented = _augment_query(query, category)

    for attempt in range(max_retries):
        try:
            # DDGS is synchronous, run in thread to avoid blocking
            def _search():
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(augmented, max_results=num_results):
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        })
                return results

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _search)

        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "too many" in msg:
                if attempt < max_retries - 1:
                    # Jittered exponential backoff: 1s, 2s, 4s + jitter
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                # Rate-limit retries exhausted — surface it
                warnings.warn(f"DDG rate limit exceeded after {max_retries} attempts")
            return []

    return []
