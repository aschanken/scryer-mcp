"""URL fetch with trafilatura extraction and boilerplate removal."""
from __future__ import annotations

import asyncio
import re
import httpx
from trafilatura import extract as trafilatura_extract
from bs4 import BeautifulSoup


async def fetch_url(
    url: str,
    client: httpx.AsyncClient,
    mode: str = "highlights",
) -> dict:
    """Fetch and extract content from a single URL.

    Args:
        url: The URL to fetch.
        client: Shared httpx.AsyncClient.
        mode: "highlights" (150-200 words) or "full_text".

    Returns:
        Dict with keys: url, title, content, status, error.
    """
    try:
        resp = await client.get(url, follow_redirects=True)
        if resp.status_code != 200:
            return {
                "url": url, "title": None, "content": None,
                "status": resp.status_code,
                "error": f"HTTP {resp.status_code}",
            }

        html = resp.text
        # Extract title
        title = _extract_title(html)

        # Try trafilatura first (best for articles/news)
        content = trafilatura_extract(
            html, output_format="markdown",
            include_links=False, include_images=False,
            include_tables=False,
        )

        # Fallback to BeautifulSoup if trafilatura returns nothing
        if not content or len(content.strip()) < 50:
            content = _bs4_extract(html)

        if not content:
            return {
                "url": url, "title": title, "content": None,
                "status": 200, "error": "No extractable content",
            }

        if mode == "highlights":
            content = _truncate_words(content, 200)

        return {
            "url": url, "title": title, "content": content,
            "status": 200, "error": None,
        }

    except httpx.TimeoutException:
        return {
            "url": url, "title": None, "content": None,
            "status": 0, "error": "timeout",
        }
    except Exception as e:
        return {
            "url": url, "title": None, "content": None,
            "status": 0, "error": str(e),
        }


def _extract_title(html: str) -> str | None:
    """Extract the page title from HTML."""
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _bs4_extract(html: str) -> str:
    """Fallback content extraction using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, "lxml")
        # Remove noisy elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        # Get main content
        for sel in ["article", "main", '[role="main"]', ".content", ".post", ".article"]:
            el = soup.select_one(sel)
            if el:
                return el.get_text(separator="\n", strip=True)
        # Fallback to body
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)[:5000]
        return ""
    except Exception:
        return ""


async def fetch_urls(
    urls: list[str],
    mode: str = "highlights",
    timeout_ms: int = 8000,
    max_concurrent: int = 10,
) -> list[dict]:
    """Fetch and extract content from multiple URLs in parallel.

    Args:
        urls: URLs to fetch (capped at 15).
        mode: "highlights" or "full_text".
        timeout_ms: Per-URL timeout in milliseconds.
        max_concurrent: Maximum concurrent fetches.

    Returns:
        List of result dicts, one per URL.
    """
    sem = asyncio.Semaphore(max_concurrent)
    timeout_s = timeout_ms / 1000.0
    limits = httpx.Limits(max_connections=max_concurrent, max_keepalive_connections=5)

    async def fetch_with_sem(client: httpx.AsyncClient, url: str) -> dict:
        async with sem:
            return await fetch_url(url, client, mode)

    async with httpx.AsyncClient(
        timeout=timeout_s, limits=limits, follow_redirects=True
    ) as client:
        tasks = [fetch_with_sem(client, u) for u in urls[:15]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for url, result in zip(urls[:15], results):
        if isinstance(result, BaseException):
            out.append({
                "url": url, "title": None, "content": None,
                "status": 0, "error": str(result),
            })
        else:
            out.append(result)
    return out


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"
