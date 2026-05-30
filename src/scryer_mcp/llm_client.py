"""Async DeepSeek API client for synthesis and structured extraction."""
from __future__ import annotations

import asyncio
import json
import os
import httpx
import warnings


class LLMClient:
    """Async client for LLM-dependent features.

    Points at the token proxy by default. Degrades gracefully:
    if unavailable, synthesis/extraction tools return an error message.
    """

    def __init__(self, endpoint: str | None = None, model: str | None = None,
                 api_key: str | None = None):
        self.endpoint = endpoint or os.getenv("LLM_ENDPOINT") or "http://localhost:8734/v1/chat/completions"
        self.model = model or os.getenv("SCRYER_LLM_MODEL", "deepseek-v4-flash")
        self.api_key = api_key or os.getenv("SCRYER_API_KEY")
        self._client: httpx.AsyncClient | None = None
        self._client_lock: asyncio.Lock | None = None
        if self.api_key:
            parsed = httpx.URL(self.endpoint)
            is_local = parsed.host in ("localhost", "127.0.0.1", "0.0.0.0")
            if parsed.scheme == "http" and not is_local:
                warnings.warn(
                    f"API key will be transmitted in cleartext to {parsed.host} "
                    f"via HTTP. Use HTTPS in production.",
                    stacklevel=2,
                )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            if self._client_lock is None:
                self._client_lock = asyncio.Lock()
            async with self._client_lock:
                if self._client is None:  # Double-check after acquiring lock
                    headers = {}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    self._client = httpx.AsyncClient(timeout=60.0, headers=headers)
        return self._client

    async def is_available(self) -> bool:
        """Check if the LLM endpoint is reachable."""
        try:
            client = await self._get_client()
            health_url = self.endpoint.rsplit("/", 1)[0] + "/health"
            resp = await client.get(health_url, timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def _chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str | None:
        """Send a chat completion request. Returns text or None on failure."""
        try:
            client = await self._get_client()
            resp = await client.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            return None
        except Exception:
            return None

    async def synthesize(self, query: str, results: list[dict]) -> dict:
        """Synthesize a grounded, cited answer from search results.

        Returns {"grounded_answer": str, "citations": [str]} or error dict.
        """
        if not await self.is_available():
            return {"error": "LLM endpoint unavailable — synthesis requires LLM_ENDPOINT"}

        results_json = _format_results(results)
        system = (
            "You are a precise research synthesizer. Write a 2-3 paragraph summary "
            "from the provided search results. For every factual claim, cite the "
            "source URL inline as [cite: url]. If sources conflict, note the "
            "disagreement. If information is missing, state what was not found. "
            "Use specific numbers, names, and dates. Do NOT invent information "
            "not present in the results. Return ONLY a JSON object: "
            '{"grounded_answer": "...", "citations": ["url1", ...]}'
        )
        user = f"Query: {query}\n\nSearch Results:\n{results_json}"

        text = await self._chat(system, user, temperature=0.3, max_tokens=2000)
        if not text:
            return {"error": "LLM synthesis failed — no response"}

        try:
            parsed = json.loads(_extract_json(text))
            if isinstance(parsed, dict):
                citations = parsed.get("citations", [])
                result_urls = [r.get("url", "") for r in results]
                valid = [c for c in citations if any(c in u for u in result_urls)]
                parsed["citations"] = valid
                parsed["_hallucinated_citations"] = len(citations) - len(valid)
            return parsed
        except Exception:
            return {"error": "LLM synthesis returned unparseable output"}

    async def extract_structured(self, content: str, schema: dict) -> dict:
        """Extract structured data from content per JSON Schema."""
        if not await self.is_available():
            return {"error": "LLM endpoint unavailable"}

        schema_json = json.dumps(schema)
        system = (
            "You are a precise data extractor. Extract structured data from the "
            "provided content according to the JSON Schema. If a field cannot be "
            "found, use null. Do NOT hallucinate values. Return ONLY a JSON object "
            "conforming exactly to the schema."
        )
        truncated = len(content) > 8000
        user = f"Schema: {schema_json}\n\nContent:\n{content[:8000]}"
        if truncated:
            user += "\n\n[Note: Content was truncated to 8000 characters.]"

        text = await self._chat(system, user, temperature=0.0, max_tokens=2000)
        if not text:
            return {"error": "LLM extraction failed — no response"}

        try:
            import jsonschema as _js
            parsed = json.loads(_extract_json(text))
            _js.validate(instance=parsed, schema=schema)
            return parsed
        except _js.ValidationError as e:
            return {"error": f"LLM extraction did not conform to schema: {e.message}"}
        except ImportError:
            warnings.warn("jsonschema not installed; extraction output not validated")
            try:
                return json.loads(_extract_json(text))
            except Exception:
                return {"error": "LLM extraction returned unparseable output"}
        except Exception:
            return {"error": "LLM extraction returned unparseable output"}

    async def classify_category(
        self, results: list[dict], category: str
    ) -> list[str]:
        """Classify search results by category, return matching URLs."""
        if not await self.is_available():
            return [r["url"] for r in results]

        urls_json = json.dumps([
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": r.get("snippet", "")}
            for r in results
        ])
        system = (
            f"Classify each URL as matching the category '{category}' or not. "
            "Return a JSON array of URLs that match the target category. "
            "Be conservative — if uncertain, exclude it."
        )
        text = await self._chat(system, urls_json, temperature=0.0, max_tokens=500)
        try:
            return json.loads(_extract_json(text or "[]"))
        except Exception:
            return [r["url"] for r in results]


def _format_results(results: list[dict]) -> str:
    items = []
    for r in results:
        items.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "highlights": r.get("highlights", r.get("snippet", "")),
        })
    return json.dumps(items, indent=2)


def _extract_json(text: str) -> str:
    """Extract the first complete JSON object or array from text.

    Handles embedded braces/brackets inside string values.
    Finds whichever brace ({ or [) appears first.
    """
    start = -1
    close_char = None
    for brace, close in [("{", "}"), ("[", "]")]:
        pos = text.find(brace)
        if pos != -1 and (start == -1 or pos < start):
            start = pos
            close_char = close
    if start == -1:
        return text
    in_string = False
    escape = False
    depth = 1
    for i in range(start + 1, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "[{":
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text
