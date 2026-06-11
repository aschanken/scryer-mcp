"""Async LLM client with dual Anthropic / OpenAI-compatible backends.

Auto-detects backend from model name prefix:
  - 'claude*' → Anthropic Messages API
  - 'deepseek*' → OpenAI-compatible chat completions

Falls back to the other backend when the primary is unreachable.
"""

from __future__ import annotations

import asyncio
import json
import os
import httpx
import warnings


class LLMClient:
    """Async client for LLM-dependent features with dual-backend support.

    Supports two backends:
      - **Anthropic** (claude-3-5-haiku-latest, etc.): POST to the
        ``/v1/messages`` endpoint using ``x-api-key`` auth.
      - **OpenAI-compat** (deepseek-chat, gpt-4o, etc.): POST to an
        arbitrary ``LLM_ENDPOINT`` using ``Bearer`` token auth.

    Backend is selected by model name prefix.  On failure (missing key,
    HTTP error, timeout) the other backend is tried as fallback.
    """

    FALLBACK_ANTHROPIC_MODEL = "claude-3-5-haiku-latest"
    FALLBACK_OPENAI_MODEL = "deepseek-chat"
    ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        anthropic_api_key: str | None = None,
    ) -> None:
        # OpenAI-compatible config (DeepSeek, etc.)
        self.openai_endpoint = (
            endpoint
            or os.getenv("LLM_ENDPOINT")
            or "https://api.deepseek.com/v1/chat/completions"
        )
        self.openai_api_key = api_key or os.getenv("SCRYER_DEEPSEEK_API_KEY")

        # Anthropic config
        self.anthropic_api_key = anthropic_api_key or os.getenv(
            "SCRYER_ANTHROPIC_API_KEY"
        )

        # Model + backend detection
        self.model = model or os.getenv("SCRYER_LLM_MODEL", "deepseek-v4-flash")
        self._primary_backend = self._detect_backend(self.model)
        self._last_backend_used: str | None = None

        self._client: httpx.AsyncClient | None = None
        self._client_lock: asyncio.Lock | None = None

        # Warn if sending an API key over cleartext to a non-local endpoint
        if self.openai_api_key:
            parsed = httpx.URL(self.openai_endpoint)
            is_local = parsed.host in ("localhost", "127.0.0.1", "0.0.0.0")
            if parsed.scheme == "http" and not is_local:
                warnings.warn(
                    f"API key will be transmitted in cleartext to {parsed.host} "
                    f"via HTTP. Use HTTPS in production.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend(model: str) -> str:
        """Detect API backend from model name prefix.

        Returns ``"anthropic"`` or ``"openai"``.
        """
        ml = model.lower()
        if ml.startswith("claude"):
            return "anthropic"
        if ml.startswith("deepseek"):
            return "openai"
        if any(ml.startswith(p) for p in ("gpt-", "o1", "o3", "o4")):
            return "openai"
        # Unknown → default to Anthropic (matches the default model)
        return "anthropic"

    # ------------------------------------------------------------------
    # HTTP client (shared, no auth headers — added per-request)
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            if self._client_lock is None:
                self._client_lock = asyncio.Lock()
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    # ------------------------------------------------------------------
    # Availability / health
    # ------------------------------------------------------------------

    async def is_available(self) -> bool:
        """Check if at least one LLM backend is reachable and configured."""
        status = await self.check_all_backends()
        return status["anthropic"]["ok"] or status["openai"]["ok"]

    async def check_all_backends(self) -> dict:
        """Check both backends.  Returns detailed status for health reporting."""
        result: dict = {
            "anthropic": {
                "ok": False,
                "detail": "",
                "api_key_configured": bool(self.anthropic_api_key),
            },
            "openai": {
                "ok": False,
                "detail": "",
                "api_key_configured": bool(self.openai_api_key),
            },
        }

        # -- Anthropic --
        if self.anthropic_api_key:
            try:
                client = await self._get_client()
                resp = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": self.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    timeout=5.0,
                )
                result["anthropic"]["ok"] = resp.status_code in (200, 401)
                result["anthropic"]["detail"] = f"HTTP {resp.status_code}"
            except Exception as e:
                result["anthropic"]["detail"] = str(e)
        else:
            result["anthropic"]["detail"] = "SCRYER_ANTHROPIC_API_KEY not set"

        # -- OpenAI-compat --
        if self.openai_api_key:
            try:
                client = await self._get_client()
                base_url = self.openai_endpoint.rsplit("/", 1)[0]
                resp = await client.get(
                    base_url + "/models",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    timeout=5.0,
                )
                result["openai"]["ok"] = resp.status_code in (200, 401)
                result["openai"]["detail"] = f"HTTP {resp.status_code}"
            except Exception as e:
                result["openai"]["detail"] = str(e)
        else:
            result["openai"]["detail"] = "SCRYER_DEEPSEEK_API_KEY not set"

        return result

    # ------------------------------------------------------------------
    # Chat (dispatch with fallback)
    # ------------------------------------------------------------------

    async def _chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str | None:
        """Send a chat request.  Tries primary backend, then fallback."""
        result = await self._chat_backend(
            self._primary_backend, system, user, temperature, max_tokens, self.model
        )
        if result is not None:
            return result

        # Fallback to the other backend
        fallback = "openai" if self._primary_backend == "anthropic" else "anthropic"
        fallback_model = (
            self.FALLBACK_OPENAI_MODEL
            if fallback == "openai"
            else self.FALLBACK_ANTHROPIC_MODEL
        )
        return await self._chat_backend(
            fallback, system, user, temperature, max_tokens, fallback_model
        )

    async def _chat_backend(
        self,
        backend: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> str | None:
        """Route to the correct backend implementation."""
        if backend == "anthropic":
            return await self._chat_anthropic(
                system, user, temperature, max_tokens, model
            )
        return await self._chat_openai(
            system, user, temperature, max_tokens, model
        )

    async def _chat_openai(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> str | None:
        """OpenAI-compatible chat completion (DeepSeek, GPT, etc.)."""
        if not self.openai_api_key:
            return None
        try:
            client = await self._get_client()
            resp = await client.post(
                self.openai_endpoint,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                self._last_backend_used = "openai"
                return data["choices"][0]["message"]["content"]
            return None
        except Exception:
            return None

    async def _chat_anthropic(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> str | None:
        """Anthropic Messages API call."""
        if not self.anthropic_api_key:
            return None
        try:
            client = await self._get_client()
            resp = await client.post(
                self.ANTHROPIC_ENDPOINT,
                json={
                    "model": model,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                self._last_backend_used = "anthropic"
                # Anthropic returns content as a list of content blocks
                content_blocks = data.get("content", [])
                text_blocks = [
                    b["text"] for b in content_blocks if b.get("type") == "text"
                ]
                return "\n".join(text_blocks) if text_blocks else None
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        query: str,
        results: list[dict],
        prompt: str | None = None,
    ) -> dict:
        """Synthesize a grounded, cited answer from search results.

        Args:
            query: The original search query.
            results: Search results to synthesize from.
            prompt: Optional instructional prompt appended to the system message
                    to guide the synthesis (additive, does not replace built-in
                    instructions).

        Returns:
            ``{"grounded_answer": str, "citations": [str]}`` or error dict.
        """
        if not await self.is_available():
            return {
                "error": (
                    "LLM unavailable — configure SCRYER_ANTHROPIC_API_KEY "
                    "(Anthropic) or SCRYER_DEEPSEEK_API_KEY (DeepSeek) and ensure "
                    "the endpoint is reachable"
                )
            }

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
        if prompt:
            system += f"\n\nAdditional instruction from the user:\n{prompt}"
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

    # ------------------------------------------------------------------
    # Structured extraction
    # ------------------------------------------------------------------

    async def extract_structured(self, content: str, schema: dict, prompt: str | None = None) -> dict:
        """Extract structured data from content per JSON Schema.

        Args:
            content: Raw text content to extract from.
            schema: JSON Schema defining the extraction shape.
            prompt: Optional instructional prompt appended to the system message
                    to guide extraction (additive, does not replace built-in
                    instructions).
        """
        if not await self.is_available():
            return {"error": "LLM endpoint unavailable"}

        schema_json = json.dumps(schema)
        system = (
            "You are a precise data extractor. Extract structured data from the "
            "provided content according to the JSON Schema. If a field cannot be "
            "found, use null. Do NOT hallucinate values. Return ONLY a JSON object "
            "conforming exactly to the schema."
        )
        if prompt:
            system += f"\n\nAdditional instruction from the user:\n{prompt}"
        char_cap = 8000
        truncated = len(content) > char_cap
        user = f"Schema: {schema_json}\n\nContent:\n{content[:char_cap]}"
        if truncated:
            user += "\n\n[Note: Content was truncated to 8000 characters.]"
        # If content appears word-truncated upstream, flag it so the LLM
        # doesn't treat an incomplete final item as authoritative.
        if content.rstrip().endswith("…"):
            user += (
                "\n\n[Note: The source content was word-truncated. "
                "If the last item appears incomplete, omit it rather than "
                "including a partial value.]"
            )

        text = await self._chat(system, user, temperature=0.0, max_tokens=2000)
        if not text:
            return {"error": "LLM extraction failed — no response"}

        try:
            import jsonschema as _js

            parsed = json.loads(_extract_json(text))
            _js.validate(instance=parsed, schema=schema)
            return parsed
        except _js.ValidationError as e:
            return {
                "error": f"LLM extraction did not conform to schema: {e.message}"
            }
        except ImportError:
            warnings.warn(
                "jsonschema not installed; extraction output not validated"
            )
            try:
                return json.loads(_extract_json(text))
            except Exception:
                return {"error": "LLM extraction returned unparseable output"}
        except Exception:
            return {"error": "LLM extraction returned unparseable output"}

    # ------------------------------------------------------------------
    # Category classification
    # ------------------------------------------------------------------

    async def classify_category(
        self, results: list[dict], category: str, prompt: str | None = None
    ) -> list[str]:
        """Classify search results by category, return matching URLs.

        Args:
            results: Search result dicts with title, url, snippet.
            category: Target category to match against.
            prompt: Optional instructional prompt appended to the system message.
        """
        if not await self.is_available():
            return [r["url"] for r in results]

        urls_json = json.dumps(
            [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", ""),
                }
                for r in results
            ]
        )
        system = (
            f"Classify each URL as matching the category '{category}' or not. "
            "Return a JSON array of URLs that match the target category. "
            "Be conservative — if uncertain, exclude it."
        )
        if prompt:
            system += f"\n\nAdditional instruction from the user:\n{prompt}"
        text = await self._chat(system, urls_json, temperature=0.0, max_tokens=500)
        try:
            return json.loads(_extract_json(text or "[]"))
        except Exception:
            return [r["url"] for r in results]


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _format_results(results: list[dict]) -> str:
    items = []
    for r in results:
        items.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "highlights": r.get("highlights", r.get("snippet", "")),
            }
        )
    return json.dumps(items, indent=2)


def _extract_json(text: str) -> str:
    """Extract the first complete JSON object or array from *text*.

    Handles embedded braces/brackets inside string values.
    Finds whichever brace (``{`` or ``[``) appears first.
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
                return text[start : i + 1]
    return text
