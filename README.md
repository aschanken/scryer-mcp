# scryer-mcp — Local Search MCP Server

*Scryer gazes into the web's reflective surface and extracts what you need — search works without an API key, LLM works with whatever provider you bring.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A drop-in replacement for the [Exa](https://exa.ai) MCP search server that runs entirely locally. Uses DuckDuckGo for web search (no API key needed), `trafilatura` for content extraction, and an optional LLM backend for synthesis and structured data extraction (bring your own endpoint and key).

```
                    ┌─────────────────────────────────────┐
                    │           scryer-mcp Server          │
                    │                                      │
  MCP Client ──────►│  scryer_search                       │
  (Claude Code,     │  scryer_fetch_content                │
   Cursor, etc.)    │  scryer_extract_structured ───► LLM │
                    │  scryer_synthesize        ───► (opt) │
                    │  scryer_health                       │
                    │                                      │
                    │  Search: duckduckgo_search (no key)  │
                    │  Fetch:  trafilatura + httpx         │
                    │  Cache:  filesystem with lazy TTL    │
                    └─────────────────────────────────────┘
```

## Features

- **No API key for search** — DuckDuckGo requires no authentication; search works out of the box
- **Optional LLM endpoint** — Plug in any OpenAI-compatible API for synthesis and extraction (API key supported but not required)
- **Citation validation** — LLM-generated citations cross-checked against source URLs; hallucinated citations detected and filtered
- **Schema-validated extraction** — LLM structured output validated against the caller's JSON Schema; bad output rejected with clear error
- **5 quality tiers** — From instant snippets to deep multi-pass research with chain-of-thought and adversarial verification
- **5 MCP tools** — Search, fetch, extract, synthesize, health check
- **Graceful degradation** — Every tool handles failure cleanly; LLM-dependent features return errors without crashing
- **Content extraction** — `trafilatura` + `BeautifulSoup` fallback, boilerplate removed
- **Caching** — Filesystem cache with configurable TTLs (search: 1h, fetch: 24h), pruned on startup
- **Docker-ready** — Build an image for the Docker MCP toolkit or run directly as stdio

---

## Quick Start

```bash
# Install
pip install -e .

# Run (stdio — connects to any MCP client)
python -m scryer_mcp.server

# In another terminal, test with MCP Inspector:
npx @anthropic/mcp-inspector python -m scryer_mcp.server
```

---

## MCP Tools

### `scryer_search`

Search the web using DuckDuckGo with configurable quality tiers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `string` | (required) | The search query |
| `num_results` | `integer` | `10` | Number of results (1–50, capped by tier) |
| `category` | `string` | `null` | Filter: `company`, `people`, `news`, `research`, `paper`, `code` |
| `highlights` | `boolean` | `true` | Return token-efficient content extracts |
| `full_text` | `boolean` | `false` | Return full page content (forces livecrawl) |
| `livecrawl` | `boolean` | `true` | Fetch fresh content; `false` = snippets only |
| `tier` | `string` | `"auto"` | Quality tier (see below) |
| `max_tokens` | `integer` | `null` | Soft cap on total output tokens |

**Response:** Returns `ScryerResponse` JSON with results, trace metadata, and optional grounded answer + citations.

### `scryer_fetch_content`

Fetch and extract clean content from specific URLs without searching.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | `string[]` | (required) | URLs to fetch (max 15) |
| `mode` | `string` | `"highlights"` | `"highlights"` (~200 words) or `"full_text"` |
| `timeout_ms` | `integer` | `8000` | Per-URL timeout in milliseconds |

**Response:** `{ results: [{url, title, content, status, error}], fetched, failed }`

### `scryer_extract_structured`

Extract structured data from URLs according to a JSON Schema. Uses the LLM to extract per-page, then merges and deduplicates. **Requires `LLM_ENDPOINT` to be configured.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | `string[]` | (required) | URLs to extract data from (max 10) |
| `schema` | `object` | (required) | JSON Schema defining the extraction shape |
| `mode` | `string` | `"highlights"` | Content extraction mode |

**Example schema:**
```json
{
  "type": "object",
  "properties": {
    "company_name": {"type": "string"},
    "ceo": {"type": "string"},
    "founded": {"type": "string"},
    "revenue": {"type": "string"}
  }
}
```

### `scryer_synthesize`

Synthesize a grounded, cited answer from a set of search results. Produces 2–3 paragraphs with inline `[cite: url]` references. **Requires `LLM_ENDPOINT` to be configured.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `string` | (required) | The original search query |
| `results` | `object[]` | (required) | Array of `{title, url, highlights/snippet}` |

### `scryer_health`

Check connectivity of all backend dependencies — DDG search, HTTP fetch, LLM endpoint, cache directory. Does not require LLM for basic checks.

**Response:**
```json
{
  "version": "0.1.0",
  "all_ok": true,
  "checks": {
    "ddg_search": {"ok": true, "detail": "returned 1 results"},
    "http_fetch": {"ok": true, "detail": "status 200"},
    "llm_endpoint": {"ok": true, "detail": "reachable"},
    "cache": {"ok": true, "detail": "/root/.cache/scryer-mcp"}
  }
}
```

---

## Quality Tiers

The `tier` parameter on `scryer_search` controls the depth of processing.

| Tier | Searches | URLs Fetched | LLM Synthesis | Target Time | Use Case |
|------|----------|-------------|---------------|-------------|----------|
| `instant` | 1 | 0 | None | <3s | Quick glance |
| `fast` | 1 | 3–5 | None | <15s | Get highlights |
| `auto` | 1 | 5–8 | Summary | <60s | General research |
| `deep` | 2–3 | 10–15 | Grounded answer | <180s | Thorough investigation |
| `deep-reasoning` | 3–5 | 15–20 | Verified answer | <300s | Rigorous verification |

### Tier Logic

**instant:** Single DDG search, return snippets. No fetching, no extraction.

**fast:** DDG search → parallel fetch top URLs → highlights extraction (trafilatura) → return results with highlights.

**auto:** Same as fast → LLM synthesis of a 2–3 paragraph summary with citations.

**deep:** Auto tier → identify knowledge gaps → follow-up searches → fetch gap-filling URLs → comprehensive synthesis.

**deep-reasoning:** Deep tier → chain-of-thought analysis → 3-lens adversarial verification (accuracy, completeness, bias) → final verified synthesis.

---

## Configuration

All configuration is through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENDPOINT` | `http://localhost:8734/v1/chat/completions` | OpenAI-compatible API endpoint for synthesis/extraction. **Optional.** Set to your provider's URL (OpenAI, Anthropic, local proxy, etc.). |
| `SCRYER_LLM_MODEL` | `deepseek-v4-flash` | Model name sent in LLM API requests. Set to any model your endpoint supports. |
| `SCRYER_CACHE_DIR` | `~/.cache/scryer-mcp` | Filesystem cache location |

**When `LLM_ENDPOINT` is unset or unreachable:**
- `scryer_search` (instant, fast) — works fully
- `scryer_search` (auto, deep, deep-reasoning) — returns results without grounded answer, sets `error` field
- `scryer_fetch_content` — works fully
- `scryer_extract_structured` — returns `{error: "LLM endpoint unavailable"}`
- `scryer_synthesize` — returns `{error: "LLM endpoint unavailable"}`
- `scryer_health` — works fully, reports LLM as "unavailable"

---

## Registration

### Claude Code

```bash
# Replace ENDPOINT/MODEL with your provider's values (omit both for search-only)
claude mcp add scryer \
  -e LLM_ENDPOINT=http://localhost:8734/v1/chat/completions \
  -e SCRYER_LLM_MODEL=deepseek-v4-flash \
  -- python -m scryer_mcp.server
```

### Cursor / Any MCP Client

*Replace ENDPOINT and MODEL with your provider's values. Both env vars are optional — omit for search-only.*

```json
{
  "mcpServers": {
    "scryer": {
      "command": "python",
      "args": ["-m", "scryer_mcp.server"],
      "env": {
        "LLM_ENDPOINT": "http://localhost:8734/v1/chat/completions",
        "SCRYER_LLM_MODEL": "deepseek-v4-flash"
      }
    }
  }
}
```

### Docker MCP Toolkit

```bash
docker build -t scryer-mcp .
docker mcp profile server add default --server docker://scryer-mcp:latest
docker mcp catalog add my-catalog scryer ./catalog.yaml
```

---

## Docker

```dockerfile
FROM python:3.12-slim
# Dependencies installed, source copied, entrypoint set
```

```bash
# Build
docker build -t scryer-mcp .

# Run with an LLM endpoint (replace ENDPOINT and MODEL with your provider)
docker run --rm \
  -e LLM_ENDPOINT=http://host.docker.internal:8734/v1/chat/completions \
  -e SCRYER_LLM_MODEL=deepseek-v4-flash \
  scryer-mcp

# Run with no LLM (search + fetch only — works out of the box)
docker run --rm scryer-mcp
```

**Security:** The Docker image contains no API keys, no secrets, and no credentials. The `LLM_ENDPOINT` defaults to `localhost:8734` — set it at runtime only if you need synthesis/extraction. An API key can be injected via `SCRYER_API_KEY` (Docker secrets or env var). When set, a runtime warning is emitted if the endpoint uses HTTP to a non-local address.

---

## Architecture

```
src/scryer_mcp/
├── __init__.py          Package metadata
├── server.py            FastMCP entry point, 5 tool definitions
├── tiers.py             Tier dispatch: instant → fast → auto → deep → deep-reasoning
├── search.py            DDG search adapter with jittered 429 retry
├── fetch.py             HTTP fetch + trafilatura extraction + bs4 fallback
├── llm_client.py        Async LLM client, optional synthesis/extraction
├── cache.py             Filesystem cache with lazy TTL enforcement
├── schema.py            Pydantic v2 request/response models
└── extract.py           Structured data merge/dedup
```

### Key Design Decisions

**Map-Reduce pattern.** Fetch and extraction are parallelized with `asyncio.gather`. The LLM-independent path never blocks on the LLM-dependent path.

**Graceful degradation.** Every tool has a fallback chain. If all fetches fail at `fast` tier, it downgrades to `instant`. If the LLM is down, synthesis returns an error but search results are still delivered.

**Search requires no API key.** DuckDuckGo web search works out of the box — no registration, no token. The optional LLM endpoint supports Bearer token auth via `SCRYER_API_KEY` for providers that require it, or no auth for local proxies.

**Citation validation.** When the LLM generates citations, they are cross-checked against the actual search results. Hallucinated or invented URLs are filtered out, and the count of hallucinated citations is surfaced in the response. This prevents the LLM from fabricating sources.

**Schema-validated extraction.** Structured data extraction validates the LLM's output against the caller's JSON Schema. If the LLM returns data that doesn't match (wrong types, missing fields, extra keys), the extraction returns an error rather than silently passing corrupt data to the caller.

**Jittered retry on 429s.** DuckDuckGo rate-limits aggressively. The search adapter uses exponential backoff with jitter to handle this transparently.

---

## Development

```bash
# Install
pip install -e .
pip install pytest pytest-asyncio

# Run unit tests (no network)
python -m pytest tests/ -v --ignore=tests/test_integration.py

# Run integration tests (requires internet)
python -m pytest tests/test_integration.py -v

# Run all tests
python -m pytest tests/ -v
```

### Project Structure

```
scryer-mcp/
├── src/scryer_mcp/          # Source code
├── tests/                    # Tests (59 unit + 4 integration)
├── Dockerfile                # OCI image
├── catalog.yaml              # Docker MCP toolkit catalog entry
├── pyproject.toml            # Package metadata
└── README.md                 # This file
```

---

## Security

- **No secrets in the image.** The Docker image contains no API keys, tokens, passwords, or credentials. `SCRYER_API_KEY` is injected at runtime via env var or Docker secrets.
- **Endpoint-agnostic.** LLM endpoint is configured at runtime via `LLM_ENDPOINT`. Set it to any OpenAI-compatible API — OpenAI, Anthropic, a local proxy, or skip it entirely for search-only operation.
- **API key transport warning.** When `SCRYER_API_KEY` is set and the endpoint uses HTTP to a non-local address, Scryer emits a runtime warning. Localhost HTTP (the default) is safe — traffic never leaves the machine.
- **No data leakage.** Outbound HTTP requests go only to DuckDuckGo and optionally your LLM endpoint. No telemetry, tracking, or analytics.
- **Cache is local.** All cached data stays in `SCRYER_CACHE_DIR`. Nothing is sent anywhere.
- **Tracebacks not exposed.** Internal error details are never returned to the MCP client; only `str(e)` error messages are surfaced.

---

## Changelog

### v0.1.1 (2026-05-30)

**Security fixes:**
- Full tracebacks no longer leaked in error responses (use `str(e)` only)
- API key cleartext warning when endpoint is HTTP and remote
- HTTP response body size limited via `httpx.Limits` (5 MB cap)
- Connection pooling — single `AsyncClient` per fetch batch instead of per-URL

**Data integrity fixes:**
- `livecrawl=False` now correctly prevents live HTTP fetches (was inverted)
- Deep-reasoning adversarial verdicts stored on response (were discarded)
- LLM citation hallucination guarded — citations cross-checked against source URLs
- LLM extraction output validated against caller-supplied JSON Schema
- Naive JSON extractor replaced with robust version (handles arrays, embedded braces)
- Gap-filling no longer appends duplicate URLs
- `full_text` parameter actually selects full-text fetch mode

**Quality of life:**
- URL list truncation signalled via `truncated`/`original_count` fields
- Content truncation flagged in extraction responses
- Token counting based on actual content length (chars / 4)
- Cache pruned on startup (stale entries cleaned)
- Health check uses `os.access()` instead of file write
- DDG library updated from deprecated `duckduckgo_search` to `ddgs`
- `_get_client` race condition guarded with `asyncio.Lock`
- Dead code removed (`noise_strip.py`)
- Duplicate `validate_extraction_schema` consolidated

**Tests:** 63 total (59 unit + 4 integration) — 19 new tests covering all fixes.

### v0.1.0 (2026-05-29)

Initial release. 5 MCP tools, 5 quality tiers, DDG search, optional LLM synthesis.

---

## License

MIT

---

## Etymology

A **scryer** is one who practices scrying — the ancient art of gazing into a reflective surface (a crystal ball, a pool of water, a polished mirror) to perceive distant or hidden knowledge. This server does the same for the web: it gazes through DuckDuckGo, extracts meaning from what it finds, and reports back what's worth knowing. No mysticism required — just good engineering and a touch of whimsy.
