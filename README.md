# scryer-mcp — Local Search MCP Server

*Scryer gazes into the web's reflective surface and extracts what you need — no API key required.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A drop-in replacement for the [Exa](https://exa.ai) MCP search server that runs entirely locally — **no API keys required**. Uses DuckDuckGo for web search, `trafilatura` for content extraction, and an optional LLM backend for synthesis and structured data extraction.

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

- **Zero API keys** — Web search and content extraction work unconditionally
- **5 quality tiers** — From instant snippets to deep multi-pass research
- **5 MCP tools** — Search, fetch, extract, synthesize, health check
- **Optional LLM synthesis** — Plug in any OpenAI-compatible endpoint for grounded answers and structured extraction
- **Graceful degradation** — Every tool handles failure cleanly; LLM-dependent features return errors without crashing
- **Content extraction** — `trafilatura` + `BeautifulSoup` fallback, boilerplate removed
- **Caching** — Filesystem cache with configurable TTLs (search: 1h, fetch: 24h)
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
| `LLM_ENDPOINT` | `http://localhost:8734/v1/chat/completions` | OpenAI-compatible API endpoint for synthesis/extraction. **Optional.** |
| `SCRYER_LLM_MODEL` | `deepseek-v4-flash` | Model name sent in LLM API requests |
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
claude mcp add scryer -e LLM_ENDPOINT=http://localhost:8734/v1/chat/completions \
  -- python -m scryer_mcp.server
```

### Cursor / Any MCP Client

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

# Run (stdio)
docker run --rm -e LLM_ENDPOINT=http://host:8734/v1/chat/completions scryer-mcp

# Run with no LLM (search + fetch only)
docker run --rm scryer-mcp
```

**Security:** The Docker image contains no API keys, no secrets, and no credentials. The `LLM_ENDPOINT` is empty by default — set it at runtime only if you need synthesis/extraction.

---

## Architecture

```
src/scryer_mcp/
├── __init__.py          Package metadata
├── server.py            FastMCP entry point, 5 tool definitions (230 lines)
├── tiers.py             Tier dispatch: instant → fast → auto → deep → deep-reasoning (279 lines)
├── search.py            DDG search adapter with jittered 429 retry (71 lines)
├── fetch.py             HTTP fetch + trafilatura extraction + bs4 fallback (149 lines)
├── llm_client.py        Async LLM client, optional synthesis/extraction (182 lines)
├── cache.py             Filesystem cache with lazy TTL enforcement (79 lines)
├── schema.py            Pydantic v2 request/response models
├── noise_strip.py       HTML/markdown boilerplate removal
└── extract.py           Structured data merge/dedup
```

### Key Design Decisions

**Map-Reduce pattern.** Fetch and extraction are parallelized with `asyncio.gather`. The LLM-independent path never blocks on the LLM-dependent path.

**Graceful degradation.** Every tool has a fallback chain. If all fetches fail at `fast` tier, it downgrades to `instant`. If the LLM is down, synthesis returns an error but search results are still delivered.

**No API keys.** DuckDuckGo requires no authentication. The LLM endpoint is optional and user-configured.

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
├── tests/                    # Tests (34 unit + 4 integration)
├── Dockerfile                # OCI image
├── catalog.yaml              # Docker MCP toolkit catalog entry
├── pyproject.toml            # Package metadata
└── README.md                 # This file
```

---

## Security

- **No secrets in the image.** The Docker image contains no API keys, tokens, passwords, or credentials.
- **No hardcoded endpoints.** The LLM endpoint is configured at runtime via `LLM_ENDPOINT`. Default (`localhost:8734`) only applies when explicitly set.
- **No data leakage.** Outbound HTTP requests go to DuckDuckGo and optionally your LLM endpoint. No telemetry, tracking, or analytics.
- **Cache is local.** All cached data stays in `SCRYER_CACHE_DIR`. Nothing is sent anywhere.

---

## License

MIT

---

## Etymology

A **scryer** is one who practices scrying — the ancient art of gazing into a reflective surface (a crystal ball, a pool of water, a polished mirror) to perceive distant or hidden knowledge. This server does the same for the web: it gazes through DuckDuckGo, extracts meaning from what it finds, and reports back what's worth knowing. No mysticism required — just good engineering and a touch of whimsy.
