# scryer-mcp — Local scryer search MCP server
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    mcp \
    duckduckgo-search \
    httpx \
    trafilatura \
    beautifulsoup4 \
    lxml \
    pydantic \
    jsonschema

COPY src/ /app/src/
COPY pyproject.toml /app/

WORKDIR /app
RUN pip install -e . --no-deps

ENV LLM_ENDPOINT="https://api.deepseek.com/v1/chat/completions"
ENV SCRYER_LLM_MODEL="claude-3-5-haiku-latest"
ENV SCRYER_DEEPSEEK_API_KEY=""
ENV SCRYER_ANTHROPIC_API_KEY=""
ENV SCRYER_CACHE_DIR="/root/.cache/scryer-mcp"

ENTRYPOINT ["python", "-m", "scryer_mcp.server"]
