# scryer-mcp — Local scryer search MCP server
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    mcp \
    duckduckgo-search \
    httpx \
    trafilatura \
    beautifulsoup4 \
    lxml \
    pydantic

COPY src/ /app/src/
COPY pyproject.toml /app/

WORKDIR /app
RUN pip install -e . --no-deps

ENV LLM_ENDPOINT=""
ENV SCRYER_LLM_MODEL="deepseek-v4-flash"
ENV SCRYER_CACHE_DIR="/root/.cache/scryer-mcp"

ENTRYPOINT ["python", "-m", "scryer_mcp.server"]
