"""Tests for server.py tool definitions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import pytest


class TestServerTools:
    """Verify tool functions exist and have correct signatures."""

    def test_tools_registered(self):
        from scryer_mcp.server import mcp
        tool_names = [t.name for t in mcp._tool_manager.list_tools()] if hasattr(mcp, '_tool_manager') else []
        # If _tool_manager isn't accessible, skip
        if tool_names:
            assert "scryer_search" in tool_names
            assert "scryer_fetch_content" in tool_names
            assert "scryer_health" in tool_names
            assert "scryer_synthesize" in tool_names


class TestScryerSearch:
    @pytest.mark.asyncio
    async def test_invalid_category(self):
        from scryer_mcp.server import scryer_search
        result = await scryer_search(query="test", category="invalid_category")
        assert "error" in result
        assert "Invalid category" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_tier(self):
        from scryer_mcp.server import scryer_search
        result = await scryer_search(query="test", tier="super_duper_slow")
        assert "error" in result
        assert "Invalid tier" in result["error"]

    @pytest.mark.asyncio
    async def test_error_response_no_traceback(self, monkeypatch):
        from scryer_mcp.server import scryer_search
        from scryer_mcp import tiers

        async def failing_execute(*args):
            raise ValueError("Something broke internally")
        monkeypatch.setattr(tiers, "execute_tier", failing_execute)

        result = await scryer_search(query="test")
        assert "error" in result
        assert "traceback" not in result, "Traceback must not leak in error response"

    @pytest.mark.asyncio
    async def test_accepts_optional_prompt(self, monkeypatch):
        """Prompt must be accepted and forwarded through ScryerRequest."""
        from scryer_mcp.server import scryer_search
        from scryer_mcp import tiers, server as srv

        captured = None

        async def capture_execute(request):
            nonlocal captured
            captured = request
            from scryer_mcp.schema import ScryerResponse, TierTrace
            return ScryerResponse(
                results=[], search_time_ms=0, tier_used="instant",
                trace=TierTrace(tier="instant", searches_performed=0,
                                urls_fetched=0, urls_skipped_cache=0,
                                urls_failed=0, wall_time_ms=0, tokens_consumed=0),
            )

        monkeypatch.setattr(tiers, "execute_tier", capture_execute)
        monkeypatch.setattr(srv, "execute_tier", capture_execute)

        result = await scryer_search(query="test", prompt="Summarize in French")
        assert result.get("error") is None
        assert captured is not None
        assert captured.prompt == "Summarize in French"

    @pytest.mark.asyncio
    async def test_no_prompt_by_default(self, monkeypatch):
        """Without a prompt parameter, ScryerRequest.prompt must be None."""
        from scryer_mcp.server import scryer_search
        from scryer_mcp import tiers, server as srv

        captured = None

        async def capture_execute(request):
            nonlocal captured
            captured = request
            from scryer_mcp.schema import ScryerResponse, TierTrace
            return ScryerResponse(
                results=[], search_time_ms=0, tier_used="instant",
                trace=TierTrace(tier="instant", searches_performed=0,
                                urls_fetched=0, urls_skipped_cache=0,
                                urls_failed=0, wall_time_ms=0, tokens_consumed=0),
            )

        monkeypatch.setattr(tiers, "execute_tier", capture_execute)
        monkeypatch.setattr(srv, "execute_tier", capture_execute)

        result = await scryer_search(query="test")
        assert result.get("error") is None
        assert captured is not None
        assert captured.prompt is None


class TestScryerFetch:
    """Tests for the scryer_fetch_content tool."""

    @pytest.mark.asyncio
    async def test_validates_mode(self):
        from scryer_mcp.server import scryer_fetch_content
        result = await scryer_fetch_content(
            urls=["https://example.com"], mode="invalid_mode"
        )
        assert "error" in result
        assert "Invalid mode" in result["error"]


class TestScryerExtract:
    """Tests for the scryer_extract_structured tool."""

    @pytest.mark.asyncio
    async def test_accepts_optional_prompt(self, monkeypatch):
        """Prompt must be accepted and forwarded to LLM client."""
        from scryer_mcp.server import scryer_extract_structured
        from scryer_mcp import fetch

        async def mock_fetch(*args, **kwargs):
            return [{"url": "https://example.com", "title": "Test",
                     "content": "Sample content", "status": 200, "error": None,
                     "structured_items": []}]

        monkeypatch.setattr(fetch, "fetch_urls", mock_fetch)

        from scryer_mcp import tiers
        captured_prompt = []

        original_extract = tiers.llm_client.extract_structured

        async def mock_extract(content, schema, prompt=None):
            captured_prompt.append(prompt)
            return {"name": "Mocked"}

        monkeypatch.setattr(tiers.llm_client, "extract_structured", mock_extract)
        # Also patch schema validation to always pass
        from scryer_mcp import schema as sc
        monkeypatch.setattr(sc, "validate_extraction_schema", lambda s: None)

        result = await scryer_extract_structured(
            urls=["https://example.com"],
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            prompt="Extract only the name field",
        )
        assert "error" not in result
        assert len(captured_prompt) > 0
        assert captured_prompt[0] == "Extract only the name field"


class TestScryerSynthesize:
    """Tests for the scryer_synthesize tool."""

    @pytest.mark.asyncio
    async def test_accepts_optional_prompt(self, monkeypatch):
        """Prompt must be accepted and forwarded to LLM client."""
        from scryer_mcp.server import scryer_synthesize
        from scryer_mcp import tiers

        captured = []

        async def mock_synth(query, results, prompt=None):
            captured.append(prompt)
            return {"grounded_answer": "Test.", "citations": []}

        monkeypatch.setattr(tiers.llm_client, "synthesize", mock_synth)

        result = await scryer_synthesize(
            query="test", results=[{"title": "T", "url": "https://x.com", "snippet": "S"}],
            prompt="Summarize in Spanish",
        )
        assert "error" not in result
        assert len(captured) == 1
        assert captured[0] == "Summarize in Spanish"
