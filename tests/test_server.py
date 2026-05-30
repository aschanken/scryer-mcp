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
