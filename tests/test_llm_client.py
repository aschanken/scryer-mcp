"""Tests for llm_client.py."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scryer_mcp.llm_client import LLMClient, _extract_json, _format_results


class TestExtractJSON:
    def test_simple_object(self):
        text = 'prefix {"key": "value"} suffix'
        assert _extract_json(text) == '{"key": "value"}'

    def test_nested_object(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        assert _extract_json(text) == text

    def test_no_braces(self):
        text = "no json here"
        assert _extract_json(text) == text

    def test_unclosed_brace(self):
        text = '{"key": "value"'
        assert _extract_json(text) == text


class TestFormatResults:
    def test_basic(self):
        results = [
            {"title": "T", "url": "https://x.com", "snippet": "snippet text"},
            {"title": "T2", "url": "https://y.com", "highlights": "hl text"},
        ]
        formatted = _format_results(results)
        assert "T" in formatted
        assert "snippet text" in formatted
        assert "hl text" in formatted


class TestLLMClient:
    @pytest.fixture(autouse=True)
    def _clear_api_key(self, monkeypatch):
        """Ensure SCRYER_API_KEY is unset for every test in this class."""
        monkeypatch.delenv("SCRYER_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_unavailable(self, mock_llm_unavailable):
        client = LLMClient()
        result = await client.synthesize("query", [{"title": "T", "url": "x", "snippet": "s"}])
        assert "error" in result
        assert "unavailable" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_classify_passthrough_on_unavailable(self, mock_llm_unavailable):
        client = LLMClient()
        results = [{"url": "https://x.com"}, {"url": "https://y.com"}]
        urls = await client.classify_category(results, "news")
        assert urls == ["https://x.com", "https://y.com"]  # all pass through


class TestLLMClientAuth:
    """Tests for API key / Bearer token behavior."""

    def test_api_key_from_kwarg(self):
        client = LLMClient(api_key="sk-test-key")
        assert client.api_key == "sk-test-key"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SCRYER_API_KEY", "sk-env-key")
        client = LLMClient()
        assert client.api_key == "sk-env-key"

    def test_api_key_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("SCRYER_API_KEY", raising=False)
        client = LLMClient()
        assert client.api_key is None

    def test_kwarg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SCRYER_API_KEY", "sk-env-key")
        client = LLMClient(api_key="sk-explicit")
        assert client.api_key == "sk-explicit"

    @pytest.mark.asyncio
    async def test_get_client_adds_auth_header(self, monkeypatch):
        monkeypatch.setenv("SCRYER_API_KEY", "sk-test")
        client = LLMClient()
        c = await client._get_client()
        assert c.headers.get("Authorization") == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_get_client_no_auth_header_when_key_unset(self, monkeypatch):
        monkeypatch.delenv("SCRYER_API_KEY", raising=False)
        client = LLMClient()
        c = await client._get_client()
        assert "Authorization" not in c.headers


class TestExtractJSONExtended:
    """Tests for _extract_json with arrays and embedded braces (C7)."""

    def test_embedded_braces_in_string(self):
        text = '{"key": "text with { and } braces", "nested": {"inner": "value"}}'
        assert _extract_json(text) == text

    def test_json_array(self):
        text = 'prefix ["query1", "query2 with [bracket]"] suffix'
        assert _extract_json(text) == '["query1", "query2 with [bracket]"]'

    def test_finds_array_when_first(self):
        text = '["a", "b"] and more {"key": "val"}'
        assert _extract_json(text) == '["a", "b"]'

    def test_finds_object_when_first(self):
        text = '{"key": "val"} and more ["a", "b"]'
        assert _extract_json(text) == '{"key": "val"}'


class TestSynthesizeCitationFiltering:
    """Tests for citation hallucination validation (C3)."""

    @pytest.mark.asyncio
    async def test_hallucinated_citations_filtered(self, monkeypatch):
        from scryer_mcp.llm_client import LLMClient

        async def mock_available(self):
            return True
        monkeypatch.setattr(LLMClient, "is_available", mock_available)

        # Note: _chat is patched on the class, so mock must accept self
        async def mock_chat(self, system, user, temperature=0.3, max_tokens=2000):
            return (
                '{"grounded_answer": "Some answer.", '
                '"citations": ["https://example.com/1", "https://madeup.com/fake"]}'
            )
        monkeypatch.setattr(LLMClient, "_chat", mock_chat)

        client = LLMClient()
        results = [{"url": "https://example.com/1", "title": "Real Result", "snippet": "Real"}]
        result = await client.synthesize("test query", results)
        assert "https://example.com/1" in result["citations"]
        assert "https://madeup.com/fake" not in result["citations"]
        assert result.get("_hallucinated_citations") == 1

    @pytest.mark.asyncio
    async def test_all_valid_citations_preserved(self, monkeypatch):
        from scryer_mcp.llm_client import LLMClient

        async def mock_available(self):
            return True
        monkeypatch.setattr(LLMClient, "is_available", mock_available)

        async def mock_chat(self, system, user, temperature=0.3, max_tokens=2000):
            return (
                '{"grounded_answer": "Valid answer.", '
                '"citations": ["https://example.com/1", "https://example.com/2"]}'
            )
        monkeypatch.setattr(LLMClient, "_chat", mock_chat)

        client = LLMClient()
        results = [
            {"url": "https://example.com/1", "title": "R1", "snippet": "S1"},
            {"url": "https://example.com/2", "title": "R2", "snippet": "S2"},
        ]
        result = await client.synthesize("query", results)
        assert len(result["citations"]) == 2
        assert result.get("_hallucinated_citations") == 0

    @pytest.mark.asyncio
    async def test_no_citations_handled(self, monkeypatch):
        from scryer_mcp.llm_client import LLMClient

        async def mock_available(self):
            return True
        monkeypatch.setattr(LLMClient, "is_available", mock_available)

        async def mock_chat(self, system, user, temperature=0.3, max_tokens=2000):
            return '{"grounded_answer": "No citations."}'
        monkeypatch.setattr(LLMClient, "_chat", mock_chat)

        client = LLMClient()
        result = await client.synthesize("query", [])
        assert result.get("_hallucinated_citations", 0) == 0


class TestExtractStructuredValidation:
    """Tests for schema validation in extract_structured (C4)."""

    @pytest.mark.asyncio
    async def test_schema_validation_rejects_bad_output(self, monkeypatch):
        from scryer_mcp.llm_client import LLMClient

        async def mock_available(self):
            return True
        monkeypatch.setattr(LLMClient, "is_available", mock_available)

        async def mock_chat(self, system, user, temperature=0.0, max_tokens=2000):
            return '{"age": "not-a-number"}'
        monkeypatch.setattr(LLMClient, "_chat", mock_chat)

        client = LLMClient()
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}
        result = await client.extract_structured("Some content", schema)
        assert "error" in result
        assert "schema" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_schema_validation_accepts_good_output(self, monkeypatch):
        from scryer_mcp.llm_client import LLMClient

        async def mock_available(self):
            return True
        monkeypatch.setattr(LLMClient, "is_available", mock_available)

        async def mock_chat(self, system, user, temperature=0.0, max_tokens=2000):
            return '{"name": "Alice", "age": 30}'
        monkeypatch.setattr(LLMClient, "_chat", mock_chat)

        client = LLMClient()
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}
        result = await client.extract_structured("Content about Alice", schema)
        assert "error" not in result
        assert result["name"] == "Alice"
        assert result["age"] == 30


class TestLLMClientCleartextWarning:
    """Tests for cleartext API key warnings (C8)."""

    def test_warning_on_http_non_localhost(self, monkeypatch):
        import warnings
        from scryer_mcp.llm_client import LLMClient

        monkeypatch.delenv("SCRYER_API_KEY", raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMClient(endpoint="http://api.example.com/v1/chat/completions", api_key="sk-test")
            assert any("cleartext" in str(x.message).lower() for x in w)

    def test_no_warning_on_https(self, monkeypatch):
        import warnings
        from scryer_mcp.llm_client import LLMClient

        monkeypatch.delenv("SCRYER_API_KEY", raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMClient(endpoint="https://api.example.com/v1/chat/completions", api_key="sk-test")
            assert not any("cleartext" in str(x.message).lower() for x in w)

    def test_no_warning_on_localhost_http(self, monkeypatch):
        import warnings
        from scryer_mcp.llm_client import LLMClient

        monkeypatch.delenv("SCRYER_API_KEY", raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMClient(endpoint="http://localhost:8734/v1/chat/completions", api_key="sk-test")
            assert not any("cleartext" in str(x.message).lower() for x in w)
