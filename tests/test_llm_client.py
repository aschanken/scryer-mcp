"""Tests for llm_client.py — dual Anthropic / OpenAI-compatible backends."""
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
    def _clear_keys(self, monkeypatch):
        """Ensure both API keys are unset."""
        monkeypatch.delenv("SCRYER_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("SCRYER_ANTHROPIC_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_unavailable(self, mock_llm_unavailable):
        client = LLMClient()
        result = await client.synthesize(
            "query", [{"title": "T", "url": "x", "snippet": "s"}]
        )
        assert "error" in result
        assert "unavailable" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_classify_passthrough_on_unavailable(self, mock_llm_unavailable):
        client = LLMClient()
        results = [{"url": "https://x.com"}, {"url": "https://y.com"}]
        urls = await client.classify_category(results, "news")
        assert urls == ["https://x.com", "https://y.com"]


class TestLLMClientAuth:
    """Tests for API key / token behavior (OpenAI-compat path)."""

    def test_api_key_from_kwarg(self):
        client = LLMClient(api_key="sk-test-key")
        assert client.openai_api_key == "sk-test-key"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SCRYER_DEEPSEEK_API_KEY", "sk-env-key")
        client = LLMClient()
        assert client.openai_api_key == "sk-env-key"

    def test_api_key_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("SCRYER_DEEPSEEK_API_KEY", raising=False)
        client = LLMClient()
        assert client.openai_api_key is None

    def test_kwarg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SCRYER_DEEPSEEK_API_KEY", "sk-env-key")
        client = LLMClient(api_key="sk-explicit")
        assert client.openai_api_key == "sk-explicit"

    @pytest.mark.asyncio
    async def test_get_client_no_auth_header_by_default(self, monkeypatch):
        """Shared client no longer carries auth headers — added per-request."""
        monkeypatch.delenv("SCRYER_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("SCRYER_ANTHROPIC_API_KEY", raising=False)
        client = LLMClient()
        c = await client._get_client()
        assert "Authorization" not in c.headers
        assert "x-api-key" not in c.headers


class TestLLMClientBackendDetection:
    """Tests for backend auto-detection from model name prefix."""

    def test_claude_model_uses_anthropic(self):
        client = LLMClient(model="claude-3-5-haiku-latest")
        assert client._primary_backend == "anthropic"

    def test_claude_opus_uses_anthropic(self):
        client = LLMClient(model="claude-opus-4-8")
        assert client._primary_backend == "anthropic"

    def test_deepseek_model_uses_openai(self):
        client = LLMClient(model="deepseek-chat")
        assert client._primary_backend == "openai"

    def test_deepseek_v4_uses_openai(self):
        client = LLMClient(model="deepseek-v4-flash")
        assert client._primary_backend == "openai"

    def test_gpt_model_uses_openai(self):
        client = LLMClient(model="gpt-4o")
        assert client._primary_backend == "openai"

    def test_unknown_model_defaults_to_anthropic(self):
        client = LLMClient(model="some-unknown-model")
        assert client._primary_backend == "anthropic"

    def test_default_model_is_claude_haiku(self, monkeypatch):
        monkeypatch.delenv("SCRYER_LLM_MODEL", raising=False)
        client = LLMClient()
        assert client.model == "claude-3-5-haiku-latest"
        assert client._primary_backend == "anthropic"


class TestLLMClientDualBackend:
    """Tests for dual-backend config and fallback."""

    @pytest.fixture(autouse=True)
    def _clear_keys(self, monkeypatch):
        monkeypatch.delenv("SCRYER_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("SCRYER_ANTHROPIC_API_KEY", raising=False)

    def test_anthropic_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SCRYER_ANTHROPIC_API_KEY", "sk-ant-test")
        client = LLMClient()
        assert client.anthropic_api_key == "sk-ant-test"

    def test_anthropic_api_key_from_kwarg(self):
        client = LLMClient(anthropic_api_key="sk-ant-kwarg")
        assert client.anthropic_api_key == "sk-ant-kwarg"

    def test_both_keys_independent(self, monkeypatch):
        monkeypatch.setenv("SCRYER_DEEPSEEK_API_KEY", "sk-ds")
        monkeypatch.setenv("SCRYER_ANTHROPIC_API_KEY", "sk-ant")
        client = LLMClient()
        assert client.openai_api_key == "sk-ds"
        assert client.anthropic_api_key == "sk-ant"

    @pytest.mark.asyncio
    async def test_chat_anthropic_returns_none_without_key(self):
        client = LLMClient(model="claude-3-5-haiku-latest")
        result = await client._chat_anthropic(
            "sys", "user", 0.3, 100, "claude-3-5-haiku-latest"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_chat_openai_returns_none_without_key(self):
        client = LLMClient(model="deepseek-chat")
        result = await client._chat_openai(
            "sys", "user", 0.3, 100, "deepseek-chat"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_when_primary_unavailable(self, monkeypatch):
        """When primary (Anthropic) has no key, fall back to OpenAI."""
        monkeypatch.setenv("SCRYER_DEEPSEEK_API_KEY", "sk-ds")
        client = LLMClient(model="claude-3-5-haiku-latest")

        call_count = 0

        async def mock_chat_openai(self_inst, sys, usr, temp, mt, model):
            nonlocal call_count
            call_count += 1
            assert model == "deepseek-v4-flash"  # fallback model
            return "fallback response"

        monkeypatch.setattr(LLMClient, "_chat_openai", mock_chat_openai)

        result = await client._chat("sys", "user")
        assert result == "fallback response"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_no_fallback_when_primary_succeeds(self, monkeypatch):
        """When primary (Anthropic) succeeds, never call fallback."""
        monkeypatch.setenv("SCRYER_ANTHROPIC_API_KEY", "sk-ant")
        client = LLMClient(model="claude-3-5-haiku-latest")

        async def mock_chat_anthropic(self_inst, sys, usr, temp, mt, model):
            return "anthropic response"

        monkeypatch.setattr(LLMClient, "_chat_anthropic", mock_chat_anthropic)

        fallback_called = False

        async def mock_chat_openai(self_inst, *args, **kwargs):
            nonlocal fallback_called
            fallback_called = True
            return "should not be called"

        monkeypatch.setattr(LLMClient, "_chat_openai", mock_chat_openai)

        result = await client._chat("sys", "user")
        assert result == "anthropic response"
        assert not fallback_called

    @pytest.mark.asyncio
    async def test_check_all_backends_reports_both(self, monkeypatch):
        monkeypatch.setenv("SCRYER_DEEPSEEK_API_KEY", "sk-ds")
        monkeypatch.setenv("SCRYER_ANTHROPIC_API_KEY", "sk-ant")
        client = LLMClient()

        status = await client.check_all_backends()
        assert "anthropic" in status
        assert "openai" in status
        assert status["anthropic"]["api_key_configured"] is True
        assert status["openai"]["api_key_configured"] is True


class TestExtractJSONExtended:
    """Tests for _extract_json with arrays and embedded braces."""

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
    """Tests for citation hallucination validation."""

    @pytest.mark.asyncio
    async def test_hallucinated_citations_filtered(self, monkeypatch):
        async def mock_available(self_inst):
            return True

        monkeypatch.setattr(LLMClient, "is_available", mock_available)

        async def mock_chat(self_inst, system, user, temperature=0.3, max_tokens=2000):
            return (
                '{"grounded_answer": "Some answer.", '
                '"citations": ["https://example.com/1", "https://madeup.com/fake"]}'
            )

        monkeypatch.setattr(LLMClient, "_chat", mock_chat)

        client = LLMClient()
        results = [
            {"url": "https://example.com/1", "title": "Real Result", "snippet": "Real"}
        ]
        result = await client.synthesize("test query", results)
        assert "https://example.com/1" in result["citations"]
        assert "https://madeup.com/fake" not in result["citations"]
        assert result.get("_hallucinated_citations") == 1

    @pytest.mark.asyncio
    async def test_all_valid_citations_preserved(self, monkeypatch):
        async def mock_available(self_inst):
            return True

        monkeypatch.setattr(LLMClient, "is_available", mock_available)

        async def mock_chat(self_inst, system, user, temperature=0.3, max_tokens=2000):
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
        async def mock_available(self_inst):
            return True

        monkeypatch.setattr(LLMClient, "is_available", mock_available)

        async def mock_chat(self_inst, system, user, temperature=0.3, max_tokens=2000):
            return '{"grounded_answer": "No citations."}'

        monkeypatch.setattr(LLMClient, "_chat", mock_chat)

        client = LLMClient()
        result = await client.synthesize("query", [])
        assert result.get("_hallucinated_citations", 0) == 0


class TestExtractStructuredValidation:
    """Tests for schema validation in extract_structured."""

    @pytest.mark.asyncio
    async def test_schema_validation_rejects_bad_output(self, monkeypatch):
        async def mock_available(self_inst):
            return True

        monkeypatch.setattr(LLMClient, "is_available", mock_available)

        async def mock_chat(self_inst, system, user, temperature=0.0, max_tokens=2000):
            return '{"age": "not-a-number"}'

        monkeypatch.setattr(LLMClient, "_chat", mock_chat)

        client = LLMClient()
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"],
        }
        result = await client.extract_structured("Some content", schema)
        assert "error" in result
        assert "schema" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_schema_validation_accepts_good_output(self, monkeypatch):
        async def mock_available(self_inst):
            return True

        monkeypatch.setattr(LLMClient, "is_available", mock_available)

        async def mock_chat(self_inst, system, user, temperature=0.0, max_tokens=2000):
            return '{"name": "Alice", "age": 30}'

        monkeypatch.setattr(LLMClient, "_chat", mock_chat)

        client = LLMClient()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = await client.extract_structured("Content about Alice", schema)
        assert "error" not in result
        assert result["name"] == "Alice"
        assert result["age"] == 30


class TestLLMClientCleartextWarning:
    """Tests for cleartext API key warnings."""

    def test_warning_on_http_non_localhost(self, monkeypatch):
        import warnings

        monkeypatch.delenv("SCRYER_DEEPSEEK_API_KEY", raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMClient(
                endpoint="http://api.example.com/v1/chat/completions",
                api_key="sk-test",
            )
            assert any("cleartext" in str(x.message).lower() for x in w)

    def test_no_warning_on_https(self, monkeypatch):
        import warnings

        monkeypatch.delenv("SCRYER_DEEPSEEK_API_KEY", raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMClient(
                endpoint="https://api.example.com/v1/chat/completions",
                api_key="sk-test",
            )
            assert not any("cleartext" in str(x.message).lower() for x in w)

    def test_no_warning_on_localhost_http(self, monkeypatch):
        import warnings

        monkeypatch.delenv("SCRYER_DEEPSEEK_API_KEY", raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMClient(
                endpoint="http://localhost:8734/v1/chat/completions",
                api_key="sk-test",
            )
            assert not any("cleartext" in str(x.message).lower() for x in w)
