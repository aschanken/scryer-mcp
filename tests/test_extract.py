"""Tests for extract.py — structured content discovery and merge extraction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


# ---- HTML fixtures for structured content discovery ----

JSON_LD_HTML = """<!DOCTYPE html>
<html><head>
<script type="application/ld+json">
{"@context": "https://schema.org", "@type": "Person", "name": "Ada Lovelace", "birthDate": "1815-12-10"}
</script>
</head><body><p>First computer programmer.</p></body></html>"""

JSON_LD_ARRAY_HTML = """<!DOCTYPE html>
<html><head>
<script type="application/ld+json">
[
  {"@context": "https://schema.org", "@type": "Person", "name": "Ada Lovelace"},
  {"@context": "https://schema.org", "@type": "Person", "name": "Charles Babbage"}
]
</script>
</head><body></body></html>"""

OG_HTML = """<!DOCTYPE html>
<html><head>
<meta property="og:title" content="Test Page">
<meta property="og:description" content="A test page for OG discovery.">
<meta property="og:image" content="https://example.com/image.jpg">
</head><body><p>Hello</p></body></html>"""

MICRODATA_HTML = """<!DOCTYPE html>
<html><body>
<div itemscope itemtype="https://schema.org/Product">
  <span itemprop="name">Widget</span>
  <span itemprop="price">$9.99</span>
</div>
</body></html>"""

TABLE_HTML = """<!DOCTYPE html>
<html><body>
<table summary="Pricing">
  <caption>Plan Pricing</caption>
  <tr><th>Plan</th><th>Price</th></tr>
  <tr><td>Basic</td><td>$10</td></tr>
  <tr><td>Pro</td><td>$20</td></tr>
</table>
</body></html>"""

ALL_IN_ONE_HTML = """<!DOCTYPE html>
<html><head>
<script type="application/ld+json">{"@type": "Organization", "name": "Corp"}</script>
<meta property="og:title" content="All In One">
<meta property="og:type" content="website">
</head><body>
<div itemscope itemtype="https://schema.org/Product">
  <span itemprop="name">Gadget</span>
</div>
<table>
  <tr><th>Feature</th><th>Value</th></tr>
  <tr><td>Weight</td><td>1kg</td></tr>
</table>
</body></html>"""

EMPTY_HTML = "<html><body></body></html>"

INVALID_JSON_LD_HTML = """<!DOCTYPE html>
<html><head>
<script type="application/ld+json">not valid json</script>
</head><body></body></html>"""

NO_STRUCTURED_HTML = """<!DOCTYPE html>
<html><head><title>Plain page</title></head>
<body><p>Just some text with no structured metadata.</p></body></html>"""


class TestDiscoverStructuredContent:
    """Tests for discover_structured_content()."""

    def test_json_ld_single(self):
        from scryer_mcp.extract import discover_structured_content

        items = discover_structured_content(JSON_LD_HTML)
        json_ld = [i for i in items if i["type"] == "json_ld"]
        assert len(json_ld) == 1
        assert json_ld[0]["name"] == "Ada Lovelace"
        assert json_ld[0]["data"]["@type"] == "Person"

    def test_json_ld_array(self):
        from scryer_mcp.extract import discover_structured_content

        items = discover_structured_content(JSON_LD_ARRAY_HTML)
        json_ld = [i for i in items if i["type"] == "json_ld"]
        assert len(json_ld) == 2
        names = {i["name"] for i in json_ld}
        assert names == {"Ada Lovelace", "Charles Babbage"}

    def test_open_graph(self):
        from scryer_mcp.extract import discover_structured_content

        items = discover_structured_content(OG_HTML)
        og_items = [i for i in items if i["type"] == "open_graph"]
        assert len(og_items) == 1
        og = og_items[0]
        assert og["name"] == "Test Page"
        assert og["data"]["title"] == "Test Page"
        assert og["data"]["description"] == "A test page for OG discovery."

    def test_microdata(self):
        from scryer_mcp.extract import discover_structured_content

        items = discover_structured_content(MICRODATA_HTML)
        md_items = [i for i in items if i["type"] == "microdata"]
        assert len(md_items) == 1
        md = md_items[0]
        assert md["name"] == "Product"
        assert md["data"]["properties"]["name"] == "Widget"

    def test_table(self):
        from scryer_mcp.extract import discover_structured_content

        items = discover_structured_content(TABLE_HTML)
        table_items = [i for i in items if i["type"] == "table"]
        assert len(table_items) == 1
        t = table_items[0]
        assert t["name"] == "Plan Pricing"  # from <caption>
        assert t["data"]["headers"] == ["Plan", "Price"]
        assert len(t["data"]["rows"]) == 2
        assert t["data"]["rows"][0] == {"Plan": "Basic", "Price": "$10"}

    def test_table_with_summary_fallback(self):
        from scryer_mcp.extract import discover_structured_content

        html = """<html><body>
        <table summary="Stats"><tr><th>X</th><th>Y</th></tr><tr><td>1</td><td>2</td></tr></table>
        </body></html>"""
        items = discover_structured_content(html)
        table_items = [i for i in items if i["type"] == "table"]
        assert len(table_items) == 1
        assert table_items[0]["name"] == "Stats"

    def test_all_four_types_in_one_page(self):
        from scryer_mcp.extract import discover_structured_content

        items = discover_structured_content(ALL_IN_ONE_HTML)
        types = {i["type"] for i in items}
        assert types == {"json_ld", "open_graph", "microdata", "table"}

    def test_empty_html_returns_empty(self):
        from scryer_mcp.extract import discover_structured_content

        assert discover_structured_content(EMPTY_HTML) == []

    def test_invalid_json_ld_skipped_gracefully(self):
        from scryer_mcp.extract import discover_structured_content

        items = discover_structured_content(INVALID_JSON_LD_HTML)
        json_ld = [i for i in items if i["type"] == "json_ld"]
        assert json_ld == []

    def test_no_structured_content(self):
        from scryer_mcp.extract import discover_structured_content

        items = discover_structured_content(NO_STRUCTURED_HTML)
        assert items == []

    def test_none_html_returns_empty(self):
        from scryer_mcp.extract import discover_structured_content

        assert discover_structured_content("") == []

    def test_short_html_returns_empty(self):
        from scryer_mcp.extract import discover_structured_content

        assert discover_structured_content("<p>Hi</p>") == []


class TestDiscoverStructuredContentIntegration:
    """Verify discovery works end-to-end through the fetch layer."""

    @pytest.mark.asyncio
    async def test_fetch_includes_structured_items(self, monkeypatch):
        """When fetch_url processes HTML with JSON-LD, structured_items must appear."""
        from scryer_mcp import fetch

        html = JSON_LD_HTML
        original_fetch = fetch.fetch_url

        async def mock_fetch(url, client, mode="highlights"):
            result = await original_fetch(url, client, mode)
            # Override the fetched content with our test fixture
            result["structured_items"] = [
                {"type": "json_ld", "name": "Ada Lovelace", "data": {"@type": "Person", "name": "Ada Lovelace"}}
            ]
            return result

        monkeypatch.setattr(fetch, "fetch_url", mock_fetch)
        results = await fetch.fetch_urls(["https://example.com"], "highlights", 5000)
        assert len(results) == 1
        assert "structured_items" in results[0]
        assert len(results[0]["structured_items"]) == 1
        assert results[0]["structured_items"][0]["type"] == "json_ld"
