"""extract.py — Structured output extraction with merge/dedup.

Map-Reduce pattern for extracting structured data from web pages.
- Map: Per-page extraction via subagent (invoked by SKILL.md instructions)
- Reduce: Merge per-page dicts with key-wise deduplication (this file)

Also provides automatic structured-content discovery from raw HTML:
JSON-LD, Open Graph, microdata, and HTML tables.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Optional

from bs4 import BeautifulSoup

from .schema import validate_extraction_schema

def merge_extractions(
    extractions: list[dict[str, Any]],
    schema: Optional[dict] = None,
) -> dict[str, Any]:
    """Merge per-page extraction dicts via key-wise set union.

    For each key, collects all non-None values across all extractions
    and deduplicates. Array values are concatenated then deduplicated.
    Scalar values are collected into a list and deduplicated.

    Args:
        extractions: List of per-page extraction dicts.
        schema: Optional JSON Schema (unused in merge, passed through).

    Returns:
        Merged dict with deduplicated values per key.
    """
    if not extractions:
        return {}

    # Collect all keys
    all_keys: set[str] = set()
    for ext in extractions:
        if isinstance(ext, dict):
            all_keys.update(ext.keys())

    merged: dict[str, Any] = {}

    for key in sorted(all_keys):
        values: list[Any] = []
        for ext in extractions:
            if isinstance(ext, dict) and key in ext:
                val = ext[key]
                if val is not None:
                    if isinstance(val, list):
                        values.extend(val)
                    else:
                        values.append(val)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[Any] = []
        for v in values:
            # Use JSON serialization for hashable dedup key
            v_key = json.dumps(v, sort_keys=True, default=str)
            if v_key not in seen:
                seen.add(v_key)
                deduped.append(v)

        # Unwrap single-element lists for scalar fields
        merged[key] = deduped if len(deduped) != 1 else deduped[0]

    return merged


# ------------------------------------------------------------------
# Structured content discovery from raw HTML
# ------------------------------------------------------------------

_OG_PROPERTY = re.compile(r"^og:(.+)$", re.IGNORECASE)


def discover_structured_content(html: str) -> list[dict]:
    """Scan raw HTML for embedded structured content.

    Automatically discovers:
      - **JSON-LD** (``<script type="application/ld+json">``) — parsed as dicts
      - **Open Graph** (``<meta property="og:*">``) — collected as ``{key: val}``
      - **Microdata** (``itemscope`` / ``itemprop``) — shallow tree walk
      - **Tables** (``<table>`` with at least one ``<th>``) — list of row lists

    Args:
        html: Raw HTML content of a web page.

    Returns:
        List of structured item dicts, each with ``type``, ``name``, and ``data`` keys.
        Empty list if nothing is found.
    """
    items: list[dict] = []

    if not html or len(html.strip()) < 50:
        return items

    # Cap HTML size to prevent unbounded CPU on malicious/large pages
    max_html_chars = 262_144  # 256 KB
    if len(html) > max_html_chars:
        html = html[:max_html_chars]

    soup = BeautifulSoup(html, "lxml")

    # ---- 1. JSON-LD ----
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.get_text(strip=True)
        if not raw:
            continue
        # A single <script> may hold a list or a single object
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        entries = parsed if isinstance(parsed, list) else [parsed]
        for entry in entries:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("headline") or None
                items.append({
                    "type": "json_ld",
                    "name": name,
                    "data": entry,
                })

    # ---- 2. Open Graph ----
    og_data: dict[str, str] = {}
    for tag in soup.find_all("meta", property=_OG_PROPERTY):
        prop = tag.get("property", "")
        content = tag.get("content", "")
        m = _OG_PROPERTY.match(prop)
        if m and content:
            key = m.group(1)
            og_data[key] = content.strip()

    if og_data:
        # Derive a human-readable name from og:title
        og_name = og_data.get("title", "Open Graph")
        items.append({
            "type": "open_graph",
            "name": og_name,
            "data": dict(og_data),
        })

    # ---- 3. Microdata ----
    microdata_items = _extract_microdata(soup)
    for md in microdata_items:
        items.append({
            "type": "microdata",
            "name": md.get("itemtype", "").split("/")[-1] or None,
            "data": md,
        })

    # ---- 4. Well-formed HTML tables ----
    for i, table in enumerate(soup.find_all("table")):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        headers = [th.get_text(strip=True) for th in rows[0].find_all("th") if th.get_text(strip=True)]
        if not headers:
            continue
        table_data: list[dict[str, str | None]] = []
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            row_dict: dict[str, str | None] = {}
            for j, cell in enumerate(cells):
                key = headers[j] if j < len(headers) else f"column_{j}"
                row_dict[key] = cell.get_text(strip=True) or None
            table_data.append(row_dict)
        if table_data:
            caption_text = table.caption.get_text(strip=True) if table.caption else None
            name = caption_text or table.get("summary") or f"Table {i + 1}"
            items.append({
                "type": "table",
                "name": name,
                "data": {
                    "headers": headers,
                    "rows": table_data,
                },
            })

    return items


def _extract_microdata(soup: BeautifulSoup) -> list[dict]:
    """Walk top-level ``itemscope`` elements and collect ``itemprop`` values.

    Returns a list of dicts, one per top-level itemscope.
    """
    items: list[dict] = []
    for scope in soup.find_all(itemscope=True):
        itemtype = scope.get("itemtype", "")
        props: dict[str, Any] = {}
        for prop in scope.find_all(itemprop=True):
            # Only direct children (skip nested scopes)
            if prop.find_parents(itemscope=True, limit=2) != [scope]:
                continue
            key = prop.get("itemprop", "")
            if key in props:
                if not isinstance(props[key], list):
                    props[key] = [props[key]]
                props[key].append(_microdata_value(prop))
            else:
                props[key] = _microdata_value(prop)
        if itemtype:
            items.append({"itemtype": itemtype, "properties": props})
    return items


def _microdata_value(tag) -> str | None:
    """Extract the itemprop value from a tag (content attr, src, href, or text)."""
    for attr in ("content", "src", "href"):
        val = tag.get(attr)
        if val:
            return str(val).strip()
    return tag.get_text(strip=True) or None


def main() -> None:
    """CLI: python3 extract.py merge <file1.json> [file2.json] ...

    Reads per-page extraction JSON files, merges them, prints merged JSON.
    """
    if len(sys.argv) < 2:
        print("Usage: extract.py merge <file1.json> [file2.json ...]", file=sys.stderr)
        print("       extract.py validate <schema.json>", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "merge":
        files = sys.argv[2:]
        extractions: list[dict[str, Any]] = []
        errors = 0
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        extractions.append(data)
                    elif isinstance(data, list):
                        extractions.extend(data)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: skipping {f}: {e}", file=sys.stderr)
                errors += 1

        merged = merge_extractions(extractions)
        output = {
            "items": [merged] if merged else [],
            "merged_from": len(extractions),
            "files_skipped": errors,
        }
        json.dump(output, sys.stdout, indent=2)

    elif command == "validate":
        if len(sys.argv) != 3:
            print("Usage: extract.py validate <schema.json>", file=sys.stderr)
            sys.exit(1)
        with open(sys.argv[2]) as f:
            schema = json.load(f)
        err = validate_extraction_schema(schema)
        if err:
            print(f"INVALID: {err}")
            sys.exit(1)
        print("VALID")
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
