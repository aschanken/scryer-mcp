"""extract.py — Structured output extraction with merge/dedup.

Map-Reduce pattern for extracting structured data from web pages.
- Map: Per-page extraction via subagent (invoked by SKILL.md instructions)
- Reduce: Merge per-page dicts with key-wise deduplication (this file)
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

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
