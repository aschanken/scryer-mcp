"""Filesystem cache with lazy TTL enforcement — ported from cache.sh."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

DEFAULT_ROOT = Path(os.getenv("SCRYER_CACHE_DIR", Path.home() / ".cache" / "scryer-mcp"))


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _ns_dir(namespace: str) -> Path:
    d = DEFAULT_ROOT / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d


def get(namespace: str, key: str) -> str | None:
    """Get a cached value. Returns None on miss."""
    h = _hash(key)
    f = DEFAULT_ROOT / namespace / f"{h}.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
        return data["value"]
    except Exception:
        return None


def put(namespace: str, key: str, value: str) -> None:
    """Store a value in the cache."""
    d = _ns_dir(namespace)
    h = _hash(key)
    entry = {"value": value, "cached_at": int(time.time())}
    tmp = d / f"{h}.tmp"
    dst = d / f"{h}.json"
    tmp.write_text(json.dumps(entry))
    tmp.rename(dst)


def ttl(namespace: str, key: str, ttl_seconds: int) -> bool:
    """Check if a cached entry exists and is within TTL. Returns True if valid."""
    h = _hash(key)
    f = DEFAULT_ROOT / namespace / f"{h}.json"
    if not f.exists():
        return False
    try:
        data = json.loads(f.read_text())
        age = int(time.time()) - data["cached_at"]
        if age < ttl_seconds:
            return True
        f.unlink(missing_ok=True)
        return False
    except Exception:
        return False


def prune(namespace: str, ttl_seconds: int) -> int:
    """Remove expired entries. Returns count removed."""
    d = DEFAULT_ROOT / namespace
    if not d.exists():
        return 0
    now = int(time.time())
    removed = 0
    for f in d.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if now - data["cached_at"] >= ttl_seconds:
                f.unlink()
                removed += 1
        except Exception:
            pass
    return removed
