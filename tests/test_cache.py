"""Tests for cache.py."""
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scryer_mcp import cache


def test_put_and_get(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_ROOT", tmp_path / "cache")
    cache.put("test", "key1", '{"value": "hello"}')
    result = cache.get("test", "key1")
    assert result == '{"value": "hello"}'


def test_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_ROOT", tmp_path / "cache")
    result = cache.get("test", "nonexistent")
    assert result is None


def test_ttl_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_ROOT", tmp_path / "cache")
    cache.put("test", "key2", '{"value": "fresh"}')
    assert cache.ttl("test", "key2", 3600) is True


def test_ttl_expired(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_ROOT", tmp_path / "cache")
    cache.put("test", "key3", '{"value": "stale"}')
    # Write a cached_at in the past
    import hashlib
    h = hashlib.sha256("key3".encode()).hexdigest()
    f = tmp_path / "cache" / "test" / f"{h}.json"
    data = json.loads(f.read_text())
    data["cached_at"] = int(time.time()) - 10000
    f.write_text(json.dumps(data))
    assert cache.ttl("test", "key3", 60) is False


def test_expired_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_ROOT", tmp_path / "cache")
    cache.put("test", "key4", '{"value": "old"}')
    import hashlib
    h = hashlib.sha256("key4".encode()).hexdigest()
    f = tmp_path / "cache" / "test" / f"{h}.json"
    data = json.loads(f.read_text())
    data["cached_at"] = int(time.time()) - 10000
    f.write_text(json.dumps(data))
    # TTL check should delete the expired entry
    assert cache.ttl("test", "key4", 60) is False
    assert cache.get("test", "key4") is None


def test_prune(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_ROOT", tmp_path / "cache")
    cache.put("test", "key5", '{"value": "old"}')
    cache.put("test", "key6", '{"value": "new"}')
    import hashlib
    # Make key5 old
    h = hashlib.sha256("key5".encode()).hexdigest()
    f = tmp_path / "cache" / "test" / f"{h}.json"
    data = json.loads(f.read_text())
    data["cached_at"] = int(time.time()) - 10000
    f.write_text(json.dumps(data))
    removed = cache.prune("test", 3600)
    assert removed >= 1
    assert cache.get("test", "key5") is None
    assert cache.get("test", "key6") == '{"value": "new"}'
