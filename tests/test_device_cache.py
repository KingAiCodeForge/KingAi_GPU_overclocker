"""Optional cache fixtures are task-owned; never read or replace a user cache."""

from kingai_gpu.lib import device_cache as cache


def test_cache_missing_corrupt_and_unicode_path(monkeypatch, tmp_path):
    directory = tmp_path / "cache with spaces \u03a9"
    target = directory / "device_cache.json"
    monkeypatch.setattr(cache, "_CACHE_DIR", directory)
    monkeypatch.setattr(cache, "_CACHE_FILE", target)
    assert cache.load_cache() == {}
    entry = cache.GpuCacheEntry(gpu_name="Synthetic GPU", bus_id=0, driver_version="test")
    saved = cache.put_entry(entry)
    assert cache.get_entry("Synthetic GPU", 0, "test") == entry
    other = cache.GpuCacheEntry(gpu_name="Another synthetic GPU", bus_id=1,
                               driver_version="test")
    cache.put_entry(other)
    assert set(cache.load_cache()) == set(saved) | {cache._cache_key(other.gpu_name, 1, "test")}
    target.write_text("invalid JSON", encoding="utf-8")
    assert cache.load_cache() == {}
