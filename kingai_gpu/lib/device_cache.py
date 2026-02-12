"""
Per-GPU device cache — remembers probed struct layouts across sessions.

Problem: NVAPI struct layouts vary by GPU generation and driver version.
The code in nvapi.py has fallback probes (buffer scans, shift detection,
API fallback) that re-run every launch. They're fast (<50ms) but redundant
after the first successful run on a given GPU + driver combo.

Solution: After a successful get_oc_status(), save what we learned:
  - Power info: did primary offsets work, or which scanned offsets were found?
  - Thermal info: are the values <<8 shifted?
  - Fan API: did the new ClientFanCoolersSetControl work, or did we fall back?

Cache invalidation: keyed by (gpu_name, bus_id, driver_version).
If the driver updates, the cache auto-invalidates and re-probes.

File location: ~/.kingai_gpu/device_cache.json
The file is optional — if missing or corrupt, everything still works
(just re-probes as if no cache existed). This is a polish optimization,
not a correctness requirement.

Design principles:
  - Cache is NEVER trusted for correctness — it's a hint that skips probing
  - If a cached hint causes an NVAPI error, we fall back to full probing
  - The cache only stores layout metadata, never OC settings or sensor values
  - Cache file is human-readable JSON (useful for debugging driver quirks)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# Default cache location — in user's home directory, not the repo.
# This survives Python venv changes and works for installed packages.
_CACHE_DIR = Path.home() / ".kingai_gpu"
_CACHE_FILE = _CACHE_DIR / "device_cache.json"


@dataclass
class GpuCacheEntry:
    """Cached probe results for a single GPU + driver version.

    All fields are optional hints. None = "not yet probed" or "use default".
    The nvapi.py code should check these before running expensive probes,
    but must handle the case where a cached value is wrong (driver update
    that didn't change version string, GPU BIOS flash, etc).
    """

    # ── Identity (cache key) ──
    gpu_name: str = ""
    bus_id: int = 0
    driver_version: str = ""

    # ── Power info probing results ──
    # If True, the primary offsets (_PWR_INFO_MIN/DEF/MAX) worked directly.
    # If False, a buffer scan was needed — scanned_power_offsets has the results.
    power_primary_ok: bool | None = None
    scanned_power_offsets: list[int] | None = None  # [min_off, def_off, max_off]

    # ── Thermal info probing results ──
    # If True, raw thermal values needed >>8 shift.
    thermal_shifted: bool | None = None

    # ── Fan API preference ──
    # "new" = ClientFanCoolersSetControl works
    # "old" = only SetCoolerLevels works (new API failed or unavailable)
    fan_api: str | None = None  # "new" or "old"
    fan_entry_size: int | None = None  # entry size for new API (typically 68)
    fan_count: int | None = None  # number of fan entries

    # ── Metadata ──
    cached_at: str = ""  # ISO timestamp of when this was saved
    probe_time_ms: float = 0.0  # how long the full probe took (for diagnostics)


def _cache_key(gpu_name: str, bus_id: int, driver_version: str) -> str:
    """Build a unique cache key from GPU identity + driver version.

    Format: "GeForce RTX 4090 [Bus 1] @ 560.94"
    Driver version is part of the key so cache auto-invalidates on update.
    """
    return f"{gpu_name} [Bus {bus_id}] @ {driver_version}"


def load_cache() -> dict[str, GpuCacheEntry]:
    """Load the device cache from disk. Returns empty dict on any failure.

    This is deliberately forgiving — a corrupt or missing cache file
    just means we re-probe everything (same as first run).
    """
    if not _CACHE_FILE.exists():
        return {}
    try:
        raw = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        cache = {}
        for key, data in raw.items():
            if isinstance(data, dict):
                entry = GpuCacheEntry(
                    gpu_name=data.get("gpu_name", ""),
                    bus_id=data.get("bus_id", 0),
                    driver_version=data.get("driver_version", ""),
                    power_primary_ok=data.get("power_primary_ok"),
                    scanned_power_offsets=data.get("scanned_power_offsets"),
                    thermal_shifted=data.get("thermal_shifted"),
                    fan_api=data.get("fan_api"),
                    fan_entry_size=data.get("fan_entry_size"),
                    fan_count=data.get("fan_count"),
                    cached_at=data.get("cached_at", ""),
                    probe_time_ms=data.get("probe_time_ms", 0.0),
                )
                cache[key] = entry
        return cache
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def save_cache(cache: dict[str, GpuCacheEntry]) -> bool:
    """Save the device cache to disk. Returns False on failure (non-fatal).

    Creates ~/.kingai_gpu/ if it doesn't exist. Any write failure is
    silently ignored — the cache is optional and re-probing works fine.
    """
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        serializable = {key: asdict(entry) for key, entry in cache.items()}
        _CACHE_FILE.write_text(
            json.dumps(serializable, indent=2),
            encoding="utf-8",
        )
        return True
    except (OSError, TypeError):
        return False


def get_entry(
    gpu_name: str,
    bus_id: int,
    driver_version: str,
    cache: dict[str, GpuCacheEntry] | None = None,
) -> GpuCacheEntry | None:
    """Look up a cache entry for a specific GPU + driver combo.

    If cache is None, loads from disk first.
    Returns None on cache miss (unknown GPU or driver version changed).
    """
    if cache is None:
        cache = load_cache()
    key = _cache_key(gpu_name, bus_id, driver_version)
    return cache.get(key)


def put_entry(
    entry: GpuCacheEntry,
    cache: dict[str, GpuCacheEntry] | None = None,
) -> dict[str, GpuCacheEntry]:
    """Store a cache entry and persist to disk.

    If cache is None, loads existing cache first (preserves other GPU entries).
    Returns the updated cache dict.
    """
    if cache is None:
        cache = load_cache()
    key = _cache_key(entry.gpu_name, entry.bus_id, entry.driver_version)
    cache[key] = entry
    save_cache(cache)
    return cache


def clear_cache() -> bool:
    """Delete the cache file entirely. Used for troubleshooting."""
    try:
        if _CACHE_FILE.exists():
            _CACHE_FILE.unlink()
        return True
    except OSError:
        return False
