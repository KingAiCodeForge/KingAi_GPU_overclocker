"""
NVML wrapper — clean Python interface to NVIDIA GPU monitoring.

Uses nvidia-ml-py (pynvml), the official NVIDIA Python binding for
the NVIDIA Management Library (NVML). NVML is a C library that ships
with every NVIDIA driver and exposes read-only GPU sensor data.

Key design decisions:
  - Single GpuSnapshot dataclass captures ALL readable sensor values
    in one atomic read — no stale partial state.
  - Every sensor read is wrapped in _safe() which catches NVMLError
    and returns a default value. This prevents one unsupported sensor
    from crashing the entire snapshot.
  - Works on both Windows and Linux. No admin/root required.
  - No overclocking here — that's nvapi.py (Windows-only, requires NVAPI).

Provides:
  - snapshot(index)    → GpuSnapshot (one GPU, one moment in time)
  - snapshot_all()     → list[GpuSnapshot] (all GPUs)
  - poll(index, interval) → generator yielding snapshots forever
  - gpu_count()        → int (number of NVIDIA GPUs)
"""

from __future__ import annotations

import atexit
import time
from dataclasses import dataclass, field


import warnings as _warnings

_warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)

import pynvml as nvml  # noqa: E402

# ── Globals ──
# NVML must be initialized once before any calls, and shut down on exit.
# We track this with _initialized and register an atexit handler for cleanup.

_initialized = False


def _ensure_init() -> None:
    """Lazily initialize NVML on first use. Thread-safe enough for our purposes."""
    global _initialized
    if not _initialized:
        nvml.nvmlInit()  # Must be called before ANY nvml function
        _initialized = True
        atexit.register(_shutdown)  # Clean up on process exit


def _shutdown() -> None:
    """Clean up NVML on exit. Silently ignores errors (process may be dying)."""
    global _initialized
    if _initialized:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass  # Don't crash during shutdown
        _initialized = False


# ── Throttle reason flags ──
# NVML reports WHY the GPU is throttling as a bitmask. Multiple reasons
# can be active simultaneously (e.g., both thermal AND power throttling).
# Understanding these is critical for OC tuning — they tell you what
# limit you're hitting and what to adjust.

THROTTLE_REASONS = {
    0x0000_0001: "GPU_IDLE",            # GPU has no work — clocks dropped to save power
    0x0000_0002: "APP_CLOCK_SETTING",   # Application requested lower clocks
    0x0000_0004: "SW_POWER_CAP",        # Hit software power limit (adjustable via OC)
    0x0000_0008: "HW_SLOWDOWN",         # Hardware protection — voltage/temp emergency
    0x0000_0010: "SYNC_BOOST",          # Multi-GPU sync limiting clocks
    0x0000_0020: "SW_THERMAL",          # Hit software thermal limit (adjustable via OC)
    0x0000_0040: "HW_THERMAL",          # Hit hardware thermal limit (NOT adjustable)
    0x0000_0080: "HW_POWER_BRAKE",      # External power brake signal (server boards)
    0x0000_0100: "DISPLAY_CLOCK",       # Display clock setting is limiting GPU clock
}


def decode_throttle_reasons(bitmask: int) -> list[str]:
    """Decode throttle reason bitmask into human-readable strings."""
    if bitmask == 0:
        return ["NONE"]
    reasons = []
    for flag, name in THROTTLE_REASONS.items():
        if bitmask & flag:
            reasons.append(name)
    return reasons or [f"UNKNOWN(0x{bitmask:08x})"]


# ── GPU snapshot dataclass ──
# This is the core data model for monitoring. Every field has a safe default
# so partial snapshots (where some sensors fail) are still usable.

@dataclass
class GpuSnapshot:
    """Point-in-time snapshot of all readable GPU sensors.

    Designed for:
      - CLI dashboard display (render_dashboard in monitor.py)
      - JSON/CSV export for logging and automation
      - GUI real-time updates (future PySide6 frontend)
      - Memtest monitoring (temp/clock during stability tests)

    All values are in human-friendly units:
      - Clocks in MHz, temperatures in °C, power in W, memory in MB
      - Percentages are 0-100 integers (fan, utilization)
    """

    # Identity — static info that doesn't change between snapshots
    index: int = 0
    name: str = ""
    driver_version: str = ""
    pci_bus_id: str = ""
    uuid: str = ""

    # Clocks (MHz) — current and maximum boost frequencies
    clock_gpu: int = 0          # Current core/graphics clock
    clock_mem: int = 0          # Current memory clock
    clock_sm: int = 0           # Streaming multiprocessor clock (usually = core)
    clock_video: int = 0        # Video encoder/decoder clock
    clock_gpu_max: int = 0      # Max boost clock (from BIOS)
    clock_mem_max: int = 0      # Max memory clock (from BIOS)

    # Temperature (°C)
    temp_gpu: int = 0           # Current GPU die temperature
    temp_gpu_max: int = 0       # Max temp threshold (from BIOS, typically 93-100°C)

    # Fan (%) — 0 = fan stopped (0 RPM or passive cooling)
    fan_speed: int = 0

    # Power (W) — NVML returns milliwatts, we convert to watts
    power_draw: float = 0.0     # Current power consumption
    power_limit: float = 0.0    # Current power target (may be OC'd above default)
    power_default: float = 0.0  # Factory default power limit
    power_min: float = 0.0      # Minimum allowed power limit
    power_max: float = 0.0      # Maximum allowed power limit

    # Memory (MB) — VRAM usage
    vram_total: int = 0
    vram_used: int = 0
    vram_free: int = 0

    # Utilization (%) — percentage of time GPU/memory bus is active
    util_gpu: int = 0
    util_mem: int = 0           # Memory controller utilization, NOT VRAM usage

    # State
    pstate: str = ""            # Performance state: P0=max, P8=idle, P12=minimum
    throttle_reasons: list[str] = field(default_factory=list)  # Human-readable
    throttle_raw: int = 0       # Raw bitmask for programmatic use

    # Timestamp — when this snapshot was taken (time.time())
    timestamp: float = 0.0

    @property
    def vram_used_pct(self) -> float:
        if self.vram_total == 0:
            return 0.0
        return 100.0 * self.vram_used / self.vram_total

    @property
    def power_pct(self) -> float:
        if self.power_limit == 0:
            return 0.0
        return 100.0 * self.power_draw / self.power_limit

    def summary_line(self) -> str:
        """One-line summary for dashboard display."""
        throttle = ",".join(self.throttle_reasons)
        return (
            f"{self.name} | "
            f"{self.temp_gpu}°C | "
            f"{self.clock_gpu}/{self.clock_gpu_max} MHz | "
            f"Mem {self.clock_mem} MHz | "
            f"{self.power_draw:.0f}/{self.power_limit:.0f}W ({self.power_pct:.0f}%) | "
            f"Fan {self.fan_speed}% | "
            f"GPU {self.util_gpu}% | "
            f"VRAM {self.vram_used}MB/{self.vram_total}MB ({self.vram_used_pct:.0f}%) | "
            f"{self.pstate} | "
            f"{throttle}"
        )


# ── Safe sensor readers ──
# NVML functions can throw NVMLError_NotSupported for sensors that don't
# exist on a particular GPU (e.g., no fan on passively cooled cards,
# no video clock on some Quadro models). Rather than checking capabilities
# upfront, we just try each read and fall back to a default. This is
# simpler and more robust across the wide range of NVIDIA GPUs.

def _safe(fn, *args, default=None):
    """Call an NVML function, returning default on any error.

    This is the core defensive pattern — every sensor read goes through here.
    """
    try:
        return fn(*args)
    except (nvml.NVMLError, Exception):
        return default


# ── Main API ─────────────────────────────────────────────────────────────────

def gpu_count() -> int:
    """Return number of NVIDIA GPUs."""
    _ensure_init()
    return nvml.nvmlDeviceGetCount()


def get_handle(index: int = 0):
    """Get NVML device handle."""
    _ensure_init()
    return nvml.nvmlDeviceGetHandleByIndex(index)


def snapshot(index: int = 0) -> GpuSnapshot:
    """Take a full sensor snapshot of a GPU.

    This is the workhorse function. It reads ~20 different sensor values
    in one call, producing an atomic snapshot. Each sensor is read via
    _safe() so failures in one don't affect others.

    Performance: ~1-2ms per call on modern GPUs (NVML is very fast).
    Safe to call at 1 Hz for dashboard, or 10 Hz for detailed logging.
    """
    _ensure_init()
    h = nvml.nvmlDeviceGetHandleByIndex(index)
    s = GpuSnapshot(index=index, timestamp=time.time())

    # Identity — these are static but we read them every time for simplicity.
    # Could be cached per-GPU for slight perf gain (not worth the complexity).
    s.name = _safe(nvml.nvmlDeviceGetName, h, default="Unknown")
    s.driver_version = _safe(nvml.nvmlSystemGetDriverVersion, default="?")
    # PCI bus ID may come back as bytes or str depending on pynvml version
    raw_bus = _safe(lambda hh: nvml.nvmlDeviceGetPciInfo(hh).busId, h, default="")
    s.pci_bus_id = raw_bus.decode("utf-8", errors="replace") if isinstance(raw_bus, bytes) else str(raw_bus)
    s.uuid = _safe(nvml.nvmlDeviceGetUUID, h, default="")

    # Clocks — current frequencies and max boost clocks
    s.clock_gpu = _safe(nvml.nvmlDeviceGetClockInfo, h, nvml.NVML_CLOCK_GRAPHICS, default=0)
    s.clock_mem = _safe(nvml.nvmlDeviceGetClockInfo, h, nvml.NVML_CLOCK_MEM, default=0)
    s.clock_sm = _safe(nvml.nvmlDeviceGetClockInfo, h, nvml.NVML_CLOCK_SM, default=0)
    s.clock_video = _safe(nvml.nvmlDeviceGetClockInfo, h, nvml.NVML_CLOCK_VIDEO, default=0)
    s.clock_gpu_max = _safe(nvml.nvmlDeviceGetMaxClockInfo, h, nvml.NVML_CLOCK_GRAPHICS, default=0)
    s.clock_mem_max = _safe(nvml.nvmlDeviceGetMaxClockInfo, h, nvml.NVML_CLOCK_MEM, default=0)

    # Temperature
    s.temp_gpu = _safe(nvml.nvmlDeviceGetTemperature, h, nvml.NVML_TEMPERATURE_GPU, default=0)
    s.temp_gpu_max = _safe(
        nvml.nvmlDeviceGetTemperatureThreshold,
        h,
        nvml.NVML_TEMPERATURE_THRESHOLD_GPU_MAX,
        default=0,
    )

    # Fan speed — 0-100%. Returns 0 for passively cooled cards.
    s.fan_speed = _safe(nvml.nvmlDeviceGetFanSpeed, h, default=0)

    # Power — NVML returns milliwatts, we want watts for display.
    # power_draw = actual current consumption
    # power_limit = current target (may have been raised by OC)
    # power_default = factory target
    # power_min/max = allowed range for set_power_limit()
    pw = _safe(nvml.nvmlDeviceGetPowerUsage, h, default=0)
    s.power_draw = pw / 1000.0 if pw else 0.0
    pl = _safe(nvml.nvmlDeviceGetPowerManagementLimit, h, default=0)
    s.power_limit = pl / 1000.0 if pl else 0.0
    pd = _safe(nvml.nvmlDeviceGetPowerManagementDefaultLimit, h, default=0)
    s.power_default = pd / 1000.0 if pd else 0.0
    pmin, pmax = 0, 0
    try:
        pmin = nvml.nvmlDeviceGetPowerManagementLimitConstraints(h)[0]
        pmax = nvml.nvmlDeviceGetPowerManagementLimitConstraints(h)[1]
    except Exception:
        pass
    s.power_min = pmin / 1000.0 if pmin else 0.0
    s.power_max = pmax / 1000.0 if pmax else 0.0

    # Memory — NVML returns bytes, we convert to MB for readability
    mem = _safe(nvml.nvmlDeviceGetMemoryInfo, h, default=None)
    if mem:
        s.vram_total = mem.total // (1024 * 1024)
        s.vram_used = mem.used // (1024 * 1024)
        s.vram_free = mem.free // (1024 * 1024)

    # Utilization — percentage of time the GPU/memory bus is busy.
    # NOTE: util_mem is memory CONTROLLER utilization, not VRAM usage percentage.
    # High util_mem with low VRAM usage means lots of small transfers.
    util = _safe(nvml.nvmlDeviceGetUtilizationRates, h, default=None)
    if util:
        s.util_gpu = util.gpu
        s.util_mem = util.memory

    # Performance state — P0=max perf (gaming/compute), P8=idle, P12=minimum power
    ps = _safe(nvml.nvmlDeviceGetPerformanceState, h, default=-1)
    s.pstate = f"P{ps}" if ps >= 0 else "?"

    # Throttle reasons — bitmask telling us WHY the GPU isn't running at max clock.
    # Critical for OC tuning: if SW_POWER_CAP is set, raise power limit.
    # If SW_THERMAL, raise thermal limit or increase fan speed.
    s.throttle_raw = _safe(nvml.nvmlDeviceGetCurrentClocksThrottleReasons, h, default=0)
    s.throttle_reasons = decode_throttle_reasons(s.throttle_raw)

    return s


def snapshot_all() -> list[GpuSnapshot]:
    """Snapshot all GPUs in the system."""
    return [snapshot(i) for i in range(gpu_count())]


def poll(index: int = 0, interval: float = 1.0):
    """Generator that yields GpuSnapshot at the given interval. Runs forever.

    Usage:
        for snap in poll(0, interval=1.0):
            print(snap.temp_gpu)

    Stop with Ctrl+C or break. Used by cli/monitor.py for continuous mode.
    """
    while True:
        yield snapshot(index)
        time.sleep(interval)
