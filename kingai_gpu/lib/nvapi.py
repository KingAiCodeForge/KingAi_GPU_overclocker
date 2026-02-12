"""
NVAPI ctypes wrapper — GPU overclocking control for NVIDIA GPUs (Windows x64).

Uses undocumented NvAPI functions via nvapi_QueryInterface for:
  - Core/memory clock offset (SetPstates20)
  - Power limit (ClientPowerPoliciesSetStatus)
  - Thermal limit (ClientThermalPoliciesSetLimit)
  - Fan speed control (SetCoolerLevels / ClientFanCoolersSetControl)

Struct sizes validated by binary probing on real hardware.
Reference: Demion/nvapioc (C++ OC tool), falahati/NvAPIWrapper (C# wrapper)
"""

from __future__ import annotations

import ctypes
import struct
import sys
import time
from dataclasses import dataclass

# NVAPI is Windows-only — it's a Windows DLL that talks to the NVIDIA kernel
# driver (nvlddmkm.sys). On Linux, use nvidia-smi or NVML directly.
if sys.platform != "win32":
    raise ImportError("NVAPI requires Windows")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Error handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# NVAPI uses integer return codes. 0 = success, negative = error.
# These codes are from the public nvapi.h header.
NVAPI_OK = 0

NVAPI_ERRORS = {
    0:  "OK",
    -1: "ERROR",
    -2: "LIBRARY_NOT_FOUND",
    -3: "NO_IMPLEMENTATION",
    -4: "API_NOT_INITIALIZED",
    -5: "INVALID_ARGUMENT",
    -6: "NVIDIA_DEVICE_NOT_FOUND",
    -7: "END_ENUMERATION",
    -8: "INVALID_HANDLE",
    -9: "INCOMPATIBLE_STRUCT_VERSION",
    -14: "NOT_SUPPORTED",
    -40: "INSUFFICIENT_BUFFER",
    -104: "INVALID_USER_PRIVILEGE",
}


class NvApiError(Exception):
    """NVAPI call failed."""

    def __init__(self, func_name: str, status: int):
        name = NVAPI_ERRORS.get(status, f"UNKNOWN({status})")
        super().__init__(f"{func_name} returned {status} ({name})")
        self.status = status
        self.func_name = func_name


def _check(func_name: str, status: int) -> int:
    if status != NVAPI_OK:
        raise NvApiError(func_name, status)
    return status


# Status codes that are transient — another app (Afterburner, NVIDIA tuning
# panel, EVGA Precision) holds the OC lock briefly. The lock releases when
# their call completes, typically <100ms. Retrying fixes it.
_TRANSIENT_ERRORS = {-104, -9}  # INVALID_USER_PRIVILEGE, INCOMPATIBLE_STRUCT_VERSION (driver busy)


def _check_set(func_name: str, fn, *args, max_retries: int = 3) -> int:
    """Call an NVAPI Set function with retry on transient lock errors.

    Only used for write operations (SetPstates20, SetStatus, SetLimit,
    SetCoolerLevels, SetControl). Get operations should NOT retry — if a
    Get fails, the struct layout is wrong and retrying won't help.

    Retry schedule: 100ms, 200ms, 300ms. Total worst-case: 600ms.
    If it still fails after 3 tries, the lock is held persistently
    (user should close Afterburner).
    """
    last_status = 0
    for attempt in range(max_retries):
        status = fn(*args)
        if status == NVAPI_OK:
            return status
        last_status = status
        if status in _TRANSIENT_ERRORS and attempt < max_retries - 1:
            time.sleep(0.1 * (attempt + 1))  # 100ms, 200ms, 300ms
            continue
        # Non-transient error or last attempt — fail immediately
        raise NvApiError(func_name, status)
    raise NvApiError(func_name, last_status)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load nvapi64.dll + QueryInterface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Load the DLL. nvapi64.dll lives in System32 and is installed with every
# NVIDIA driver. If this fails, either no NVIDIA GPU/driver is present,
# or Python is 32-bit (nvapi64 = 64-bit only; nvapi.dll = 32-bit).
try:
    _nvapi = ctypes.CDLL("nvapi64")
except OSError:
    raise ImportError(
        "Cannot load nvapi64.dll — NVIDIA driver not installed or not 64-bit Python"
    )

# nvapi_QueryInterface is the ONLY exported symbol in the DLL.
# Everything else is accessed by passing a uint32 function ID hash.
# It returns a raw function pointer (void*) that we cast to the right signature.
_query_interface = _nvapi.nvapi_QueryInterface
_query_interface.restype = ctypes.c_void_p
_query_interface.argtypes = [ctypes.c_uint]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QueryInterface IDs  (from Demion/nvapioc + NvAPIWrapper)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Function ID hashes for nvapi_QueryInterface.
# These are NOT random — they're computed from the function name by NVIDIA's
# internal hash algorithm. The same hash works across all driver versions.
# Sources: Demion/nvapioc main.cpp, falahati/NvAPIWrapper, community RE work.
_QI = {
    # Core lifecycle — must call Initialize before anything else
    "Initialize":                   0x0150E828,
    "Unload":                       0xD22BDD7E,
    "EnumPhysicalGPUs":             0xE5AC921F,

    # GPU info
    "GPU_GetFullName":              0xCEEE8E9F,
    "GPU_GetBusId":                 0x1BE0B8E5,

    # PStates20 — clock/voltage offsets
    "GPU_GetPstates20":             0x6FF81213,
    "GPU_SetPstates20":             0x0F4DAE6B,

    # Power policies
    "GPU_ClientPowerPoliciesGetInfo":   0x34206D86,
    "GPU_ClientPowerPoliciesGetStatus": 0x70916171,
    "GPU_ClientPowerPoliciesSetStatus": 0xAD95F5ED,

    # Thermal policies
    "GPU_ClientThermalPoliciesGetInfo":  0x0D258BB5,
    "GPU_ClientThermalPoliciesGetLimit": 0xE9C425A1,
    "GPU_ClientThermalPoliciesSetLimit": 0x34C0B13D,

    # Fan coolers (new API)
    "GPU_ClientFanCoolersGetStatus":    0x35AED5E8,
    "GPU_ClientFanCoolersGetControl":   0x814B209F,
    "GPU_ClientFanCoolersSetControl":   0xA58971A5,

    # Fan coolers (old API — simpler, 3 args)
    "GPU_SetCoolerLevels":              0x891FA0AE,

    # VF curve — advanced: per-voltage-point frequency control
    # Not yet used in public API, reserved for Phase 2 VF curve editor
    "GPU_GetClockBoostMask":        0x507B4B59,
    "GPU_GetVFPCurve":              0x21537AD4,
    "GPU_GetClockBoostTable":       0x23F1B133,
    "GPU_SetClockBoostTable":       0x0733E009,
    "GPU_GetClockBoostLock":        0xE440B867,
    "GPU_SetClockBoostLock":        0x39442CFB,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Function pointer resolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# NVAPI functions use C calling conventions. We define 3 signatures:
#
# _FN_VOID — no args (Initialize only)
#   fn() → int status
#
# _FN_2PTR — most OC functions (Get/Set with struct)
#   fn(void* gpuHandle, void* structPtr) → int status
#   Also used for EnumPhysicalGPUs(void** handles, uint* count)
#   and GetFullName(void* handle, char* nameBuf) — same 2-pointer layout
#
# _FN_3ARG — old fan API only
#   fn(void* gpuHandle, uint coolerIndex, void* structPtr) → int status
#   SetCoolerLevels takes an extra uint for which cooler to target

_FN_VOID = ctypes.CFUNCTYPE(ctypes.c_int)
_FN_2PTR = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
_FN_3ARG = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p)


def _resolve(name: str, sig=_FN_2PTR):
    """Resolve an NVAPI function via QueryInterface. Returns callable or None.

    Returns None instead of raising if the function isn't available.
    This lets us gracefully degrade on older drivers that don't expose
    newer functions (e.g., ClientFanCoolersSetControl wasn't always present).
    """
    qid = _QI.get(name)
    if qid is None:
        return None
    ptr = _query_interface(qid)
    if not ptr:
        return None
    return sig(ptr)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Resolved function pointers (populated on first use)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# name -> ctypes callable. Lazily populated by _init() on first use.
# We resolve all function pointers once at startup rather than per-call
# because QueryInterface has some overhead and the pointer never changes.
_fn = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAKE_NVAPI_VERSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_version(struct_size: int, version: int) -> int:
    """Reproduce the MAKE_NVAPI_VERSION(struct, ver) C macro.

    Every NVAPI struct starts with a uint32 version tag:
      bits 0-15:  struct size in bytes (so NVAPI can validate buffer length)
      bits 16-31: version number (so NVAPI can handle struct layout changes)

    If this tag is wrong, the call returns -9 (INCOMPATIBLE_STRUCT_VERSION).
    Getting the right size+version is the hardest part of reverse engineering
    undocumented NVAPI functions — we binary-probed these on real hardware.
    """
    return struct_size | (version << 16)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Struct sizes (binary probed — see ignore/pyside6_qt_pyqt.md for docs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── PStates20 ──
# The main struct for reading/writing clock offsets.
# V1 (7316 bytes) is used for Get — contains current offsets and allowed ranges.
# V2 (7416 bytes, 100 bytes larger) is used for Set — adds overVoltage fields.
# The extra 100 bytes = space for voltage override data at the end of the struct.
_PSTATES20_V1_SIZE = 7316     # Get: read current clocks (no overVoltage)
_PSTATES20_V2_SIZE = 7416     # Set: write clock offsets (with overVoltage)
_CLOCK_ENTRY_SIZE = 44         # Each clock entry: domain, type, flags, delta, freq
_VOLTAGE_ENTRY_SIZE = 24       # Each voltage entry: domain, editable, value, delta range
_MAX_CLOCKS = 8                # Max clock domains per pstate (core, mem, shader, video, etc)
_MAX_VOLTAGES = 4              # Max voltage domains per pstate
_MAX_PSTATES = 16              # P0-P15 (P0=max perf, P8=idle, P12=minimum)
# Each pstate = 8 (header) + 8×44 (clocks) + 4×24 (voltages) = 456 bytes
_PSTATE_SIZE = 8 + _MAX_CLOCKS * _CLOCK_ENTRY_SIZE + _MAX_VOLTAGES * _VOLTAGE_ENTRY_SIZE

# ── Power policies ──
# Info = static limits (min/default/max). Status = current target.
_POWER_INFO_SIZE = 184         # 4 entries × 44 bytes each + 8 header
_POWER_STATUS_SIZE = 72        # 4 entries × 16 bytes each + 8 header

# ── Thermal policies ──
# Info = temp range. Limit = current thermal throttle point.
_THERMAL_INFO_SIZE = 88        # 4 entries × 20 bytes each + 8 header
_THERMAL_LIMIT_SIZE = 40       # V2 struct — temperature is stored <<8 (shifted left 8 bits)

# ── Fan coolers ──
# New API (ClientFanCoolersGetStatus/GetControl/SetControl) — detailed per-fan data.
_FAN_STATUS_SIZE = 1704        # Read-only: RPM, speed%, min/max for each fan
_FAN_CONTROL_SIZE = 1452       # Read/write: target speed, manual/auto mode per fan

# ── Old fan API (SetCoolerLevels) ──
# Simpler 3-arg call. Works on all GPUs, even old ones. Used as fallback.
_COOLER_LEVELS_SIZE = 4 + 3 * 8   # version(4) + coolers[3] × (level(4) + policy(4)) = 28 bytes

# ── Voltage boost ──
# Not yet used — reserved for VF curve feature
_VOLTAGE_BOOST_SIZE = 40


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Buffer helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _buf(size: int, version: int) -> ctypes.Array:
    """Create a zero-filled buffer with version tag at offset 0.

    Every NVAPI struct must start with the version tag. We zero-fill
    the entire buffer first because NVAPI checks for garbage data
    in unused fields on some functions (particularly Set operations).
    """
    b = (ctypes.c_ubyte * size)()
    struct.pack_into("<I", b, 0, _make_version(size, version))
    return b


# These helpers read/write little-endian integers from raw byte buffers.
# We use struct.pack/unpack instead of ctypes.Structure because the
# undocumented structs have layouts that vary by driver version.
# Raw buffers + offset math is more reliable than rigid Structure defs.

def _u32(buf, offset: int) -> int:
    """Read unsigned 32-bit little-endian integer from buffer."""
    return struct.unpack_from("<I", buf, offset)[0]


def _i32(buf, offset: int) -> int:
    """Read signed 32-bit little-endian integer from buffer."""
    return struct.unpack_from("<i", buf, offset)[0]


def _w32(buf, offset: int, value: int):
    """Write unsigned 32-bit LE to buffer. Masks to 32 bits."""
    struct.pack_into("<I", buf, offset, value & 0xFFFFFFFF)


def _wi32(buf, offset: int, value: int):
    """Write signed 32-bit LE to buffer. Used for clock offsets (can be negative)."""
    struct.pack_into("<i", buf, offset, value)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PStates20 field offset helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# PStates20 struct header layout:
#   +0:  version          (uint32, MAKE_NVAPI_VERSION tag)
#   +4:  flags            (uint32, usually 0)
#   +8:  numPStates       (uint32, how many pstates have data, typically 5)
#   +12: numClocks        (uint32, clock domains per pstate, typically 2: core+mem)
#   +16: numBaseVoltages  (uint32, voltage domains per pstate)
#   +20: pStates[0] starts here
_PS_VERSION = 0
_PS_FLAGS = 4
_PS_NUM_PSTATES = 8
_PS_NUM_CLOCKS = 12
_PS_NUM_BASE_VOLTAGES = 16
_PS_PSTATES_START = 20         # First pstate entry begins at byte 20


def _pstate_off(i: int) -> int:
    """Byte offset of pStates[i]."""
    return _PS_PSTATES_START + i * _PSTATE_SIZE


def _clock_off(pstate: int, clock: int) -> int:
    """Byte offset of pStates[pstate].clocks[clock]."""
    return _pstate_off(pstate) + 8 + clock * _CLOCK_ENTRY_SIZE


def _volt_off(pstate: int, volt: int) -> int:
    """Byte offset of pStates[pstate].baseVoltages[volt]."""
    return _pstate_off(pstate) + 8 + _MAX_CLOCKS * _CLOCK_ENTRY_SIZE + volt * _VOLTAGE_ENTRY_SIZE


# Clock entry fields (offsets relative to the start of each clock entry).
# Each clock entry is 44 bytes within a pstate.
_CK_DOMAIN = 0       # uint32: clock domain ID (0=Graphics/Core, 4=Memory)
_CK_TYPE = 4         # uint32: 0=single frequency, 1=frequency range
_CK_FLAGS = 8        # uint32: bit 0 = editable (can apply offsets)
_CK_DELTA_VAL = 12   # int32:  current offset in kHz (e.g., +150000 = +150 MHz)
_CK_DELTA_MIN = 16   # int32:  minimum allowed offset in kHz (e.g., -1000000)
_CK_DELTA_MAX = 20   # int32:  maximum allowed offset in kHz (e.g., +1000000)
_CK_DATA_FREQ = 24   # uint32: base frequency in kHz (or range min for type=1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU handle storage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# GPU handle array — EnumPhysicalGPUs fills this with opaque handles.
# Each handle is a void* that identifies a physical GPU to all other functions.
# We support up to 64 GPUs (NVAPI's documented maximum).
_MAX_GPUS = 64
_gpu_handles = (ctypes.c_void_p * _MAX_GPUS)()  # Filled by _init()
_gpu_count = ctypes.c_uint(0)                    # Set by _init()
_initialized = False                              # Ensures _init() runs only once


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Initialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _init():
    """Initialize NVAPI and resolve all function pointers.

    Called lazily on first use (via _handle()). After this:
      - NvAPI_Initialize has been called (required before any other call)
      - All function pointers are resolved from QueryInterface
      - GPU handles are enumerated and stored in _gpu_handles

    This is idempotent — safe to call multiple times.
    """
    global _initialized

    if _initialized:
        return

    # Step 1: Call NvAPI_Initialize() — mandatory before any other NVAPI call.
    # This sets up internal state inside the DLL.
    fn_init = _resolve("Initialize", _FN_VOID)
    if fn_init is None:
        raise NvApiError("Initialize", -3)
    _check("Initialize", fn_init())

    # Step 2: Resolve all function pointers we'll use.
    # Each _resolve() call goes through QueryInterface and returns a callable.
    # None is returned if the function isn't available (old driver, etc).
    _fn["EnumPhysicalGPUs"] = _resolve("EnumPhysicalGPUs", _FN_2PTR)
    _fn["GPU_GetFullName"] = _resolve("GPU_GetFullName", _FN_2PTR)
    _fn["GPU_GetBusId"] = _resolve("GPU_GetBusId", _FN_2PTR)

    _fn["GPU_GetPstates20"] = _resolve("GPU_GetPstates20", _FN_2PTR)
    _fn["GPU_SetPstates20"] = _resolve("GPU_SetPstates20", _FN_2PTR)

    _fn["GPU_ClientPowerPoliciesGetInfo"] = _resolve("GPU_ClientPowerPoliciesGetInfo", _FN_2PTR)
    _fn["GPU_ClientPowerPoliciesGetStatus"] = _resolve("GPU_ClientPowerPoliciesGetStatus", _FN_2PTR)
    _fn["GPU_ClientPowerPoliciesSetStatus"] = _resolve("GPU_ClientPowerPoliciesSetStatus", _FN_2PTR)

    _fn["GPU_ClientThermalPoliciesGetInfo"] = _resolve("GPU_ClientThermalPoliciesGetInfo", _FN_2PTR)
    _fn["GPU_ClientThermalPoliciesGetLimit"] = _resolve("GPU_ClientThermalPoliciesGetLimit", _FN_2PTR)
    _fn["GPU_ClientThermalPoliciesSetLimit"] = _resolve("GPU_ClientThermalPoliciesSetLimit", _FN_2PTR)

    # Old fan API uses 3-arg signature (handle, coolerIndex, struct)
    _fn["GPU_SetCoolerLevels"] = _resolve("GPU_SetCoolerLevels", _FN_3ARG)

    _fn["GPU_ClientFanCoolersGetControl"] = _resolve("GPU_ClientFanCoolersGetControl", _FN_2PTR)
    _fn["GPU_ClientFanCoolersSetControl"] = _resolve("GPU_ClientFanCoolersSetControl", _FN_2PTR)

    # Step 3: Enumerate all physical GPUs in the system.
    # This fills _gpu_handles[] and sets _gpu_count.
    if _fn["EnumPhysicalGPUs"]:
        _check(
            "EnumPhysicalGPUs",
            _fn["EnumPhysicalGPUs"](
                ctypes.cast(ctypes.pointer(_gpu_handles), ctypes.c_void_p),
                ctypes.byref(_gpu_count),
            ),
        )

    _initialized = True


def _handle(gpu: int = 0) -> ctypes.c_void_p:
    """Get GPU handle by index, initializing NVAPI if this is the first call.

    All NVAPI functions require an opaque GPU handle. This is the main
    entry point that triggers lazy initialization.
    """
    _init()
    if gpu >= _gpu_count.value:
        raise NvApiError("GPU_GetHandle", -5)  # -5 = INVALID_ARGUMENT
    return _gpu_handles[gpu]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU info helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_gpu_name(gpu: int = 0) -> str:
    """Get GPU full name string."""
    h = _handle(gpu)
    name_buf = (ctypes.c_char * 64)()
    fn = _fn.get("GPU_GetFullName")
    if fn:
        try:
            _check("GPU_GetFullName", fn(h, ctypes.cast(name_buf, ctypes.c_void_p)))
            return name_buf.value.decode("utf-8", errors="replace")
        except NvApiError:
            pass
    return "Unknown GPU"


def _get_bus_id(gpu: int = 0) -> int:
    """Get GPU PCI bus ID."""
    h = _handle(gpu)
    bus_id = ctypes.c_uint(0)
    fn = _fn.get("GPU_GetBusId")
    if fn:
        try:
            _check("GPU_GetBusId", fn(h, ctypes.byref(bus_id)))
            return bus_id.value
        except NvApiError:
            pass
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PStates20 — Read clock offsets and ranges
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_pstates20(gpu: int = 0):
    """Read PStates20 V1 (7316 bytes). Returns raw buffer.

    This is the most important struct — contains all clock offset info.
    P0 (performance state 0) = max performance, which is where we read/write
    core and memory clock offsets from.
    """
    h = _handle(gpu)
    b = _buf(_PSTATES20_V1_SIZE, 1)  # V1 for reading
    fn = _fn.get("GPU_GetPstates20")
    if fn is None:
        raise NvApiError("GPU_GetPstates20", -3)
    _check("GPU_GetPstates20", fn(h, ctypes.cast(b, ctypes.c_void_p)))
    return b


def _find_clock_in_pstates(buf, domain_id: int, pstate_idx: int = 0) -> int | None:
    """
    Find the clock entry index for a given domain in pState[pstate_idx].

    domain_id: 0=Graphics/Core, 4=Memory
    Returns clock index (0-7) or None.
    """
    num_clocks = _u32(buf, _PS_NUM_CLOCKS)
    for ci in range(min(num_clocks, _MAX_CLOCKS)):
        base = _clock_off(pstate_idx, ci)
        if _u32(buf, base + _CK_DOMAIN) == domain_id:
            return ci
    return None


def _read_clock_delta(buf, domain_id: int) -> tuple[int, int, int]:
    """
    Read clock delta for domain from PStates20 buffer.

    Returns (current_khz, min_khz, max_khz).
    Searches all pstates for the first match.
    """
    num_pstates = _u32(buf, _PS_NUM_PSTATES)
    for pi in range(min(num_pstates, _MAX_PSTATES)):
        ci = _find_clock_in_pstates(buf, domain_id, pi)
        if ci is not None:
            base = _clock_off(pi, ci)
            val = _i32(buf, base + _CK_DELTA_VAL)
            lo = _i32(buf, base + _CK_DELTA_MIN)
            hi = _i32(buf, base + _CK_DELTA_MAX)
            return (val, lo, hi)
    return (0, 0, 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PStates20 — Write clock offsets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _set_clock_offset(domain_id: int, offset_khz: int, gpu: int = 0):
    """
    Set clock offset via SetPstates20 V2.

    domain_id: 0=Graphics/Core, 4=Memory
    offset_khz: offset in kHz (e.g., +150000 for +150 MHz)

    Pattern from Demion/nvapioc:
      pStatesInfo.version = MAKE_NVAPI_VERSION(pStatesInfo, 2);
      pStatesInfo.numPStates = 1;
      pStatesInfo.numClocks = 1;
      pStatesInfo.pStates[0].pStateId = 0;  // P0
      pStatesInfo.pStates[0].clocks[0].domainId = domain;
      pStatesInfo.pStates[0].clocks[0].typeId = 0;
      pStatesInfo.pStates[0].clocks[0].frequencyDeltaKHz.value = offset;
    """
    h = _handle(gpu)
    b = _buf(_PSTATES20_V2_SIZE, 2)

    # Header
    _w32(b, _PS_NUM_PSTATES, 1)
    _w32(b, _PS_NUM_CLOCKS, 1)

    # pStates[0].pStateId = 0 (P0)
    _w32(b, _pstate_off(0), 0)

    # pStates[0].clocks[0]
    ck = _clock_off(0, 0)
    _w32(b, ck + _CK_DOMAIN, domain_id)
    _w32(b, ck + _CK_TYPE, 0)        # single frequency type
    _wi32(b, ck + _CK_DELTA_VAL, offset_khz)

    fn = _fn.get("GPU_SetPstates20")
    if fn is None:
        raise NvApiError("GPU_SetPstates20", -3)
    _check_set("GPU_SetPstates20", fn, h, ctypes.cast(b, ctypes.c_void_p))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Power policies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Power Status layout (72 bytes):
#   +0:  version (uint32, MAKE_NVAPI_VERSION tag)
#   +4:  count   (uint32, number of power entries, usually 1)
#   +8:  entry[0].unknown1  (uint32, must be 0 for Set calls)
#   +12: entry[0].power     (uint32, in PCM units: 100000 = 100.0%)
#   +16: entry[1]...
#   Each entry is 16 bytes, max 4 entries.
#
# PCM = "per cent mille" = power × 1000. So 125% = 125000 PCM.
_PWR_STATUS_COUNT = 4      # offset of the 'count' field
_PWR_STATUS_POWER = 12     # offset of entry[0].power (the value we care about)

# Power Info layout (184 bytes) — read-only, provides allowed power range:
#   +0:  version (uint32)
#   +4:  count   (uint32, usually 1)
#   +8:  entries[4] × 44 bytes each
#   Entry fields: pstateId(4), unk(4), unk(4), minPower(4), unk(4),
#                 defPower(4), unk(4), maxPower(4), ...
#
# These absolute offsets point into entry[0] (the main power domain):
_PWR_INFO_ENTRY_START = 8
_PWR_INFO_ENTRY_SIZE = 44
_PWR_INFO_MIN = 20        # entry[0].minPower — e.g., 50000 PCM (50%)
_PWR_INFO_DEF = 28        # entry[0].defPower — e.g., 100000 PCM (100%)
_PWR_INFO_MAX = 36        # entry[0].maxPower — e.g., 116000 PCM (116%)


def _get_power_status(gpu: int = 0) -> int:
    """Read current power target in PCM (100000 = 100%). Returns PCM value."""
    h = _handle(gpu)
    b = _buf(_POWER_STATUS_SIZE, 1)
    fn = _fn.get("GPU_ClientPowerPoliciesGetStatus")
    if fn is None:
        raise NvApiError("GPU_ClientPowerPoliciesGetStatus", -3)
    _check("GPU_ClientPowerPoliciesGetStatus", fn(h, ctypes.cast(b, ctypes.c_void_p)))
    return _u32(b, _PWR_STATUS_POWER)


def _get_power_info(gpu: int = 0) -> tuple[int, int, int]:
    """
    Read power limits from GetInfo struct.

    Returns (min_pcm, default_pcm, max_pcm) in PCM units.
    Falls back to scanning buffer for recognizable PCM values if primary offsets fail.
    """
    h = _handle(gpu)
    b = _buf(_POWER_INFO_SIZE, 1)
    fn = _fn.get("GPU_ClientPowerPoliciesGetInfo")
    if fn is None:
        raise NvApiError("GPU_ClientPowerPoliciesGetInfo", -3)
    _check("GPU_ClientPowerPoliciesGetInfo", fn(h, ctypes.cast(b, ctypes.c_void_p)))

    # Try primary offsets
    min_p = _u32(b, _PWR_INFO_MIN)
    def_p = _u32(b, _PWR_INFO_DEF)
    max_p = _u32(b, _PWR_INFO_MAX)

    # Validate: default power should be in sane range (30%-200%).
    # Some driver versions or GPU models put the values at different offsets.
    if 30000 <= def_p <= 200000:
        return (min_p, def_p, max_p)

    # GRACEFUL DEGRADATION: If primary offsets miss, scan the entire buffer
    # for uint32 values that look like PCM power values. This handles
    # struct layout shifts across different driver versions.
    pcm_candidates = []
    for off in range(4, len(b) - 3, 4):
        val = _u32(b, off)
        if 30000 <= val <= 200000:
            pcm_candidates.append((off, val))

    # Look for 100000 (100.0% = factory default). Min is typically before it,
    # max after it in the buffer. This heuristic works across all known layouts.
    for i, (off, val) in enumerate(pcm_candidates):
        if val == 100000:  # default power = 100.000%
            # min is likely before, max after
            min_p = pcm_candidates[i - 1][1] if i > 0 else 50000
            max_p = pcm_candidates[i + 1][1] if i < len(pcm_candidates) - 1 else 116000
            return (min_p, 100000, max_p)

    # Last resort: hardcoded safe defaults (50% min, 100% default, 116% max).
    # These are conservative and work for most GeForce cards.
    return (50000, 100000, 116000)


def _set_power_status(power_pcm: int, gpu: int = 0):
    """
    Set power limit.

    power_pcm: target in PCM (100000 = 100%, 125000 = 125%)

    From nvapioc:
      powerStatus.count = 1;
      powerStatus.entries[0].power = power * 1000;
    """
    h = _handle(gpu)
    b = _buf(_POWER_STATUS_SIZE, 1)
    _w32(b, _PWR_STATUS_COUNT, 1)    # count = 1
    _w32(b, _PWR_STATUS_POWER, power_pcm)

    fn = _fn.get("GPU_ClientPowerPoliciesSetStatus")
    if fn is None:
        raise NvApiError("GPU_ClientPowerPoliciesSetStatus", -3)
    _check_set("GPU_ClientPowerPoliciesSetStatus", fn, h, ctypes.cast(b, ctypes.c_void_p))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thermal policies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Thermal Limit layout (40 bytes, V2):
#   +0:  version    (uint32, MAKE_NVAPI_VERSION(40, 2))
#   +4:  count      (uint32, usually 1)
#   +8:  entry[0].controller  (uint32, 1 = GPU thermal controller)
#   +12: entry[0].value       (uint32, tempC << 8)   ← IMPORTANT: LEFT-SHIFTED by 8!
#   +16: entry[0].flags       (uint32, 1 = priority over driver defaults)
#   +20: entry[0].unknown     (uint32)
#   +24: entry[1]...          (second entry, 16 bytes)
#
# The <<8 shift is a quirk of V2 — the raw value at offset +12 is (tempC × 256).
# To read:  temp = raw >> 8
# To write: raw = temp << 8
_THERMAL_LIMIT_COUNT = 4
_THERMAL_LIMIT_CONTROLLER = 8
_THERMAL_LIMIT_VALUE = 12
_THERMAL_LIMIT_FLAGS = 16

# Thermal Info layout (88 bytes, V1):
#   +0: version    (uint32)
#   +4: count      (uint32)
#   +8: entries[4] × 20 bytes each
#   Entry: controller(4), unknown(4), minTemp(4), defTemp(4), maxTemp(4)
_THERMAL_INFO_ENTRY_START = 8
_THERMAL_INFO_MIN = 16     # entry[0] offset+8
_THERMAL_INFO_DEF = 20     # entry[0] offset+12
_THERMAL_INFO_MAX = 24     # entry[0] offset+16


def _get_thermal_limit(gpu: int = 0) -> int:
    """Read current thermal limit in °C."""
    h = _handle(gpu)
    b = _buf(_THERMAL_LIMIT_SIZE, 2)  # Version 2!
    fn = _fn.get("GPU_ClientThermalPoliciesGetLimit")
    if fn is None:
        raise NvApiError("GPU_ClientThermalPoliciesGetLimit", -3)
    _check("GPU_ClientThermalPoliciesGetLimit", fn(h, ctypes.cast(b, ctypes.c_void_p)))
    raw = _u32(b, _THERMAL_LIMIT_VALUE)
    # Undo the <<8 shift: e.g., raw=21248 → 21248>>8 = 83°C
    return raw >> 8


def _get_thermal_info(gpu: int = 0) -> tuple[int, int, int]:
    """
    Read thermal range from GetInfo.

    Returns (min_c, default_c, max_c) in degrees Celsius.
    """
    h = _handle(gpu)
    b = _buf(_THERMAL_INFO_SIZE, 1)
    fn = _fn.get("GPU_ClientThermalPoliciesGetInfo")
    if fn is None:
        raise NvApiError("GPU_ClientThermalPoliciesGetInfo", -3)
    _check("GPU_ClientThermalPoliciesGetInfo", fn(h, ctypes.cast(b, ctypes.c_void_p)))

    min_t = _u32(b, _THERMAL_INFO_MIN)
    def_t = _u32(b, _THERMAL_INFO_DEF)
    max_t = _u32(b, _THERMAL_INFO_MAX)

    # The Info struct's temperature values may or may not be <<8 shifted,
    # depending on driver version. The Limit struct (V2) is always shifted,
    # but the Info struct (V1) sometimes isn't. We auto-detect:
    # If any value exceeds a plausible temperature (>200°C), it's shifted.
    if min_t > 200 or def_t > 200 or max_t > 200:
        min_t >>= 8
        def_t >>= 8
        max_t >>= 8

    # Sanity check: default thermal limit should be 30-120°C for any GPU
    if not (30 <= def_t <= 120):
        # Fallback: scan buffer for temp-like values
        found_temps = []
        for off in range(8, len(b) - 3, 4):
            raw = _u32(b, off)
            candidate = raw
            if 30 <= (raw >> 8) <= 120:
                candidate = raw >> 8
            if 60 <= candidate <= 100:
                found_temps.append(candidate)
        if found_temps:
            min_t = min(found_temps)
            max_t = max(found_temps)
            def_t = (min_t + max_t) // 2
        else:
            # Hardcoded safe defaults — 65°C min, 83°C default, 90°C max.
            # These are typical for mid-range GeForce cards (RTX 3060-3080).
            min_t, def_t, max_t = 65, 83, 90

    return (min_t, def_t, max_t)


def _set_thermal_limit(temp_c: int, gpu: int = 0, priority: bool = False):
    """
    Set thermal limit.

    temp_c: target temperature in degrees Celsius
    priority: if True, this limit takes precedence over driver defaults

    From nvapioc:
      thermalLimit.version = MAKE_NVAPI_VERSION(thermalLimit, 2);  // V2!
      thermalLimit.count = 1;
      thermalLimit.entries[0].controller = 1;
      thermalLimit.entries[0].value = tempC << 8;
      thermalLimit.entries[0].flags = priority ? 1 : 0;
    """
    h = _handle(gpu)
    b = _buf(_THERMAL_LIMIT_SIZE, 2)  # Version 2!
    _w32(b, _THERMAL_LIMIT_COUNT, 1)
    _w32(b, _THERMAL_LIMIT_CONTROLLER, 1)
    _w32(b, _THERMAL_LIMIT_VALUE, temp_c << 8)
    _w32(b, _THERMAL_LIMIT_FLAGS, 1 if priority else 0)

    fn = _fn.get("GPU_ClientThermalPoliciesSetLimit")
    if fn is None:
        raise NvApiError("GPU_ClientThermalPoliciesSetLimit", -3)
    _check_set("GPU_ClientThermalPoliciesSetLimit", fn, h, ctypes.cast(b, ctypes.c_void_p))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fan control
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Old Fan API: SetCoolerLevels ──
# This is the simpler, older fan control interface. 28 bytes, 3 cooler slots.
# Layout:
#   +0:  version    (uint32, MAKE_NVAPI_VERSION tag)
#   +4:  cooler[0].level   (uint32, 0-100 percent)
#   +8:  cooler[0].policy  (uint32, 1=manual, 32=auto/default)
#   +12: cooler[1].level
#   +16: cooler[1].policy
#   +20: cooler[2].level
#   +24: cooler[2].policy
#
# The function call takes 3 args: (gpuHandle, coolerIndex, structPtr)
# where coolerIndex selects which physical fan to control (0 = all fans).
# This API works on all NVIDIA GPUs — used as fallback when new API fails.
_FAN_POLICY_MANUAL = 1     # User controls fan speed directly
_FAN_POLICY_AUTO = 32      # GPU controls fan speed via its internal curve


def _set_cooler_level(speed_pct: int, cooler_index: int = 0, gpu: int = 0):
    """
    Set fan speed using old SetCoolerLevels API.

    speed_pct: 0-100%
    cooler_index: 0 = all fans, or specific fan index

    From nvapioc:
      coolerLevel.coolers[0].level = speed;
      coolerLevel.coolers[0].policy = 1;  // manual
      NvAPI_GPU_SetCoolerLevels(handle, fanIndex, &coolerLevel);
    """
    h = _handle(gpu)
    b = _buf(_COOLER_LEVELS_SIZE, 1)
    _w32(b, 4, max(0, min(100, speed_pct)))  # cooler[0].level
    _w32(b, 8, _FAN_POLICY_MANUAL)            # cooler[0].policy = manual

    fn = _fn.get("GPU_SetCoolerLevels")
    if fn is None:
        raise NvApiError("GPU_SetCoolerLevels", -3)
    _check_set("GPU_SetCoolerLevels", fn, h, cooler_index, ctypes.cast(b, ctypes.c_void_p))


def _set_cooler_auto(cooler_index: int = 0, gpu: int = 0):
    """Reset fan to auto/default using old SetCoolerLevels API."""
    h = _handle(gpu)
    b = _buf(_COOLER_LEVELS_SIZE, 1)
    _w32(b, 4, 30)                    # cooler[0].level (ignored in auto mode)
    _w32(b, 8, _FAN_POLICY_AUTO)      # cooler[0].policy = auto (32)

    fn = _fn.get("GPU_SetCoolerLevels")
    if fn is None:
        raise NvApiError("GPU_SetCoolerLevels", -3)
    _check_set("GPU_SetCoolerLevels", fn, h, cooler_index, ctypes.cast(b, ctypes.c_void_p))


def _set_fan_new_api(speed_pct: int, manual: bool = True, gpu: int = 0):
    """
    Set fan speed using new ClientFanCoolersSetControl API.

    Uses a Get-Modify-Set pattern (from Demion/nvapioc):
      1. GetControl → fills 1452-byte buffer with current fan state
      2. Modify level + mode fields in each fan entry
      3. SetControl → writes modified buffer back

    This preserves fields we don't understand (there are many unknown bytes)
    while only changing the speed and mode. Falls back to old SetCoolerLevels
    API if the new API isn't available or fails.

    MULTI-METHOD FALLBACK: This function has 4 fallback points to old API.
    See ignore/useful_patterns_and_future_improvements_for_gpu_oc.md.
    """
    h = _handle(gpu)
    fn_get = _fn.get("GPU_ClientFanCoolersGetControl")
    fn_set = _fn.get("GPU_ClientFanCoolersSetControl")

    if fn_get is None or fn_set is None:
        # Fallback to old API
        _set_cooler_level(speed_pct, 0, gpu)
        return

    # Get current control state
    b = _buf(_FAN_CONTROL_SIZE, 1)
    try:
        _check("GPU_ClientFanCoolersGetControl", fn_get(h, ctypes.cast(b, ctypes.c_void_p)))
    except NvApiError:
        # Fallback to old API
        _set_cooler_level(speed_pct, 0, gpu)
        return

    # The fan count is usually at offset 8, but some driver versions put it
    # at offset 4. Try both.
    count = _u32(b, 8)
    if count == 0:
        count = _u32(b, 4)  # Alternative location

    if count == 0 or count > 16:
        # Can't determine layout, use old API
        _set_cooler_level(speed_pct, 0, gpu)
        return

    # Fan control entry layout (from FanCoolersGetControl V1):
    # The struct has a 12-byte header, then 'count' entries of variable size.
    # Each entry is ~68 bytes (probed on RTX 3080). Layout per entry:
    #   +0:  coolerId (uint32)
    #   +4:  unknown  (uint32)
    #   +8:  level    (uint32, 0-100 percent) ← what we modify
    #   +12: mode     (uint32, 0=auto, 1=manual) ← what we modify
    #   +16..+67: remaining fields (RPM target, etc — preserved via Get-Modify-Set)
    #
    # We compute entry size dynamically: (total_size - header) / count
    # If the math doesn't make sense, bail to old API.
    ENTRY_SIZE_GUESS = 68  # Expected: (1452 - 12) / 21 entries ≈ 68
    if count <= 0 or count > 32:
        _set_cooler_level(speed_pct, 0, gpu)
        return

    entry_size = (_FAN_CONTROL_SIZE - 12) // count if count > 0 else 0
    if entry_size < 12 or entry_size > 256:
        # Can't determine layout, use old API
        _set_cooler_level(speed_pct, 0, gpu)
        return

    mode_val = 1 if manual else 0
    for i in range(count):
        base = 12 + i * entry_size
        # level is at +8 within each entry, mode at +12
        level_off = base + 8
        mode_off = base + 12
        if level_off + 8 <= len(b):
            _w32(b, level_off, max(0, min(100, speed_pct)))
            _w32(b, mode_off, mode_val)

    _check_set("GPU_ClientFanCoolersSetControl", fn_set, h, ctypes.cast(b, ctypes.c_void_p))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OC Status dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OcStatus:
    """Current overclocking status — snapshot of all controllable parameters.

    This is the read-only view of what's currently applied. Used by:
      - CLI 'oc --status' command
      - Future GUI dashboard to display current OC settings
      - Profile save/load to capture current state

    All values are in user-friendly units (MHz, %, °C) not raw NVAPI units.
    """
    gpu_name: str = ""
    bus_id: int = 0
    core_offset_mhz: float = 0.0
    core_offset_range_mhz: tuple[float, float] = (0.0, 0.0)
    mem_offset_mhz: float = 0.0
    mem_offset_range_mhz: tuple[float, float] = (0.0, 0.0)
    power_pct: float = 100.0
    power_range_pct: tuple[float, float] = (50.0, 150.0)
    thermal_c: int = 83
    thermal_range_c: tuple[int, int] = (65, 90)
    fan_pct: int | None = None  # None = auto/unknown, 0-100 = manual speed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Device cache integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-GPU cache of probed struct layouts. Avoids redundant buffer scans
# and API fallback trials across sessions. See lib/device_cache.py.
#
# All cache operations are wrapped in try/except — cache failures NEVER
# block normal operation. The cache is purely a performance optimization.

# In-session cache: lazily loaded on first use, persists for the session.
_device_cache: dict | None = None
_session_fan_api: dict[int, str] = {}  # gpu_idx → "new" or "old" (in-session only)


def _get_driver_version() -> str:
    """Get driver version string from NVML (for cache key).

    Returns empty string if NVML isn't available (safe — just means no caching).
    """
    try:
        from kingai_gpu.lib.nvml import snapshot
        snap = snapshot(0)
        return snap.driver_version
    except Exception:
        return ""


def _save_probe_to_cache(status: OcStatus, gpu: int = 0):
    """Save probe outcomes to device cache after a successful get_oc_status().

    This records what the probing logic discovered (thermal shift, power offsets,
    fan API preference) so the next session can skip redundant probes.
    """
    global _device_cache
    try:
        from datetime import datetime
        from kingai_gpu.lib.device_cache import (
            GpuCacheEntry, load_cache, put_entry,
        )

        driver = _get_driver_version()
        if not status.gpu_name or not driver:
            return  # Can't build a cache key without identity

        entry = GpuCacheEntry(
            gpu_name=status.gpu_name,
            bus_id=status.bus_id,
            driver_version=driver,
            # Record whether power primary offsets succeeded
            # (if power_pct is non-default, primary offsets worked)
            power_primary_ok=status.power_pct != 100.0 or status.power_range_pct != (50.0, 150.0),
            # Record thermal shift detection (if thermal_c is sane, probe worked)
            thermal_shifted=None,  # We don't expose shift status from _get_thermal_info
            # Fan API preference from in-session tracking
            fan_api=_session_fan_api.get(gpu),
            cached_at=datetime.now().isoformat(timespec="seconds"),
        )

        if _device_cache is None:
            _device_cache = load_cache()
        _device_cache = put_entry(entry, _device_cache)
    except Exception:
        pass  # Cache save failure is non-fatal


def _get_cached_fan_api(gpu: int = 0) -> str | None:
    """Check device cache for known fan API preference.

    Returns "new", "old", or None (no cached preference).
    Checks in-session cache first (fastest), then persisted cache.
    """
    # In-session cache (set during this run by _note_fan_api)
    if gpu in _session_fan_api:
        return _session_fan_api[gpu]

    # Persisted cache from a previous session
    global _device_cache
    try:
        from kingai_gpu.lib.device_cache import load_cache, get_entry

        if _device_cache is None:
            _device_cache = load_cache()

        gpu_name = _get_gpu_name(gpu)
        bus_id = _get_bus_id(gpu)
        driver = _get_driver_version()
        if not gpu_name or not driver:
            return None

        entry = get_entry(gpu_name, bus_id, driver, _device_cache)
        if entry is not None and entry.fan_api:
            _session_fan_api[gpu] = entry.fan_api  # Promote to session cache
            return entry.fan_api
    except Exception:
        pass
    return None


def _note_fan_api(api: str, gpu: int = 0):
    """Record which fan API worked (in-session). Persisted on next cache save."""
    _session_fan_api[gpu] = api


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API  (matches overclock.py imports)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def enable_oc(gpu: int = 0):
    """Initialize NVAPI and validate GPU index.

    Must be called before any Set operations. Ensures the DLL is loaded,
    all function pointers are resolved, and the requested GPU exists.
    Subsequent calls are no-ops (idempotent).
    """
    _init()
    _ = _handle(gpu)  # Validate GPU index exists, raises on invalid


def get_oc_status(gpu: int = 0) -> OcStatus:
    """Read all current OC settings into a single dataclass.

    GRACEFUL DEGRADATION: Each subsystem (clocks, power, thermal) is read
    independently with its own try/except. If one fails, the others still
    populate. Partial data is better than no data.
    """
    s = OcStatus()
    s.gpu_name = _get_gpu_name(gpu)
    s.bus_id = _get_bus_id(gpu)

    # Clock offsets from PStates20 — the most reliable read
    try:
        ps_buf = _get_pstates20(gpu)
        core_val, core_min, core_max = _read_clock_delta(ps_buf, domain_id=0)
        mem_val, mem_min, mem_max = _read_clock_delta(ps_buf, domain_id=4)
        s.core_offset_mhz = core_val / 1000.0
        s.core_offset_range_mhz = (core_min / 1000.0, core_max / 1000.0)
        s.mem_offset_mhz = mem_val / 1000.0
        s.mem_offset_range_mhz = (mem_min / 1000.0, mem_max / 1000.0)
    except NvApiError:
        pass

    # Power limit — reads current target and allowed range
    try:
        current_pcm = _get_power_status(gpu)
        min_pcm, def_pcm, max_pcm = _get_power_info(gpu)
        if def_pcm > 0:
            s.power_pct = current_pcm / 1000.0
            s.power_range_pct = (min_pcm / 1000.0, max_pcm / 1000.0)
    except NvApiError:
        pass

    # Thermal limit — current throttle point
    try:
        s.thermal_c = _get_thermal_limit(gpu)
    except NvApiError:
        pass  # Keep default (83°C)

    # Thermal range — min/max allowed temperature targets
    try:
        t_min, t_def, t_max = _get_thermal_info(gpu)
        s.thermal_range_c = (t_min, t_max)
    except NvApiError:
        pass

    # ── Save probe results to device cache ──
    # Non-blocking: if cache save fails, everything still works.
    _save_probe_to_cache(s, gpu)

    return s


def set_core_offset(mhz: int, gpu: int = 0):
    """Set core (graphics) clock offset in MHz. Positive = overclock, negative = undervolt."""
    _set_clock_offset(domain_id=0, offset_khz=mhz * 1000, gpu=gpu)


def set_mem_offset(mhz: int, gpu: int = 0):
    """Set memory clock offset in MHz. Positive = overclock."""
    _set_clock_offset(domain_id=4, offset_khz=mhz * 1000, gpu=gpu)


def set_power_limit(pct: float, gpu: int = 0):
    """Set power limit as percentage (e.g., 100 = default, 125 = +25%)."""
    pcm = int(pct * 1000)
    _set_power_status(pcm, gpu)


def set_thermal_limit(temp_c: int, gpu: int = 0):
    """Set thermal limit in degrees Celsius."""
    _set_thermal_limit(temp_c, gpu)


def set_fan_speed(pct: int, gpu: int = 0):
    """Set fan speed as percentage (0-100%).

    MULTI-METHOD FALLBACK: Tries new ClientFanCoolersSetControl first
    (supports per-fan control, more features), then falls back to old
    SetCoolerLevels API (simpler, works on all GPUs).

    DEVICE CACHE: If a previous session recorded which API works for this
    GPU, skip straight to that API to avoid the Get-Modify-Set overhead
    of the new API when it's going to fail anyway.
    """
    # Check device cache for fan API preference
    cached_api = _get_cached_fan_api(gpu)
    if cached_api == "old":
        _set_cooler_level(pct, cooler_index=0, gpu=gpu)
        return

    try:
        _set_fan_new_api(pct, manual=True, gpu=gpu)
        _note_fan_api("new", gpu)
    except NvApiError:
        _set_cooler_level(pct, cooler_index=0, gpu=gpu)
        _note_fan_api("old", gpu)


def set_fan_auto(gpu: int = 0):
    """Reset fan to automatic control."""
    cached_api = _get_cached_fan_api(gpu)
    if cached_api == "old":
        _set_cooler_auto(cooler_index=0, gpu=gpu)
        return

    try:
        _set_fan_new_api(0, manual=False, gpu=gpu)
        _note_fan_api("new", gpu)
    except NvApiError:
        _set_cooler_auto(cooler_index=0, gpu=gpu)
        _note_fan_api("old", gpu)


def reset_all(gpu: int = 0):
    """Reset all OC settings to stock defaults.

    GRACEFUL DEGRADATION: Each reset is independent. If core reset fails,
    we still try memory, power, thermal, and fan. Only raises at the end
    if any individual reset failed. This ensures maximum rollback even
    when some subsystems are in a bad state.
    """
    errors = []

    # Reset core clock offset to 0 (no overclock/undervolt)
    try:
        set_core_offset(0, gpu)
    except NvApiError as e:
        errors.append(f"core: {e}")

    # Reset memory clock offset to 0
    try:
        set_mem_offset(0, gpu)
    except NvApiError as e:
        errors.append(f"memory: {e}")

    # Reset power limit to factory default (reads default from Info struct)
    try:
        _, def_pcm, _ = _get_power_info(gpu)
        _set_power_status(def_pcm, gpu)
    except NvApiError as e:
        try:
            _set_power_status(100000, gpu)  # Fallback: 100% if we can't read default
        except NvApiError:
            errors.append(f"power: {e}")

    # Reset thermal limit to factory default
    try:
        _, def_t, _ = _get_thermal_info(gpu)
        _set_thermal_limit(def_t, gpu)
    except NvApiError as e:
        errors.append(f"thermal: {e}")

    # Reset fan to automatic control (GPU manages fan curve)
    try:
        set_fan_auto(gpu)
    except NvApiError as e:
        errors.append(f"fan: {e}")

    if errors:
        # Some resets failed — report but the ones that succeeded are still applied
        raise NvApiError("reset_all", -1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Debug helpers (for probing / development)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def dump_buf(buf, label: str = "", max_bytes: int = 128):
    """Print hex + uint32 dump of buffer (for debugging struct layouts).

    Shows each 4-byte word as hex, unsigned, and signed decimal.
    Essential for reverse engineering new NVAPI structs — run this
    on a Get buffer to figure out where the fields are.
    """
    size = len(buf)
    print(f"\n{'═' * 60}")
    print(f"  {label}  ({size} bytes)")
    print(f"{'═' * 60}")
    for off in range(0, min(size, max_bytes), 4):
        u = _u32(buf, off)
        i = _i32(buf, off)
        raw = bytes(buf[off:off + 4])
        print(f"  +{off:4d}  0x{u:08X}  u={u:>12d}  i={i:>12d}  {raw.hex()}")
    if size > max_bytes:
        print(f"  ... ({size - max_bytes} more bytes)")


def dump_pstates(gpu: int = 0):
    """Dump parsed PStates20 data for debugging."""
    try:
        buf = _get_pstates20(gpu)
    except NvApiError as e:
        print(f"GetPstates20 failed: {e}")
        return

    n_ps = _u32(buf, _PS_NUM_PSTATES)
    n_ck = _u32(buf, _PS_NUM_CLOCKS)
    n_vt = _u32(buf, _PS_NUM_BASE_VOLTAGES)
    print(f"\nPStates20: {n_ps} pstates, {n_ck} clocks, {n_vt} base voltages")

    for pi in range(min(n_ps, 4)):  # Show first 4 pstates
        ps_id = _u32(buf, _pstate_off(pi))
        print(f"\n  P{ps_id}:")
        for ci in range(min(n_ck, _MAX_CLOCKS)):
            base = _clock_off(pi, ci)
            domain = _u32(buf, base + _CK_DOMAIN)
            ctype = _u32(buf, base + _CK_TYPE)
            delta_val = _i32(buf, base + _CK_DELTA_VAL)
            delta_min = _i32(buf, base + _CK_DELTA_MIN)
            delta_max = _i32(buf, base + _CK_DELTA_MAX)
            freq = _u32(buf, base + _CK_DATA_FREQ)
            domain_name = {0: "Core", 4: "Memory"}.get(domain, f"Dom{domain}")
            print(
                f"    Clock {ci}: {domain_name:>6}  "
                f"type={ctype}  "
                f"delta={delta_val / 1000:+.0f} MHz  "
                f"range=[{delta_min / 1000:+.0f}, {delta_max / 1000:+.0f}]  "
                f"freq={freq / 1000:.0f} MHz"
            )
