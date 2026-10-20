# KingAi GPU Overclocker

Python CLI for NVIDIA GPU monitoring, clock-offset/limit planning, explicit
NVAPI execution, profile validation, and CUDA-backed memory testing.

## Safety model

- Monitoring commands are read-only. `oc --status` is rejected before backend
  construction when combined with a setting, `--reset`, or `--load`.
- Every hardware-changing `oc`, `import-msi`, and memory-sweep command is a
  dry-run by default.
- CLI hardware writes require an explicit `--execute`. Direct Python calls to
  `NvApiBackend` or the public low-level `nvapi` writers separately require
  the keyword-only argument `execute=True`. An unauthorized backend call fails
  before loading the low-level module; an unauthorized low-level writer fails
  before `NvAPI_Initialize` or any write attempt.
- CLI execute paths check values against driver-reported live ranges before
  the first write; fan values are constrained to 0-100%.
- Version 3 profiles require an exact GPU name and a matching UUID or PCI bus
  identity.
- NVAPI devices are correlated to NVML by PCI bus rather than assumed to share
  the same enumeration order. Ambiguous mappings are not accepted for writes.
- Multi-setting operations snapshot the prior state and attempt rollback after
  any partial failure.
- Fan changes are refused when the existing control mode cannot be read, since
  that state could not be rolled back exactly.
- A memory sweep restores the exact pre-sweep memory offset, not an assumed
  `0 MHz`.

This tool does not implement NVIDIA V/F-curve voltage control. A negative core
clock offset lowers the requested clock offset; it is not, by itself,
undervolting.

The low-level Python writer functions are intentionally sharp tools. Their
`execute=True` gate prevents accidental invocation, but they do not provide
the CLI's identity checks, range validation, transaction read-back, or
rollback. Use the CLI or its validated orchestration for normal operation.

## Install

```bash
python -m pip install /path/to/kingai_gpu-0.2.1rc1-py3-none-any.whl
kingai-gpu --version
python -m kingai_gpu --help
```

Monitoring requires `nvidia-ml-py`. Hardware control is Windows-only and uses
NVAPI from the installed NVIDIA driver. A genuine GPU-memory test or sweep
requires a working CuPy package that matches the local CUDA runtime.

## Monitoring

```bash
kingai-gpu monitor --once
kingai-gpu monitor --once --json
kingai-gpu monitor --json
kingai-gpu monitor --csv
kingai-gpu info
```

Single-shot JSON is one JSON value. Continuous JSON is JSON Lines (one compact
JSON object per line). CSV and JSON stdout contain no log banner or shutdown
message; diagnostics and log paths go to stderr. stdout telemetry and stderr
diagnostics are mirrored character-for-character to per-user log files under
`%LOCALAPPDATA%\KingAiGPU\logs` on Windows or the XDG state directory on Linux.
Set `KINGAI_GPU_LOG_DIR` to choose another directory.

## Plan, execute, and inspect settings

```bash
# Read-only
kingai-gpu oc --status

# Plan only: validates and prints the intended change
kingai-gpu oc --core 75 --mem 200

# Explicit execution
kingai-gpu oc --core 75 --mem 200 --execute

# Reset is also a plan unless --execute is present
kingai-gpu oc --reset
kingai-gpu oc --reset --execute
```

Saving a profile is read-only with respect to the GPU:

```bash
kingai-gpu oc --save my-profile.json
kingai-gpu oc --load my-profile.json
kingai-gpu oc --load my-profile.json --execute
```

Profile output uses exclusive file creation. Existing files are never silently
overwritten.

## MSI Afterburner profile import

The importer reads a per-GPU `VEN_10DE...cfg`; it never edits Afterburner's
files.

```bash
kingai-gpu import-msi path/to/VEN_10DE...cfg --list
kingai-gpu import-msi path/to/VEN_10DE...cfg --section Profile3
kingai-gpu import-msi path/to/VEN_10DE...cfg --section Profile3 --execute
kingai-gpu import-msi path/to/VEN_10DE...cfg --section Profile3 --save profile.json
```

Before execution, the importer checks that the source filename contains a
complete NVIDIA device identity and that its PCI bus, slot/device, and function
match the selected GPU's complete NVML BDF. A bus-only fallback is refused.
Unprefixed numeric Afterburner BDF `BUS` and trailing `DEV`/`FN` components are
interpreted as decimal (`BUS_10` matches NVML bus `0A`); values containing A-F
are treated as hexadecimal. The GPU hardware `VEN`, first `DEV`, and `SUBSYS`
IDs are matched as hexadecimal identifiers. Extracted settings must also fit
that GPU's live ranges.
Unsupported voltage-boost/VF-curve data is reported and not applied.

## Memory testing and sweep

```bash
# Tests GPU VRAM when CuPy/CUDA works; otherwise clearly reports a
# system-RAM-only NumPy fallback.
kingai-gpu memtest --duration 10 --size 256

# Plan only
kingai-gpu memtest --sweep --start 0 --stop 500 --step 50

# Explicitly change offsets and run CUDA workloads
kingai-gpu memtest --sweep --start 0 --stop 500 --step 50 --execute
```

A sweep refuses to run without a genuine CuPy/CUDA GPU-memory backend. A
CUDA device must also have the same complete PCI domain, bus, device, and
function as the selected NVAPI/NVML GPU. Pattern verification runs for the
requested duration at each step (or stops early on an error). A bandwidth
regression is recorded as an observation; it is not claimed as proof of ECC or
error-correction activity.

## Profile format

Version 3 profiles contain:

- `schema` and `schema_version`
- exact `gpu` identity (`name`, `pci_bus_id`, `uuid`, PCI device ID, and PCI
  subsystem ID)
- `settings`
- complete saved ranges for core, memory, power, thermal, and fan controls

Older or differently shaped profiles are intentionally refused for execution;
the v3 identity expansion was not silently folded into v2. Re-save an older
profile against the intended GPU and review the generated plan.

## Known limitations

- `monitor` and `info` contain NVML initialization, missing-device, invalid
  index, and snapshot failures as concise stderr diagnostics with nonzero exit
  codes rather than exposing Python tracebacks.
- A `GpuSnapshot` is a best-effort sequence of NVML reads, not an atomic
  hardware sample. Values can therefore span a small time interval.
- The included CI workflow uses offline fake backends on Linux and Windows. It
  exercises authorization, transaction, and Windows import/CLI paths but
  cannot establish physical NVAPI write/read-back behavior or overclock
  stability.
- Install the maintained `nvidia-ml-py` distribution. If Python reports that a
  separately installed `pynvml` distribution is deprecated or shadows the
  binding, uninstall that standalone distribution and reinstall
  `nvidia-ml-py`; the import module remains named `pynvml`.

See [RELEASE_REVIEW.md](RELEASE_REVIEW.md) for the per-module review and explicit
software/hardware qualification limits, and [PRESERVED_SOURCES.md](PRESERVED_SOURCES.md)
for exact original source provenance. This prerelease fixes CLI/logging failures,
unavailable bus identity handling and automatic-fan fallback behavior using
offline regressions only. It does not establish physical hardware safety.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

Tests use an injected fake backend and a fake DLL loader for authorization
checks. They do not initialize NVAPI, apply offsets, change limits/fans, or run
a GPU stress workload. Passing them does not certify a clock, voltage, memory
offset, driver, or physical GPU as stable.

## Project layout

```text
kingai_gpu/
  cli/
    main.py
    monitor.py
    overclock.py
    memtest.py
    import_msi_profile_single.py
  lib/
    backend.py
    nvapi.py
    nvml.py
    device_cache.py
tests/
```

## Version and license

Current package version: `0.2.1rc1` (software qualification prerelease).

MIT License; see `LICENSE`.
