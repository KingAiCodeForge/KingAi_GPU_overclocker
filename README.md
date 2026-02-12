# KingAi GPU Overclocker

Open-source Python toolkit for NVIDIA GPU monitoring, overclocking, and stability testing.  
CLI-first design — scriptable, automatable, headless-friendly. A GUI frontend is planned.

**Does things MSI Afterburner can't:**
- Automated memory OC bandwidth cliff detection  
- VRAM pattern-based error testing (works on GeForce — no ECC counters needed)
- Headless CLI operation (SSH, servers, CI/CD)
- JSON/CSV sensor export for logging and automation

## Quick Start

```bash
pip install -e .
```

### Monitor your GPU
```bash
kingai-gpu monitor              # Live dashboard (refreshes every 1s)
kingai-gpu monitor --once       # Single snapshot
kingai-gpu monitor --json       # JSON output for piping
```

### Overclock
```bash
kingai-gpu oc --core +150       # Core clock offset +150 MHz
kingai-gpu oc --mem +500        # Memory clock offset +500 MHz
kingai-gpu oc --power 125       # Power limit 125%
kingai-gpu oc --fan 80          # Fan speed 80%
kingai-gpu oc --reset           # Reset everything to stock
kingai-gpu oc --save my_oc.json # Save current OC to a profile
kingai-gpu oc --load my_oc.json # Apply saved profile
```

### Import MSI Afterburner Profiles
```bash
kingai-gpu import-msi --auto                # Auto-find AB configs, list all
kingai-gpu import-msi --auto --section Profile3  # Apply Profile3 from auto-detected config
kingai-gpu import-msi path/to/VEN_10DE...cfg --list          # List sections in a specific config
kingai-gpu import-msi path/to/VEN_10DE...cfg --save out.json  # Convert to KingAi format
```

### Memory Stability Test
```bash
kingai-gpu memtest              # Quick VRAM pattern test
kingai-gpu memtest --sweep      # Automated memory OC sweep (find optimal)
kingai-gpu memtest --duration 60  # Run for 60 seconds
```

## Requirements

- Windows 10/11 (overclocking features require NVAPI)
- NVIDIA GPU with recent drivers
- Python 3.10+

Monitoring works on Linux too (via NVML), but overclocking is Windows-only due to NVAPI.

## Architecture

```
kingai_gpu/
├── lib/
│   ├── nvml.py              # NVML wrapper (monitoring - cross-platform)
│   ├── nvapi.py             # NVAPI ctypes bindings (OC control - Windows)
│   └── device_cache.py      # Per-GPU struct layout cache (skip re-probing)
├── cli/
│   ├── main.py              # CLI entry point + argument parsing
│   ├── monitor.py           # Real-time monitoring dashboard
│   ├── overclock.py         # Overclock / undervolt control + profiles
│   ├── memtest.py           # VRAM stability test + memory OC sweep
│   └── import_msi_profile_single.py  # MSI Afterburner profile importer
└── __init__.py
```

## Why Not Just Use Afterburner?

| Feature | Afterburner | KingAi GPU |
|---------|-------------|------------|
| Open source | ❌ | ✅ |
| Scriptable/CLI | ❌ | ✅ |
| Automated mem OC tuning | ❌ | ✅ |
| VRAM error detection | ❌ | ✅ |
| Bandwidth cliff detection | ❌ | ✅ |
| JSON/CSV sensor export | ❌ | ✅ |
| Headless/SSH operation | ❌ | ✅ |
| VF curve editing | ✅ | ❌ (see [MSI Afterburner Tools](https://github.com/KingAiCodeForge/KingAi_MSi_afterburner_overclocker_tools)) |
| In-game overlay (RTSS) | ✅ | ❌ |

The overlay is the one thing we can't replicate (requires a signed kernel driver). VF curve editing is handled by the companion [MSI Afterburner Tools](https://github.com/KingAiCodeForge/KingAi_MSi_afterburner_overclocker_tools) repo which reads/writes Afterburner's config files directly. So making it compatible with Afterburner is a key design goal.

## Status

**Working now (v0.2.0-dev):**
- Full GPU monitoring dashboard with 5 output modes (dashboard, JSON, CSV, compact, info)
- Core/memory clock offset control via NVAPI
- Power limit and thermal limit control
- Fan speed control (manual + auto, with new/old API fallback)
- VRAM pattern testing (CuPy GPU-accelerated, numpy CPU fallback)
- Automated memory OC bandwidth sweep with cliff detection
- Graceful degradation — partial failures don't crash the tool
- OC profile save/load (`--save` / `--load` JSON profiles)
- MSI Afterburner profile import (`import-msi` subcommand)
- Retry on transient NVAPI errors (Afterburner lock contention)
- Per-GPU device cache (skip redundant struct probing across sessions)

**Planned (v0.3+):**
- GUI frontend (DearPyGui or PySide6)
- AMD GPU support (ADL/ADLX for Radeon cards)

## License

MIT
