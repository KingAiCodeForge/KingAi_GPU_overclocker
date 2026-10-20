# 0.2.1rc1 software qualification ledger

Scope: inspect every tracked Python module and test, package and CLI portability,
and synthetic safety regressions. No live NVAPI/NVML access, GPU setting write,
GPU stress workload, physical rollback test, or driver ABI qualification was run.
This is a prerelease; hardware control and tuning stability remain unqualified.

`Inspected` means source review of the whole listed file, not complete semantic
proof. `Tested` describes the actual exercised subset. No tracked Python path is
left unreviewed; explicitly excluded paths/features follow the table.

| Path | Inspected | Changed in this candidate | Tested / limitation |
| --- | --- | --- | --- |
| `kingai_gpu/__init__.py` | yes; package version | 0.2.1rc1 | installed metadata/version agree |
| `kingai_gpu/__main__.py` | yes; module launcher | added | installed module help/version outside checkout |
| `kingai_gpu/cli/__init__.py` | yes; module descriptions | no | package import; no runtime behavior |
| `kingai_gpu/cli/main.py` | yes; parser, dispatch, logging | version, stream restoration/close, optional log failure handling, dependency/I/O diagnostics | all command help; repeated main calls, disk-full write/flush, Unicode log paths, missing runtime dependency; no hardware dispatch |
| `kingai_gpu/cli/monitor.py` | yes; render/format/poll/error paths | finite interval validation | synthetic JSON/CSV, interrupt, invalid index/interval and driver errors; no physical sensor reading |
| `kingai_gpu/cli/overclock.py` | yes; identity/schema/ranges, transactions | no | fake plan/execute/readback/reset/rollback, invalid profile, exclusive save; actual driver behavior untested |
| `kingai_gpu/cli/import_msi_profile_single.py` | yes; discovery/parsing/identity/import | no | synthetic profiles, BDF/device/subsystem mismatch, no overwrite, missing path/list; registry discovery and real profile import untested |
| `kingai_gpu/cli/memtest.py` | yes; CuPy/NumPy work, sweep, restoration | contain workload/allocation failures | fake memory backend, identity/refusal/restoration and workload failure; real CUDA/VRAM and NumPy workloads not executed |
| `kingai_gpu/lib/__init__.py` | yes; package marker | no | imports |
| `kingai_gpu/lib/backend.py` | yes; protocol/identity/authorization/rollback | no | fakes and mock low-level forwarding, ambiguous bus rejection; no driver writes |
| `kingai_gpu/lib/device_cache.py` | yes; optional per-user hint cache | no | synthetic missing/corrupt/Unicode-path save/load and preservation; no real user cache touched |
| `kingai_gpu/lib/nvapi.py` | yes; DLL/struct/read/write/cache/debug paths | unavailable bus sentinel and fan fallback mode | fake DLL authorization, bus-zero distinction and all fan fallback modes; ABI/layouts/retries and live control remain unqualified |
| `kingai_gpu/lib/nvml.py` | yes; lazy init/sensors/snapshot/shutdown | no | synthetic snapshot rendering and contained errors via monitor; no live initialization or successful sensor API call |
| `tests/__init__.py` | yes; package marker | no | collected from checkout and source distribution |
| `tests/test_fakes.py` | yes; in-memory backends | no | used by safety suites; no NVIDIA API |
| `tests/test_import_msi_safety.py` | yes | no | complete suite |
| `tests/test_memtest_safety.py` | yes | no | complete suite, fake memory/sensors |
| `tests/test_monitor_output.py` | yes | no | complete suite, synthetic output |
| `tests/test_overclock_safety.py` | yes | no | complete suite, fake writes only |
| `tests/test_write_authorization.py` | yes | bus identity/fan mode regressions | complete suite, fake DLL and mocked low-level calls |
| `tests/test_cli_portability.py` | yes | added | finite interval, stream/log lifetime, concise dependency/workload failure |
| `tests/test_device_cache.py` | yes | added | synthetic cache fixtures only |

Other inspected build/entrypoint files: `pyproject.toml`, `MANIFEST.in`,
`.github/workflows/python-tests.yml`, `.gitignore`, README, changelog and license.
Build metadata now includes source-distribution tests and review documents.
Existing CI covers offline tests on Linux Python 3.10/3.13 and Windows 3.13.

## Exclusions and remaining qualification

The absent, ignored `ignore/` research directory and `tests/probe_*` are excluded
historical/local work, not reviewed or released. No such files were recovered
from private/local repositories. `nvapi.dump_buf` and `dump_pstates` are inspected
developer helpers, not tested or supported entrypoints. Mixed per-fan state and
undocumented ABI compatibility are not certified by these tests. Linux hardware
control is unsupported by NVAPI; macOS, 32-bit Python, other Python versions,
and real Windows/Linux GPUs have not been qualified in this audit.

## Reproduce the bounded checks

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check kingai_gpu tests
python -m compileall -q kingai_gpu tests
python -m build
```

Install the wheel in a fresh environment, change to an empty directory with
spaces/non-ASCII characters, then run `kingai-gpu --version`, top-level and each
subcommand `--help`, `python -m kingai_gpu --version`, and
`kingai-gpu import-msi missing.cfg --list`. The last must exit 1 with a concise
error and no traceback. These checks never request execution or load NVAPI.
Test a no-dependencies install separately: help/version must work and monitoring
must report the missing `pynvml` runtime dependency without a traceback.

Exact commit, environment, test count, installed paths, archive inventory,
SHA-256 hashes and GitHub checks belong in the release's external receipt.
