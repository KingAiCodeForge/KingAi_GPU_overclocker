"""GPU-memory test and explicitly gated memory-offset sweep.

A sweep changes hardware and therefore requires ``--execute``. It also
requires a real CUDA/CuPy backend; the NumPy fallback is system-RAM-only and is
never accepted as sweep evidence.
"""

from __future__ import annotations

import sys
import time
import importlib
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from kingai_gpu.cli.overclock import PlanError, require_stable_identity
from kingai_gpu.lib.backend import NvApiBackend, OcBackend, parse_pci_bdf
from kingai_gpu.lib.nvml import snapshot

cp = None
_cupy_checked = False

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class MemtestResult:
    timestamp: str = ""
    duration_sec: float = 0.0
    buffer_size_mb: int = 0
    patterns_tested: int = 0
    total_bytes_checked: int = 0
    errors_detected: int = 0
    gpu_temp: int = 0
    mem_clock: int = 0
    passed: bool = True
    backend: str = ""

    def summary(self) -> str:
        status = "PASS" if self.passed else f"FAIL ({self.errors_detected} errors)"
        return (
            f"[{self.timestamp}] {status} | {self.buffer_size_mb}MB x "
            f"{self.patterns_tested} patterns | {self.duration_sec:.1f}s | "
            f"{self.backend} | {self.gpu_temp}C | {self.mem_clock} MHz"
        )


@dataclass
class BandwidthResult:
    mem_offset_mhz: int = 0
    bandwidth_gbps: float = 0.0
    errors: int = 0
    gpu_temp: int = 0
    mem_clock: int = 0
    power_draw: float = 0.0


@dataclass
class SweepResult:
    gpu_name: str = ""
    timestamp: str = ""
    results: list[BandwidthResult] = field(default_factory=list)
    optimal_offset_mhz: int = 0
    peak_bandwidth_gbps: float = 0.0
    cliff_offset_mhz: int = -1
    crash_offset_mhz: int = -1

    def summary(self) -> str:
        lines = [
            f"Memory sweep: {self.gpu_name}",
            f"  Timestamp: {self.timestamp}",
            f"  Best measured offset: {self.optimal_offset_mhz:+d} MHz",
            f"  Peak measured bandwidth: {self.peak_bandwidth_gbps:.1f} GB/s",
        ]
        if self.cliff_offset_mhz >= 0:
            lines.append(
                f"  First >2% step regression: {self.cliff_offset_mhz:+d} MHz "
                "(observation, not proof of error correction)"
            )
        if self.crash_offset_mhz >= 0:
            lines.append(f"  Failed test point: {self.crash_offset_mhz:+d} MHz")
        for result in self.results:
            lines.append(
                f"  {result.mem_offset_mhz:+6d} MHz  "
                f"{result.bandwidth_gbps:8.1f} GB/s  "
                f"errors={result.errors} temp={result.gpu_temp}C "
                f"clock={result.mem_clock}MHz power={result.power_draw:.0f}W"
            )
        return "\n".join(lines)


class GpuMemoryBackend(Protocol):
    """A workload whose buffers and verification execute in GPU memory."""

    is_gpu_memory: bool

    def pci_bus_id(self, gpu_index: int) -> str: ...

    def measure(self, size_mb: int, iterations: int, gpu_index: int) -> float: ...

    def test(
        self,
        size_mb: int,
        gpu_index: int,
        duration_sec: float = 0,
    ) -> MemtestResult: ...


class CupyMemoryBackend:
    """CUDA-backed memory copy and deterministic pattern verification."""

    is_gpu_memory = True
    _PATTERNS = (0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555)

    def _device(self, gpu_index: int):
        return cp.cuda.Device(gpu_index)

    def pci_bus_id(self, gpu_index: int) -> str:
        """Return CUDA's stable PCI identity for the selected device ordinal."""

        value = cp.cuda.runtime.deviceGetPCIBusId(gpu_index)
        return value.decode("ascii", errors="strict") if isinstance(value, bytes) else str(value)

    def measure(self, size_mb: int, iterations: int, gpu_index: int) -> float:
        count = size_mb * 1024 * 1024
        with self._device(gpu_index):
            source = cp.arange(count, dtype=cp.uint8)
            target = cp.empty_like(source)
            for _ in range(3):
                cp.copyto(target, source)
            cp.cuda.runtime.deviceSynchronize()
            started = time.perf_counter()
            for _ in range(iterations):
                cp.copyto(target, source)
            cp.cuda.runtime.deviceSynchronize()
            elapsed = time.perf_counter() - started
            del source, target
            cp.get_default_memory_pool().free_all_blocks()
        return (count * iterations * 2) / elapsed / 1e9

    def test(
        self,
        size_mb: int,
        gpu_index: int,
        duration_sec: float = 0,
    ) -> MemtestResult:
        if duration_sec < 0:
            raise PlanError("GPU-memory test duration must not be negative")
        count = size_mb * 1024 * 1024 // 4
        result = MemtestResult(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            buffer_size_mb=size_mb,
            backend="cupy-cuda-vram",
        )
        started = time.perf_counter()
        deadline = started + duration_sec
        with self._device(gpu_index):
            buffer = cp.empty(count, dtype=cp.uint32)
            errors = 0
            while result.patterns_tested == 0 or time.perf_counter() < deadline:
                for pattern in self._PATTERNS:
                    buffer.fill(pattern)
                    cp.cuda.runtime.deviceSynchronize()
                    errors += int(cp.count_nonzero(buffer != cp.uint32(pattern)).get())
                    result.patterns_tested += 1
                    if errors:
                        break
                if errors:
                    break
            result.total_bytes_checked = size_mb * 1024 * 1024 * result.patterns_tested
            result.errors_detected = errors
            result.passed = errors == 0
            del buffer
            cp.get_default_memory_pool().free_all_blocks()
        result.duration_sec = time.perf_counter() - started
        try:
            state = snapshot(gpu_index)
            result.gpu_temp = state.temp_gpu
            result.mem_clock = state.clock_mem
        except Exception:
            pass
        return result


def get_gpu_memory_backend(gpu_index: int = 0) -> GpuMemoryBackend | None:
    """Return CuPy only after a live CUDA device check succeeds."""

    global cp, _cupy_checked
    if not _cupy_checked:
        _cupy_checked = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cp = importlib.import_module("cupy")
        except (ImportError, OSError):
            cp = None
    if cp is None:
        return None
    try:
        if gpu_index < 0 or cp.cuda.runtime.getDeviceCount() <= gpu_index:
            return None
        with cp.cuda.Device(gpu_index):
            cp.cuda.runtime.deviceSynchronize()
    except Exception:
        return None
    return CupyMemoryBackend()


def _run_system_ram_fallback(size_mb: int, gpu_index: int) -> MemtestResult:
    """Clearly labelled development fallback; this does not test VRAM."""

    if np is None:
        raise RuntimeError("NumPy is required for the system-RAM fallback")
    count = size_mb * 1024 * 1024 // 4
    result = MemtestResult(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        buffer_size_mb=size_mb,
        backend="numpy-system-ram",
    )
    started = time.perf_counter()
    buffer = np.empty(count, dtype=np.uint32)
    errors = 0
    for pattern in CupyMemoryBackend._PATTERNS:
        buffer.fill(pattern)
        errors += int(np.count_nonzero(buffer != np.uint32(pattern)))
    result.patterns_tested = len(CupyMemoryBackend._PATTERNS)
    result.total_bytes_checked = size_mb * 1024 * 1024 * result.patterns_tested
    result.errors_detected = errors
    result.passed = errors == 0
    result.duration_sec = time.perf_counter() - started
    try:
        state = snapshot(gpu_index)
        result.gpu_temp = state.temp_gpu
        result.mem_clock = state.clock_mem
    except Exception:
        pass
    return result


def run_memtest(size_mb: int = 256, gpu_index: int = 0) -> MemtestResult:
    if size_mb <= 0:
        raise PlanError("Buffer size must be greater than zero")
    backend = get_gpu_memory_backend(gpu_index)
    if backend is not None:
        return backend.test(size_mb, gpu_index)
    return _run_system_ram_fallback(size_mb, gpu_index)


def validate_sweep(
    start_mhz: int,
    stop_mhz: int,
    step_mhz: int,
    test_duration: int,
    size_mb: int,
    live_state,
) -> None:
    """Reject unsafe sweep parameters before the first write call."""

    require_stable_identity(live_state.identity)
    if "mem_offset_mhz" not in live_state.valid_settings:
        raise PlanError(
            "Selected GPU memory-offset read/range is unavailable; refusing to "
            "sweep using guessed state"
        )
    if step_mhz <= 0:
        raise PlanError("Sweep step must be greater than zero")
    if start_mhz > stop_mhz:
        raise PlanError("Sweep start must not exceed sweep stop")
    if test_duration <= 0:
        raise PlanError("Sweep duration must be greater than zero")
    if size_mb <= 0:
        raise PlanError("Sweep buffer size must be greater than zero")
    minimum, maximum = live_state.mem_offset_range_mhz
    if start_mhz < minimum or stop_mhz > maximum:
        raise PlanError(
            f"Sweep [{start_mhz}, {stop_mhz}] MHz is outside the selected GPU "
            f"range [{minimum:g}, {maximum:g}]"
        )


def validate_memory_backend_identity(
    memory_backend: GpuMemoryBackend,
    gpu_index: int,
    live_state,
) -> None:
    """Require CUDA and the OC backend to resolve to the same PCI function."""

    identity_reader = getattr(memory_backend, "pci_bus_id", None)
    if not callable(identity_reader):
        raise PlanError("GPU-memory backend does not expose a PCI bus identity")
    try:
        memory_bus_id = identity_reader(gpu_index)
    except Exception as exc:
        raise PlanError(f"Unable to read GPU-memory backend PCI identity: {exc}") from exc
    memory_bdf = parse_pci_bdf(str(memory_bus_id))
    selected_bdf = parse_pci_bdf(str(live_state.identity.pci_bus_id))
    if memory_bdf is None or selected_bdf is None:
        raise PlanError(
            "CUDA and OC backends must both expose complete PCI "
            "domain:bus:device.function identities"
        )
    if memory_bdf != selected_bdf:
        raise PlanError(
            f"GPU-memory backend PCI {memory_bus_id!r} does not match selected "
            f"GPU PCI {live_state.identity.pci_bus_id!r}"
        )


def run_sweep(
    start_mhz: int = 0,
    stop_mhz: int = 1500,
    step_mhz: int = 50,
    test_duration: int = 10,
    size_mb: int = 256,
    gpu_index: int = 0,
    *,
    execute: bool = False,
    backend: OcBackend | None = None,
    memory_backend: GpuMemoryBackend | None = None,
    settle_seconds: float = 2.0,
    sensor_reader=None,
) -> SweepResult:
    """Execute a validated sweep and always restore the exact prior offset."""

    if not execute:
        raise PlanError("Memory sweep requires explicit execute=True / --execute")
    if memory_backend is None or not getattr(memory_backend, "is_gpu_memory", False):
        raise PlanError("Memory sweep requires a genuine GPU-memory backend")

    selected = backend or NvApiBackend()
    live = selected.get_state(gpu_index)
    validate_sweep(start_mhz, stop_mhz, step_mhz, test_duration, size_mb, live)
    validate_memory_backend_identity(memory_backend, gpu_index, live)
    prior_offset = live.mem_offset_mhz
    read_sensors = sensor_reader or snapshot
    result = SweepResult(
        gpu_name=live.identity.name,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        optimal_offset_mhz=int(prior_offset),
    )
    peak_bandwidth = 0.0
    previous_bandwidth = 0.0
    restore_error: Exception | None = None
    offset_write_attempted = False

    try:
        selected.prepare_write(gpu_index, execute=True)
        for offset in range(start_mhz, stop_mhz + 1, step_mhz):
            print(f"Testing {offset:+d} MHz...", end=" ", flush=True)
            try:
                offset_write_attempted = True
                selected.set_setting(
                    "mem_offset_mhz",
                    offset,
                    gpu_index,
                    execute=True,
                )
                observed = selected.get_state(gpu_index)
                if (
                    "mem_offset_mhz" not in observed.valid_settings
                    or abs(float(observed.mem_offset_mhz) - offset) > 0.01
                ):
                    raise RuntimeError(
                        "memory-offset read-back did not confirm "
                        f"{offset:+d} MHz (observed {observed.mem_offset_mhz!r})"
                    )
                if settle_seconds > 0:
                    time.sleep(settle_seconds)
                bandwidth = memory_backend.measure(
                    size_mb=size_mb,
                    iterations=max(10, test_duration * 5),
                    gpu_index=gpu_index,
                )
                tested = memory_backend.test(
                    size_mb=size_mb,
                    gpu_index=gpu_index,
                    duration_sec=test_duration,
                )
                sensors = read_sensors(gpu_index)
            except Exception as exc:
                result.crash_offset_mhz = offset
                print(f"FAILED: {exc}")
                break

            point = BandwidthResult(
                mem_offset_mhz=offset,
                bandwidth_gbps=bandwidth,
                errors=tested.errors_detected,
                gpu_temp=sensors.temp_gpu,
                mem_clock=sensors.clock_mem,
                power_draw=sensors.power_draw,
            )
            result.results.append(point)
            print(f"{bandwidth:.1f} GB/s, errors={tested.errors_detected}")

            if tested.errors_detected == 0 and bandwidth > peak_bandwidth:
                peak_bandwidth = bandwidth
                result.optimal_offset_mhz = offset
            if (
                previous_bandwidth > 0
                and bandwidth < previous_bandwidth * 0.98
                and result.cliff_offset_mhz < 0
            ):
                result.cliff_offset_mhz = offset
            if tested.errors_detected:
                break
            previous_bandwidth = bandwidth
    except KeyboardInterrupt:
        print("Sweep interrupted.", file=sys.stderr)
    finally:
        if offset_write_attempted:
            try:
                selected.set_setting(
                    "mem_offset_mhz",
                    prior_offset,
                    gpu_index,
                    execute=True,
                )
                restored = selected.get_state(gpu_index)
                if (
                    "mem_offset_mhz" not in restored.valid_settings
                    or abs(float(restored.mem_offset_mhz) - float(prior_offset)) > 0.01
                ):
                    raise RuntimeError(
                        f"restore read-back observed {restored.mem_offset_mhz!r} MHz"
                    )
            except Exception as exc:
                restore_error = exc

    result.peak_bandwidth_gbps = peak_bandwidth
    if restore_error is not None:
        raise RuntimeError(
            f"Could not restore exact prior memory offset {prior_offset:+g} MHz: "
            f"{restore_error}"
        ) from restore_error
    return result


def cmd_memtest(
    args,
    backend: OcBackend | None = None,
    memory_backend: GpuMemoryBackend | None = None,
) -> int:
    """Run a read-only test or an explicitly executed changing sweep."""

    gpu = args.gpu
    if gpu < 0:
        print("GPU index must not be negative.", file=sys.stderr)
        return 2
    if args.duration <= 0 or args.size <= 0:
        print("Duration and buffer size must be greater than zero.", file=sys.stderr)
        return 2

    if args.sweep:
        gpu_memory = memory_backend or get_gpu_memory_backend(gpu)
        if gpu_memory is None or not getattr(gpu_memory, "is_gpu_memory", False):
            print(
                "Sweep refused: a working CuPy/CUDA GPU-memory backend is required. "
                "No offset was changed.",
                file=sys.stderr,
            )
            return 2
        selected = backend or NvApiBackend()
        try:
            live = selected.get_state(gpu)
            validate_sweep(
                args.start,
                args.stop,
                args.step,
                args.duration,
                args.size,
                live,
            )
            validate_memory_backend_identity(gpu_memory, gpu, live)
        except Exception as exc:
            print(f"Sweep refused before hardware access: {exc}", file=sys.stderr)
            return 2

        if not args.execute:
            print(f"PLAN ONLY — memory sweep on GPU {gpu}: {live.identity.name}")
            print(f"  Range: {args.start:+d} to {args.stop:+d} MHz, step {args.step} MHz")
            print(f"  Exact offset to restore: {live.mem_offset_mhz:+g} MHz")
            print(f"  GPU-memory backend: {type(gpu_memory).__name__}")
            print("No hardware changes made. Re-run with --execute to run the sweep.")
            return 0

        try:
            sweep = run_sweep(
                start_mhz=args.start,
                stop_mhz=args.stop,
                step_mhz=args.step,
                test_duration=args.duration,
                size_mb=args.size,
                gpu_index=gpu,
                execute=True,
                backend=selected,
                memory_backend=gpu_memory,
            )
        except Exception as exc:
            print(f"Sweep failed: {exc}", file=sys.stderr)
            return 1
        print(sweep.summary())
        return 0 if sweep.crash_offset_mhz < 0 else 1

    if get_gpu_memory_backend(gpu) is None:
        print(
            "CuPy/CUDA unavailable: this run tests system RAM only, not GPU VRAM.",
            file=sys.stderr,
        )
    started = time.time()
    errors = 0
    iterations = 0
    while time.time() - started < args.duration:
        tested = run_memtest(size_mb=args.size, gpu_index=gpu)
        print(tested.summary())
        errors += tested.errors_detected
        iterations += 1
        if tested.errors_detected:
            break
    print(f"Iterations={iterations} errors={errors}")
    return 0 if errors == 0 else 1
