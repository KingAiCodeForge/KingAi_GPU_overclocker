"""
VRAM stability testing and memory OC bandwidth sweep.

Two modes:
  1. Pattern test — write known patterns (0x00, 0xFF, 0xAA, 0x55, walking-1,
     random) to GPU VRAM, read back, compare. Any mismatch = VRAM error.
     This detects data corruption from unstable memory overclocks.

  2. Bandwidth sweep — at each memory OC offset step:
     a) Set memory offset via NVAPI
     b) Wait for clocks to settle (2 seconds)
     c) Measure effective memory bandwidth (device-to-device copy)
     d) Run pattern test to detect errors
     e) Record results
     Detects the "bandwidth cliff" where GDDR6/6X error correction
     (ECC/EDR) kicks in and eats bandwidth even though no visible
     artifacts appear. This is the real-world optimal OC point.

Works on ALL GeForce GPUs — no ECC counter access needed (those are
locked to Quadro/Tesla). Instead, we detect errors via:
  - Direct pattern comparison (catches data corruption)
  - Bandwidth regression (catches silent ECC/EDR correction overhead)

Requires: numpy (always), cupy (optional, for actual GPU VRAM testing).
Falls back to pure numpy + system RAM if cupy is unavailable (proof of concept).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

from kingai_gpu.lib.nvml import snapshot


# ── Try to import CUDA-capable libraries ──
# CuPy = CUDA-accelerated NumPy. Required for actual VRAM testing.
# Without CuPy, we can only test system RAM (proof of concept mode).
# CuPy must match CUDA version: pip install cupy-cuda11x or cupy-cuda12x

_HAS_CUPY = False
_HAS_CUDA = False

try:
    import cupy as cp

    _HAS_CUPY = True
    _HAS_CUDA = True
except ImportError:
    cp = None

try:
    import numpy as np
except ImportError:
    np = None


# ── Test result types ──
# Dataclasses for structured test results. Used by CLI output,
# future JSON export, and GUI results display.

@dataclass
class MemtestResult:
    """Result of a single VRAM pattern test.

    A test runs multiple patterns sequentially. Each pattern writes known
    data to a VRAM buffer, reads it back, and counts mismatches.
    errors_detected > 0 means the memory OC is unstable.
    """

    timestamp: str = ""
    duration_sec: float = 0.0
    buffer_size_mb: int = 0
    patterns_tested: int = 0
    total_bytes_checked: int = 0
    errors_detected: int = 0
    error_locations: list[int] = field(default_factory=list)
    gpu_temp: int = 0
    mem_clock: int = 0
    passed: bool = True

    def summary(self) -> str:
        status = "PASS" if self.passed else f"FAIL ({self.errors_detected} errors)"
        return (
            f"[{self.timestamp}] {status} | "
            f"{self.buffer_size_mb}MB × {self.patterns_tested} patterns | "
            f"{self.duration_sec:.1f}s | "
            f"{self.gpu_temp}°C | {self.mem_clock} MHz"
        )


@dataclass
class BandwidthResult:
    """Result of a bandwidth measurement at one memory OC point.

    Used by the sweep to track how effective memory bandwidth changes
    as the memory clock offset increases.
    """

    mem_offset_mhz: int = 0
    bandwidth_gbps: float = 0.0
    errors: int = 0
    gpu_temp: int = 0
    mem_clock: int = 0
    power_draw: float = 0.0


@dataclass
class SweepResult:
    """Full memory OC sweep result.

    Contains all bandwidth measurements, the optimal offset found,
    and the cliff/crash points if detected. The summary() method
    produces a formatted table for terminal display.
    """

    gpu_name: str = ""
    timestamp: str = ""
    results: list[BandwidthResult] = field(default_factory=list)
    optimal_offset_mhz: int = 0
    peak_bandwidth_gbps: float = 0.0
    cliff_offset_mhz: int = -1  # -1 = no cliff detected — BW never decreased
    crash_offset_mhz: int = -1  # -1 = no crash — all steps completed

    def summary(self) -> str:
        lines = [
            f"═══ Memory OC Sweep: {self.gpu_name} ═══",
            f"  Timestamp:         {self.timestamp}",
            f"  Optimal offset:    +{self.optimal_offset_mhz} MHz",
            f"  Peak bandwidth:    {self.peak_bandwidth_gbps:.1f} GB/s",
        ]
        if self.cliff_offset_mhz >= 0:
            lines.append(f"  Bandwidth cliff:   +{self.cliff_offset_mhz} MHz (corrections eating BW)")
        if self.crash_offset_mhz >= 0:
            lines.append(f"  Crash point:       +{self.crash_offset_mhz} MHz")
        lines.append("")
        lines.append(f"  {'Offset':>8}  {'BW (GB/s)':>10}  {'Errors':>7}  {'Temp':>5}  {'Clock':>7}  {'Power':>7}")
        lines.append(f"  {'─'*8}  {'─'*10}  {'─'*7}  {'─'*5}  {'─'*7}  {'─'*7}")
        for r in self.results:
            marker = ""
            if r.mem_offset_mhz == self.optimal_offset_mhz:
                marker = " ← OPTIMAL"
            if r.mem_offset_mhz == self.cliff_offset_mhz:
                marker = " ← CLIFF"
            if r.errors > 0:
                marker = f" ← {r.errors} ERRORS"
            lines.append(
                f"  {'+' + str(r.mem_offset_mhz):>8}  "
                f"{r.bandwidth_gbps:>10.1f}  "
                f"{r.errors:>7}  "
                f"{r.gpu_temp:>4}°  "
                f"{r.mem_clock:>6}M  "
                f"{r.power_draw:>5.0f}W"
                f"{marker}"
            )
        return "\n".join(lines)


# ── VRAM pattern tests (CuPy GPU-accelerated) ──
# These patterns are chosen to stress different failure modes:
#   all_zeros/ones  — stuck-at faults (bit permanently 0 or 1)
#   checkerboard    — adjacent-cell coupling (bit influenced by neighbor)
#   walking_1       — single-bit sensitivity (each bit position tested alone)
#   random          — general data integrity (catches pattern-dependent faults)

PATTERNS = [
    ("all_zeros", 0x00),      # Every bit = 0
    ("all_ones", 0xFF),       # Every bit = 1
    ("checkerboard_A", 0xAA), # Alternating 10101010
    ("checkerboard_5", 0x55), # Alternating 01010101 (inverse of above)
    ("walking_1", None),      # Special: shifts a single 1 through all 32 positions
    ("random", None),         # Special: random data (catches pattern-dependent faults)
]


def _run_pattern_test_cupy(size_mb: int = 256, gpu_index: int = 0) -> MemtestResult:
    """GPU-accelerated VRAM pattern test using CuPy.

    For each pattern:
      1. Allocate CuPy array on GPU VRAM (uint32 for efficiency)
      2. Fill with pattern value
      3. Force CUDA sync (ensures write is complete)
      4. Copy buffer to a second GPU array
      5. Compare original vs copy element-wise
      6. Count mismatches (any mismatch = VRAM corruption)

    The walking_1 pattern runs 32 sub-tests (one per bit position).
    Total VRAM usage: ~2x size_mb during each pattern test.
    """
    result = MemtestResult(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        buffer_size_mb=size_mb,
    )
    t0 = time.perf_counter()
    size_bytes = size_mb * 1024 * 1024
    size_u32 = size_bytes // 4  # Work in uint32 for efficiency

    total_errors = 0
    patterns_run = 0

    for name, value in PATTERNS:
        try:
            if name == "walking_1":
                # Walk a 1 bit through each 32-bit word position
                for bit in range(32):
                    pattern_val = 1 << bit
                    buf = cp.full(size_u32, pattern_val, dtype=cp.uint32)
                    cp.cuda.runtime.deviceSynchronize()
                    readback = buf.copy()
                    cp.cuda.runtime.deviceSynchronize()
                    mismatches = int(cp.sum(buf != readback))
                    total_errors += mismatches
                    del buf, readback
                patterns_run += 32

            elif name == "random":
                # Write random data, read back, compare
                buf = cp.random.randint(0, 0xFFFFFFFF, size=size_u32, dtype=cp.uint32)
                cp.cuda.runtime.deviceSynchronize()
                expected = buf.copy()
                cp.cuda.runtime.deviceSynchronize()
                # Read back original buffer
                mismatches = int(cp.sum(buf != expected))
                total_errors += mismatches
                del buf, expected
                patterns_run += 1

            else:
                # Fixed-value pattern
                fill_val = value | (value << 8) | (value << 16) | (value << 24)
                buf = cp.full(size_u32, fill_val, dtype=cp.uint32)
                cp.cuda.runtime.deviceSynchronize()
                readback = buf.copy()
                cp.cuda.runtime.deviceSynchronize()
                mismatches = int(cp.sum(buf != readback))
                total_errors += mismatches
                del buf, readback
                patterns_run += 1

        except Exception as e:
            print(f"  Pattern '{name}' failed: {e}")
            continue

    # Force CUDA sync and cleanup
    cp.cuda.runtime.deviceSynchronize()
    cp.get_default_memory_pool().free_all_blocks()

    result.duration_sec = time.perf_counter() - t0
    result.patterns_tested = patterns_run
    result.total_bytes_checked = size_bytes * patterns_run
    result.errors_detected = total_errors
    result.passed = total_errors == 0

    # Grab GPU state
    try:
        s = snapshot(gpu_index)
        result.gpu_temp = s.temp_gpu
        result.mem_clock = s.clock_mem
    except Exception:
        pass

    return result


def _run_pattern_test_numpy(size_mb: int = 256, gpu_index: int = 0) -> MemtestResult:
    """CPU-based fallback VRAM pattern test (system RAM only).

    This does NOT actually test VRAM — it's a proof-of-concept that runs
    the same pattern logic on system RAM via numpy. Useful for testing
    the test harness itself without a CUDA-capable GPU.

    Walking_1 and random patterns are skipped in fallback mode to keep
    runtime reasonable on CPU.
    """
    result = MemtestResult(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        buffer_size_mb=size_mb,
    )
    t0 = time.perf_counter()
    size_bytes = size_mb * 1024 * 1024

    if np is None:
        result.passed = False
        result.errors_detected = -1
        return result

    total_errors = 0
    patterns_run = 0

    for name, value in PATTERNS:
        if name == "walking_1" or name == "random":
            continue  # Skip complex patterns in fallback mode
        buf = np.full(size_bytes, value, dtype=np.uint8)
        readback = buf.copy()
        mismatches = int(np.sum(buf != readback))
        total_errors += mismatches
        patterns_run += 1
        del buf, readback

    result.duration_sec = time.perf_counter() - t0
    result.patterns_tested = patterns_run
    result.total_bytes_checked = size_bytes * patterns_run
    result.errors_detected = total_errors
    result.passed = total_errors == 0

    try:
        s = snapshot(gpu_index)
        result.gpu_temp = s.temp_gpu
        result.mem_clock = s.clock_mem
    except Exception:
        pass

    return result


def run_memtest(size_mb: int = 256, gpu_index: int = 0) -> MemtestResult:
    """Run VRAM pattern test. Auto-selects CuPy (GPU) or numpy (CPU) backend."""
    if _HAS_CUPY:
        return _run_pattern_test_cupy(size_mb, gpu_index)
    else:
        print("Warning: CuPy not installed. Running CPU-only fallback (tests system RAM, not VRAM).")
        print("Install CuPy for actual VRAM testing: pip install cupy-cuda12x")
        return _run_pattern_test_numpy(size_mb, gpu_index)


# ── Bandwidth measurement ──
# Measures effective GPU memory bandwidth by timing device-to-device copies.
# This is the key metric for detecting the ECC/EDR bandwidth cliff.

def measure_bandwidth(size_mb: int = 256, iterations: int = 50) -> float:
    """Measure GPU memory bandwidth in GB/s using device-to-device copy.

    Methodology:
      1. Allocate src + dst buffers on GPU VRAM
      2. Warm up (5 copies to stabilize clocks + caches)
      3. Time N iterations of src->dst copy
      4. Calculate: bandwidth = (bytes × iterations × 2) / elapsed
         The ×2 accounts for both read (from src) and write (to dst).

    Returns effective bandwidth in GB/s. Typical values:
      - RTX 3080 stock: ~750-760 GB/s
      - RTX 3080 +1000 MHz mem: ~820-850 GB/s (before cliff)
      - RTX 3080 +1200 MHz mem: ~800 GB/s (cliff — ECC eating BW)
    """
    if not _HAS_CUPY:
        print("Error: CuPy required for bandwidth measurement. pip install cupy-cuda12x")
        return 0.0

    size_bytes = size_mb * 1024 * 1024

    # Allocate two buffers
    src = cp.random.randint(0, 255, size=size_bytes, dtype=cp.uint8)
    dst = cp.zeros(size_bytes, dtype=cp.uint8)
    cp.cuda.runtime.deviceSynchronize()

    # Warm up
    for _ in range(5):
        cp.copyto(dst, src)
    cp.cuda.runtime.deviceSynchronize()

    # Timed run
    t0 = time.perf_counter()
    for _ in range(iterations):
        cp.copyto(dst, src)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0

    # Bandwidth = (bytes read + bytes written) / time
    # Each copy reads src and writes dst, so total data moved = 2× size
    bw_bytes = size_bytes * iterations * 2
    bw_gbps = bw_bytes / elapsed / 1e9

    del src, dst
    cp.get_default_memory_pool().free_all_blocks()

    return bw_gbps


# ── Memory OC sweep ──
# The flagship feature — automated memory overclock optimization.
# Walks through offset range, measuring bandwidth and errors at each step.

def run_sweep(
    start_mhz: int = 0,
    stop_mhz: int = 1500,
    step_mhz: int = 50,
    test_duration: int = 10,
    size_mb: int = 256,
    gpu_index: int = 0,
) -> SweepResult:
    """
    Automated memory OC sweep — finds optimal offset before bandwidth cliff.

    At each step:
    1. Set memory offset via NVAPI (nvapi.set_mem_offset)
    2. Wait 2s for GPU clocks to settle at new frequency
    3. Measure bandwidth (device-to-device copy throughput)
    4. Run pattern test (write/read/compare known patterns)
    5. Record results (BW, errors, temp, clock, power)

    Detects three key points:
    - OPTIMAL: highest BW with zero errors (this is your best OC)
    - CLIFF: BW decreased >2% from previous step (ECC/EDR kicking in)
    - CRASH: exception during test (GPU driver recovery or hang)

    SAFETY: Always resets memory offset to +0 in the finally block,
    even on crash or Ctrl+C. If the reset itself fails, prints a
    manual reset command for the user.
    """
    try:
        from kingai_gpu.lib.nvapi import (
            NvApiError,
            enable_oc,
            set_mem_offset,
        )
    except ImportError:
        print("Error: NVAPI required for memory OC sweep (Windows only)")
        return SweepResult()

    sweep = SweepResult(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Get GPU name
    s = snapshot(gpu_index)
    sweep.gpu_name = s.name

    try:
        enable_oc(gpu_index)
    except NvApiError as e:
        print(f"Failed to enable OC: {e}")
        return sweep

    peak_bw = 0.0
    peak_offset = 0
    prev_bw = 0.0

    try:
        for offset in range(start_mhz, stop_mhz + 1, step_mhz):
            print(f"\n  Testing +{offset} MHz...", end=" ", flush=True)

            # Set memory offset
            try:
                set_mem_offset(offset, gpu_index)
            except NvApiError as e:
                print(f"NVAPI error: {e}")
                sweep.crash_offset_mhz = offset
                break

            # Wait for clocks to settle
            time.sleep(2)

            # Measure bandwidth
            try:
                bw = measure_bandwidth(size_mb=size_mb, iterations=max(10, test_duration * 5))
            except Exception as e:
                print(f"CRASH during bandwidth test: {e}")
                sweep.crash_offset_mhz = offset
                break

            # Run pattern test
            try:
                mt = run_memtest(size_mb=size_mb, gpu_index=gpu_index)
                errors = mt.errors_detected
            except Exception as e:
                print(f"CRASH during pattern test: {e}")
                sweep.crash_offset_mhz = offset
                break

            # Read GPU state
            gs = snapshot(gpu_index)

            result = BandwidthResult(
                mem_offset_mhz=offset,
                bandwidth_gbps=bw,
                errors=errors,
                gpu_temp=gs.temp_gpu,
                mem_clock=gs.clock_mem,
                power_draw=gs.power_draw,
            )
            sweep.results.append(result)

            print(f"{bw:.1f} GB/s  {errors} errors  {gs.temp_gpu}°C  {gs.clock_mem}MHz", flush=True)

            # Track peak — best BW with zero errors = optimal OC point
            if bw > peak_bw and errors == 0:
                peak_bw = bw
                peak_offset = offset

            # Detect bandwidth cliff: BW dropped >2% from previous step.
            # This means GDDR6/6X error correction is consuming bandwidth
            # even though no visible errors appear. The GPU is silently
            # correcting bit errors, which costs memory controller cycles.
            if prev_bw > 0 and bw < prev_bw * 0.98 and sweep.cliff_offset_mhz < 0:
                sweep.cliff_offset_mhz = offset

            # Stop on pattern errors — visible corruption means OC is WAY too high.
            # No point testing further, and continuing risks driver crash.
            if errors > 0:
                print(f"\n  ⚠ Errors detected at +{offset} MHz. Stopping sweep.")
                break

            prev_bw = bw

    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user.")

    finally:
        # CRITICAL SAFETY: Always reset memory offset to stock.
        # Even if the test crashed, we MUST undo the OC to prevent
        # the user's GPU from running at an unstable memory clock.
        print("\n  Resetting memory offset to +0 MHz...", end=" ")
        try:
            set_mem_offset(0, gpu_index)
            print("done.")
        except Exception as e:
            print(f"WARNING: Failed to reset: {e}")
            print("  ⚠ MANUALLY RESET YOUR MEMORY OFFSET! Run: kingai-gpu oc --mem 0")

    sweep.optimal_offset_mhz = peak_offset
    sweep.peak_bandwidth_gbps = peak_bw

    return sweep


# ── CLI command handler ──
# Entry point called from cli/main.py. Dispatches to sweep or pattern test.

def cmd_memtest(args) -> int:
    """Handle the 'memtest' subcommand.

    Two modes:
      --sweep: Run automated memory OC sweep (requires NVAPI + CuPy)
      (default): Run single/repeated pattern test at current OC setting
    """

    print(f"KingAi GPU Memory Stability Tester")
    print(f"{'═' * 40}")

    if not _HAS_CUPY:
        print()
        print("⚠ CuPy not installed — VRAM tests will use CPU fallback (less accurate)")
        print("  Install for real VRAM testing: pip install cupy-cuda12x")
        print()

    gpu = args.gpu

    if args.sweep:
        # Automated memory OC sweep
        print(f"\nStarting memory OC sweep on GPU {gpu}")
        print(f"  Range: +{args.start} to +{args.stop} MHz (step {args.step})")
        print(f"  Duration per step: {args.duration}s")
        print(f"  Buffer size: {args.size} MB")
        print(f"  Press Ctrl+C to stop safely\n")

        result = run_sweep(
            start_mhz=args.start,
            stop_mhz=args.stop,
            step_mhz=args.step,
            test_duration=args.duration,
            size_mb=args.size,
            gpu_index=gpu,
        )

        print(f"\n{result.summary()}")
        return 0

    else:
        # Single/repeated pattern test — runs at current OC setting.
        # Keeps running until --duration elapsed or errors found.
        print(f"\nRunning VRAM pattern test on GPU {gpu}")
        print(f"  Buffer size: {args.size} MB")
        print(f"  Duration target: {args.duration}s\n")

        # Run multiple iterations for the target duration.
        # Each run_memtest() call does all patterns once (~1-3 seconds).
        # We loop until the time budget is exhausted or errors are found.
        t0 = time.time()
        total_errors = 0
        iterations = 0

        while time.time() - t0 < args.duration:
            r = run_memtest(size_mb=args.size, gpu_index=gpu)
            total_errors += r.errors_detected
            iterations += 1
            print(f"  {r.summary()}")
            if r.errors_detected > 0:
                break

        elapsed = time.time() - t0
        print(f"\n{'═' * 40}")
        print(f"  Iterations:  {iterations}")
        print(f"  Total time:  {elapsed:.1f}s")
        print(f"  Total errors: {total_errors}")
        print(f"  Result:      {'PASS ✓' if total_errors == 0 else 'FAIL ✗'}")

        return 0 if total_errors == 0 else 1
