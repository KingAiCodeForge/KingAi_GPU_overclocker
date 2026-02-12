"""
CLI monitor — real-time GPU monitoring dashboard.

Renders NVML sensor data in multiple output formats:
  (default)   Full box-drawing dashboard with ASCII progress bars
  --once      Single snapshot, then exit (useful for scripting)
  --json      JSON output (pipe to jq, log to file, or feed to other tools)
  --csv       CSV output (append to file for time-series analysis)
  --compact   Single-line refreshing display (for narrow terminals or tmux)

All formats read from the same GpuSnapshot dataclass (lib/nvml.py).
The dashboard uses ANSI color codes for temperature/throttle highlighting.

This module is cross-platform — works on both Windows and Linux.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime

from kingai_gpu.lib.nvml import GpuSnapshot, gpu_count, snapshot, snapshot_all


# ── Formatting helpers ──
# These produce ASCII/ANSI-styled text for the terminal dashboard.
# All handle edge cases (zero max, missing data) gracefully.

def _bar(value: float, max_val: float, width: int = 30) -> str:
    """ASCII progress bar using block characters. █=filled, ░=empty.

    Clamps to [0, max_val] — won't overflow if value > max.
    Returns empty spaces if max_val is 0 or negative (avoid division by zero).
    """
    if max_val <= 0:
        return " " * width
    pct = min(value / max_val, 1.0)
    filled = int(pct * width)
    return "█" * filled + "░" * (width - filled)


def _color_temp(temp: int) -> str:
    """ANSI color code for temperature value.

    Green (<50°C) = cool/idle
    Yellow (50-69°C) = warm/light load
    Red (70-84°C) = hot/heavy load
    Magenta (85+°C) = danger zone / thermal throttling likely
    """
    if temp < 50:
        return f"\033[92m{temp}°C\033[0m"  # green
    elif temp < 70:
        return f"\033[93m{temp}°C\033[0m"  # yellow
    elif temp < 85:
        return f"\033[91m{temp}°C\033[0m"  # red
    else:
        return f"\033[95m{temp}°C\033[0m"  # magenta (danger)


def _color_throttle(reasons: list[str]) -> str:
    """Color-code throttle reasons — green if idle/none, red if actually throttling."""
    if reasons == ["NONE"] or reasons == ["GPU_IDLE"]:
        return f"\033[92m{'  '.join(reasons)}\033[0m"
    return f"\033[91m{'  '.join(reasons)}\033[0m"


def _clear_screen():
    """Clear terminal screen."""
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


# ── Dashboard render ──
# Box-drawing characters create a structured display that's easy to read.
# Fixed 70-char width fits most terminals. Future: detect terminal width.

def render_dashboard(s: GpuSnapshot) -> str:
    """Render a full monitoring dashboard for one GPU.

    Uses Unicode box-drawing characters (┌─┐│├┤└┘) for structure.
    Each section shows a metric + ASCII progress bar for visual scanning.
    """
    ts = datetime.fromtimestamp(s.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    w = 70  # box width

    lines = []
    lines.append(f"┌{'─' * w}┐")
    lines.append(f"│{'KingAi GPU Monitor':^{w}}│")
    lines.append(f"│{ts:^{w}}│")
    lines.append(f"├{'─' * w}┤")

    # Identity
    lines.append(f"│  GPU {s.index}: {s.name:<{w - 10}}│")
    lines.append(f"│  Driver: {s.driver_version}  │  PCI: {s.pci_bus_id:<{w - 25}}│")
    lines.append(f"├{'─' * w}┤")

    # Clocks
    gpu_bar = _bar(s.clock_gpu, s.clock_gpu_max, 25)
    mem_bar = _bar(s.clock_mem, s.clock_mem_max, 25)
    lines.append(f"│  Core Clock:   {s.clock_gpu:>5} / {s.clock_gpu_max} MHz  {gpu_bar}  │")
    lines.append(f"│  Mem  Clock:   {s.clock_mem:>5} / {s.clock_mem_max} MHz  {mem_bar}  │")
    lines.append(f"│  SM Clock:     {s.clock_sm:>5} MHz                                  │")
    lines.append(f"├{'─' * w}┤")

    # Temp + Fan
    temp_bar = _bar(s.temp_gpu, s.temp_gpu_max or 100, 25)
    fan_bar = _bar(s.fan_speed, 100, 25)
    lines.append(f"│  Temperature:  {s.temp_gpu:>5} / {s.temp_gpu_max or '?':>3}°C     {temp_bar}  │")
    lines.append(f"│  Fan Speed:    {s.fan_speed:>5}%              {fan_bar}  │")
    lines.append(f"├{'─' * w}┤")

    # Power
    pwr_bar = _bar(s.power_draw, s.power_limit, 25)
    lines.append(
        f"│  Power:        {s.power_draw:>5.0f} / {s.power_limit:.0f}W "
        f"({s.power_pct:.0f}%)    {pwr_bar}  │"
    )
    lines.append(
        f"│  Limits:       {s.power_min:.0f}W min  /  "
        f"{s.power_default:.0f}W default  /  {s.power_max:.0f}W max          │"
    )
    lines.append(f"├{'─' * w}┤")

    # VRAM
    vram_bar = _bar(s.vram_used, s.vram_total, 25)
    lines.append(
        f"│  VRAM:         {s.vram_used:>5} / {s.vram_total} MB "
        f"({s.vram_used_pct:.0f}%)  {vram_bar}  │"
    )
    lines.append(f"├{'─' * w}┤")

    # Utilization
    gpu_util_bar = _bar(s.util_gpu, 100, 25)
    mem_util_bar = _bar(s.util_mem, 100, 25)
    lines.append(f"│  GPU Util:     {s.util_gpu:>5}%              {gpu_util_bar}  │")
    lines.append(f"│  Mem Util:     {s.util_mem:>5}%              {mem_util_bar}  │")
    lines.append(f"├{'─' * w}┤")

    # State
    throttle_str = "  ".join(s.throttle_reasons)
    lines.append(f"│  P-State:      {s.pstate:<55}│")
    lines.append(f"│  Throttle:     {throttle_str:<55}│")
    lines.append(f"└{'─' * w}┘")
    lines.append("  Press Ctrl+C to exit")

    return "\n".join(lines)


# ── Output modes ──
# Each function converts a GpuSnapshot to a different string format.
# The choice of format is made in cmd_monitor() based on CLI flags.

def output_json(s: GpuSnapshot) -> str:
    d = asdict(s)
    return json.dumps(d, indent=2)


def output_csv_header() -> str:
    return (
        "timestamp,name,temp_gpu,clock_gpu,clock_mem,power_draw,power_limit,"
        "fan_speed,util_gpu,util_mem,vram_used,vram_total,pstate,throttle"
    )


def output_csv_row(s: GpuSnapshot) -> str:
    ts = datetime.fromtimestamp(s.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    throttle = "|".join(s.throttle_reasons)
    return (
        f"{ts},{s.name},{s.temp_gpu},{s.clock_gpu},{s.clock_mem},{s.power_draw:.1f},"
        f"{s.power_limit:.1f},{s.fan_speed},{s.util_gpu},{s.util_mem},{s.vram_used},"
        f"{s.vram_total},{s.pstate},{throttle}"
    )


# ── Commands ──
# These are the handler functions called from cli/main.py via deferred import.

def cmd_monitor(args) -> int:
    """Main monitor command handler.

    Dispatches to single-shot or continuous mode based on --once flag.
    In continuous mode, refreshes the screen at --interval rate until Ctrl+C.
    """

    # Single snapshot — read once, output, exit. Good for scripting:
    #   kingai-gpu monitor --once --json | jq .temp_gpu
    if args.once:
        s = snapshot(args.gpu)
        if args.json:
            print(output_json(s))
        elif args.csv:
            print(output_csv_header())
            print(output_csv_row(s))
        elif args.compact:
            print(s.summary_line())
        else:
            print(render_dashboard(s))
        return 0

    # Continuous monitoring — loop until Ctrl+C.
    # Each format handles its own output differently:
    #   dashboard = clear screen + redraw (full refresh)
    #   csv = append lines (header printed once)
    #   compact = \r overwrite (single line)
    #   json = print object + separator
    try:
        first = True
        while True:
            s = snapshot(args.gpu)

            if args.json:
                print(output_json(s))
                print("---")
            elif args.csv:
                if first:
                    print(output_csv_header())
                    first = False
                print(output_csv_row(s))
                sys.stdout.flush()
            elif args.compact:
                ts = datetime.fromtimestamp(s.timestamp).strftime("%H:%M:%S")
                sys.stdout.write(f"\r{ts} {s.summary_line()}")
                sys.stdout.flush()
            else:
                _clear_screen()
                print(render_dashboard(s))

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        return 0


def cmd_info(_args) -> int:
    """Print detailed GPU info for all GPUs."""
    count = gpu_count()
    print(f"Found {count} NVIDIA GPU(s)\n")

    for i in range(count):
        s = snapshot(i)
        print(f"═══ GPU {i}: {s.name} ═══")
        print(f"  Driver:        {s.driver_version}")
        print(f"  PCI Bus:       {s.pci_bus_id}")
        print(f"  UUID:          {s.uuid}")
        print(f"  VRAM:          {s.vram_total} MB")
        print(f"  Max Core:      {s.clock_gpu_max} MHz")
        print(f"  Max Mem:       {s.clock_mem_max} MHz")
        print(f"  Temp Max:      {s.temp_gpu_max}°C")
        print(f"  Power Default: {s.power_default:.0f}W")
        print(f"  Power Range:   {s.power_min:.0f}W – {s.power_max:.0f}W")
        print()
    return 0
