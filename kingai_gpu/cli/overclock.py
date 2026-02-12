"""
CLI overclock — set core/memory offsets, power/thermal limits, fan speed.

All changes go through NVAPI (Windows-only, requires NVIDIA drivers).
NVAPI is imported lazily inside cmd_overclock() so this module can be
imported on Linux without error — it just fails at runtime when called.

Command behavior:
  - No flags given + no --status: show current OC status (read-only)
  - One or more Set flags: apply settings to GPU, then confirm
  - --reset: restore everything to factory defaults
  - --save PATH: snapshot current OC settings to a JSON profile
  - --load PATH: apply OC settings from a saved JSON profile
  - --status: explicitly show OC status (redundant with no-flags)

Each Set operation is independent — you can set core + power in one call
without affecting memory or fan settings.

Profile format (JSON):
  {
    "gpu_name": "GeForce RTX 3080",
    "core_offset_mhz": 150,
    "mem_offset_mhz": 500,
    "power_pct": 110,
    "thermal_c": 85,
    "fan_pct": 70           ← null or absent = don't change fan
  }
  Fields are all optional — omitted fields are not applied on load.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _save_profile(path_str: str, status) -> str:
    """Save current OC status to a JSON profile file.

    Only saves the settable values (offsets, limits, fan speed).
    Range and metadata fields are stored for reference but ignored on load.
    Returns the resolved absolute path of the saved file.
    """
    profile = {
        "kingai_gpu_profile": "1.0",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "gpu_name": status.gpu_name,
        "bus_id": status.bus_id,
        # ── Settable values (these get applied on --load) ──
        "core_offset_mhz": int(status.core_offset_mhz),
        "mem_offset_mhz": int(status.mem_offset_mhz),
        "power_pct": round(status.power_pct, 1),
        "thermal_c": status.thermal_c,
        "fan_pct": status.fan_pct,  # None = auto, 0-100 = manual
        # ── Reference only (not applied on load) ──
        "ranges": {
            "core_mhz": list(status.core_offset_range_mhz),
            "mem_mhz": list(status.mem_offset_range_mhz),
            "power_pct": list(status.power_range_pct),
            "thermal_c": list(status.thermal_range_c),
        },
    }
    p = Path(path_str).resolve()
    # Default to .json extension if none provided
    if p.suffix == "":
        p = p.with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return str(p)


def _load_profile(path_str: str) -> dict:
    """Load a JSON profile file and return the settings dict.

    Validates the file exists and is valid JSON. Does NOT validate values
    against the current GPU's ranges — that happens when we call the Set
    functions (NVAPI will reject out-of-range values with an error).
    """
    p = Path(path_str).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Profile not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid profile: expected JSON object, got {type(data).__name__}")
    return data


def cmd_overclock(args) -> int:
    """Handle the 'oc' subcommand."""

    try:
        from kingai_gpu.lib.nvapi import (
            NvApiError,
            enable_oc,
            get_oc_status,
            reset_all,
            set_core_offset,
            set_fan_auto,
            set_fan_speed,
            set_mem_offset,
            set_power_limit,
            set_thermal_limit,
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("Overclock control requires Windows with NVIDIA drivers.")
        return 1

    gpu = args.gpu

    # Initialize NVAPI and enable OC
    try:
        enable_oc(gpu)
    except NvApiError as e:
        print(f"Failed to initialize NVAPI OC: {e}")
        return 1

    # ── Reset all ── Restores every parameter to stock defaults.
    # This is the "panic button" — undoes all overclocking.
    # reset_all() in nvapi.py handles each subsystem independently.
    if args.reset:
        print(f"Resetting GPU {gpu} to stock defaults...")
        try:
            reset_all(gpu)
            print("  Core offset:   0 MHz")
            print("  Memory offset: 0 MHz")
            print("  Power limit:   default")
            print("  Thermal limit: default")
            print("  Fan:           auto")
            print("Done.")
        except NvApiError as e:
            print(f"Error during reset: {e}")
            return 1
        return 0

    # ── Save profile ── Snapshot current OC settings to a JSON file.
    # Reads the live OcStatus and writes the settable values.
    if args.save:
        try:
            s = get_oc_status(gpu)
        except NvApiError as e:
            print(f"Error reading OC status for save: {e}")
            return 1
        try:
            saved_path = _save_profile(args.save, s)
            print(f"Saved profile to {saved_path}")
            print(f"  Core offset:   {s.core_offset_mhz:+.0f} MHz")
            print(f"  Memory offset: {s.mem_offset_mhz:+.0f} MHz")
            print(f"  Power limit:   {s.power_pct:.0f}%")
            print(f"  Thermal limit: {s.thermal_c}°C")
            if s.fan_pct is not None:
                print(f"  Fan speed:     {s.fan_pct}%")
            else:
                print(f"  Fan:           auto (not saved)")
        except (OSError, ValueError) as e:
            print(f"Failed to save profile: {e}")
            return 1
        return 0

    # ── Load profile ── Apply OC settings from a JSON file.
    # Only applies fields that are present in the profile. Missing fields
    # are left unchanged (not reset). This lets you share partial profiles
    # (e.g., just memory offset + fan speed for a memtest sweep).
    if args.load:
        try:
            profile = _load_profile(args.load)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"Failed to load profile: {e}")
            return 1

        # Show what we're about to apply
        src = profile.get("gpu_name", "unknown GPU")
        print(f"Loading profile (from {src})...")

        changes = []
        if "core_offset_mhz" in profile:
            val = int(profile["core_offset_mhz"])
            try:
                set_core_offset(val, gpu)
                changes.append(f"  Core offset:   {val:+d} MHz")
            except NvApiError as e:
                print(f"Failed to set core offset: {e}")
                return 1

        if "mem_offset_mhz" in profile:
            val = int(profile["mem_offset_mhz"])
            try:
                set_mem_offset(val, gpu)
                changes.append(f"  Memory offset: {val:+d} MHz")
            except NvApiError as e:
                print(f"Failed to set memory offset: {e}")
                return 1

        if "power_pct" in profile:
            val = profile["power_pct"]
            try:
                set_power_limit(val, gpu)
                changes.append(f"  Power limit:   {val}%")
            except NvApiError as e:
                print(f"Failed to set power limit: {e}")
                return 1

        if "thermal_c" in profile:
            val = int(profile["thermal_c"])
            try:
                set_thermal_limit(val, gpu)
                changes.append(f"  Thermal limit: {val}°C")
            except NvApiError as e:
                print(f"Failed to set thermal limit: {e}")
                return 1

        fan = profile.get("fan_pct")
        if fan is not None:
            try:
                set_fan_speed(int(fan), gpu)
                changes.append(f"  Fan speed:     {fan}%")
            except NvApiError as e:
                print(f"Failed to set fan speed: {e}")
                return 1

        if changes:
            print(f"Applied to GPU {gpu}:")
            for c in changes:
                print(c)
        else:
            print("Profile had no applicable settings.")
        return 0

    # ── Show status ── Read-only view of all current OC parameters.
    # Shown by default when no Set flags are given, or explicitly with --status.
    if args.status or (
        args.core is None
        and args.mem is None
        and args.power is None
        and args.thermal is None
        and args.fan is None
        and not args.fan_auto
        and not args.save
        and not args.load
    ):
        try:
            s = get_oc_status(gpu)
        except NvApiError as e:
            print(f"Error reading OC status: {e}")
            return 1

        print(f"═══ GPU {gpu}: {s.gpu_name} (Bus {s.bus_id}) ═══")
        print(f"  Core offset:   {s.core_offset_mhz:+.0f} MHz  "
              f"(range: {s.core_offset_range_mhz[0]:+.0f} to {s.core_offset_range_mhz[1]:+.0f})")
        print(f"  Memory offset: {s.mem_offset_mhz:+.0f} MHz  "
              f"(range: {s.mem_offset_range_mhz[0]:+.0f} to {s.mem_offset_range_mhz[1]:+.0f})")
        print(f"  Power limit:   {s.power_pct:.0f}%  "
              f"(range: {s.power_range_pct[0]:.0f}% to {s.power_range_pct[1]:.0f}%)")
        print(f"  Thermal limit: {s.thermal_c}°C  "
              f"(range: {s.thermal_range_c[0]}°C to {s.thermal_range_c[1]}°C)")
        return 0

    # ── Apply settings ── Each flag is applied independently.
    # If one fails, we return immediately with error (don't apply remaining).
    # This is intentional — partial application could leave the GPU in an
    # unexpected state. The user can re-run with correct values.
    changes = []

    if args.core is not None:
        try:
            set_core_offset(args.core, gpu)
            changes.append(f"  Core offset:   {args.core:+d} MHz")
        except NvApiError as e:
            print(f"Failed to set core offset: {e}")
            return 1

    if args.mem is not None:
        try:
            set_mem_offset(args.mem, gpu)
            changes.append(f"  Memory offset: {args.mem:+d} MHz")
        except NvApiError as e:
            print(f"Failed to set memory offset: {e}")
            return 1

    if args.power is not None:
        try:
            set_power_limit(args.power, gpu)
            changes.append(f"  Power limit:   {args.power}%")
        except NvApiError as e:
            print(f"Failed to set power limit: {e}")
            return 1

    if args.thermal is not None:
        try:
            set_thermal_limit(args.thermal, gpu)
            changes.append(f"  Thermal limit: {args.thermal}°C")
        except NvApiError as e:
            print(f"Failed to set thermal limit: {e}")
            return 1

    if args.fan is not None:
        try:
            set_fan_speed(args.fan, gpu)
            changes.append(f"  Fan speed:     {args.fan}%")
        except NvApiError as e:
            print(f"Failed to set fan speed: {e}")
            return 1

    if args.fan_auto:
        try:
            set_fan_auto(gpu)
            changes.append(f"  Fan:           auto")
        except NvApiError as e:
            print(f"Failed to set fan auto: {e}")
            return 1

    if changes:
        print(f"Applied to GPU {gpu}:")
        for c in changes:
            print(c)

    return 0
