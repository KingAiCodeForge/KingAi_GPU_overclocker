"""
Import a single MSI Afterburner profile and apply it via NVAPI.

MSI Afterburner stores OC profiles in per-GPU .cfg files (INI format).
Each file has sections: [Startup], [Profile1]–[Profile5], [PreSuspendedMode].
This tool reads a specified section, converts the values to our format,
and either applies them directly or saves as a KingAi JSON profile.

Supported fields (everything else in the .cfg is ignored):
  CoreClkBoost  → core_offset_mhz   (kHz in file, ÷1000 for MHz)
  MemClkBoost   → mem_offset_mhz    (kHz in file, ÷1000 for MHz)
  PowerLimit    → power_pct          (% in file, direct)
  ThermalLimit  → thermal_c          (°C in file, direct)
  FanSpeed      → fan_pct            (% in file, direct)
  FanMode       → auto vs manual     (0=auto, 1=manual)

VF curve data (VFCurve key) is NOT imported — NVAPI doesn't expose the
same per-point VF curve interface that Afterburner uses internally. Use
the MSI Afterburner Tools repo for VF curve manipulation.

Usage (from CLI):
  kingai-gpu import-msi  path/to/VEN_10DE...cfg                # apply Startup
  kingai-gpu import-msi  path/to/VEN_10DE...cfg --section Profile3
  kingai-gpu import-msi  path/to/VEN_10DE...cfg --save out.json  # convert only
  kingai-gpu import-msi  path/to/VEN_10DE...cfg --list           # list sections

The .cfg path is the per-GPU config, NOT MSIAfterburner.cfg (global).
Per-GPU configs are named like: VEN_10DE&DEV_2684&SUBSYS_...&BUS_01&....cfg
Find them in: C:\\Program Files (x86)\\MSI Afterburner\\Profiles\\
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


# ── Auto-detect Afterburner install ─────────────────────────────────────
# MSI Afterburner installs to Program Files (x86) by default, but can be
# anywhere. We check: 1) standard path, 2) registry, 3) common alternatives.
# Per-GPU configs are in the Profiles/ subfolder, named VEN_xxxx&DEV_...cfg.
#
# NVIDIA GPUs use VEN_10DE, AMD GPUs use VEN_1002.
# TODO: AMD support — our NVAPI backend only handles NVIDIA. When ADL/ADLX
#       support is added, AMD profiles (VEN_1002) can be imported too.

_STANDARD_PATHS = [
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
    / "MSI Afterburner" / "Profiles",
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    / "MSI Afterburner" / "Profiles",
]


def find_afterburner_profiles() -> list[Path]:
    """Auto-detect MSI Afterburner Profiles folder and return per-GPU .cfg files.

    Checks standard install paths, then falls back to registry lookup.
    Returns a list of Path objects for per-GPU configs (VEN_* pattern).
    """
    profiles_dir = None

    # Check standard paths
    for p in _STANDARD_PATHS:
        if p.is_dir():
            profiles_dir = p
            break

    # Try registry if standard paths don't exist
    if profiles_dir is None:
        profiles_dir = _find_via_registry()

    if profiles_dir is None:
        return []

    # Per-GPU configs match VEN_xxxx&DEV_...cfg pattern
    # VEN_10DE = NVIDIA, VEN_1002 = AMD
    cfgs = sorted(profiles_dir.glob("VEN_*.cfg"))
    return cfgs


def _find_via_registry() -> Path | None:
    """Try to find Afterburner install path from Windows registry."""
    try:
        import winreg
        # MSI Afterburner registers under HKLM\SOFTWARE\WOW6432Node\...
        for key_path in [
            r"SOFTWARE\WOW6432Node\MSI\Afterburner",
            r"SOFTWARE\MSI\Afterburner",
        ]:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                    install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                    profiles = Path(install_path) / "Profiles"
                    if profiles.is_dir():
                        return profiles
            except (FileNotFoundError, OSError):
                continue
    except ImportError:
        pass  # Not on Windows
    return None


# ── INI parsing ─────────────────────────────────────────────────────────
# MSI AB configs are Windows INI files: [Section] headers, Key=Value lines.
# We don't use configparser because AB files have duplicate keys and
# hex blobs that confuse Python's INI parser. Simple split is safer.

def _parse_sections(text: str) -> dict[str, str]:
    """Split INI text into {section_name: raw_body} dict."""
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current is not None:
                sections[current] = "\n".join(lines)
            current = stripped[1:-1]
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines)
    return sections


def _get_value(block: str, key: str) -> str | None:
    """Read a key=value from an INI section body. Returns None if missing/empty."""
    for line in block.splitlines():
        if line.strip().startswith(key + "="):
            val = line.split("=", 1)[1].strip()
            return val if val else None
    return None


# ── Profile extraction ──────────────────────────────────────────────────

# These are the only keys we can map to NVAPI Set operations.
# Everything else in the .cfg (VFCurve, shader clocks, voltage rails,
# monitoring sources, OSD layout) is AB-specific and gets ignored.
_FIELD_MAP = {
    # AB key          → (our key,         conversion)
    "CoreClkBoost":   ("core_offset_mhz", lambda v: int(v) // 1000),
    "MemClkBoost":    ("mem_offset_mhz",  lambda v: int(v) // 1000),
    "PowerLimit":     ("power_pct",       lambda v: int(v)),
    "ThermalLimit":   ("thermal_c",       lambda v: int(v)),
    "FanSpeed":       ("fan_pct",         lambda v: int(v)),
}


def extract_profile(cfg_path: str, section: str = "Startup") -> dict:
    """Read a .cfg file and extract OC settings from the given section.

    Returns a dict in KingAi profile format (same as --save produces):
      {
        "kingai_gpu_profile": "1.0",
        "imported_from": "msi_afterburner",
        "source_file": "...",
        "source_section": "Startup",
        "core_offset_mhz": 150,
        "mem_offset_mhz": 500,
        ...
      }
    """
    p = Path(cfg_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    # AB configs are ASCII with CRLF line endings
    text = p.read_text(encoding="ascii", errors="replace")
    sections = _parse_sections(text)

    if section not in sections:
        available = [s for s in sections if s not in ("Defaults", "Settings")]
        raise KeyError(
            f"Section [{section}] not found in {p.name}. "
            f"Available: {', '.join(available)}"
        )

    block = sections[section]

    # Build profile dict from mapped fields
    profile: dict = {
        "kingai_gpu_profile": "1.0",
        "imported_from": "msi_afterburner",
        "imported_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": p.name,
        "source_section": section,
    }

    for ab_key, (our_key, convert) in _FIELD_MAP.items():
        raw = _get_value(block, ab_key)
        if raw is not None:
            try:
                profile[our_key] = convert(raw)
            except (ValueError, TypeError):
                pass  # skip unparseable values silently

    # Fan mode: if FanMode=0 (auto), don't include fan_pct
    # (it would override the user's auto curve with a fixed speed)
    fan_mode = _get_value(block, "FanMode")
    if fan_mode == "0":
        profile.pop("fan_pct", None)
        profile["fan_auto"] = True

    # Voltage boost (informational — we don't set this via NVAPI currently)
    vboost = _get_value(block, "CoreVoltageBoost")
    if vboost and int(vboost) != 0:
        profile["_voltage_boost_mv"] = int(vboost)
        profile["_note_voltage"] = "Voltage offset not applied (not yet supported)"

    return profile


def list_sections(cfg_path: str) -> list[dict]:
    """List all profile sections in a .cfg file with their key values.

    Returns a list of dicts: [{"section": "Startup", "core": "+0", ...}, ...]
    """
    p = Path(cfg_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    text = p.read_text(encoding="ascii", errors="replace")
    sections = _parse_sections(text)

    results = []
    # Only show sections that contain OC data (skip Defaults, Settings)
    for name, block in sections.items():
        if name in ("Defaults", "Settings"):
            continue
        core_raw = _get_value(block, "CoreClkBoost")
        mem_raw = _get_value(block, "MemClkBoost")
        power = _get_value(block, "PowerLimit")
        thermal = _get_value(block, "ThermalLimit")
        fan_mode = _get_value(block, "FanMode")
        fan_spd = _get_value(block, "FanSpeed")

        core_mhz = f"{int(core_raw) // 1000:+d}" if core_raw else "N/A"
        mem_mhz = f"{int(mem_raw) // 1000:+d}" if mem_raw else "N/A"
        fan_label = f"{fan_spd}%" if fan_mode == "1" else "auto"

        results.append({
            "section": name,
            "core": core_mhz,
            "mem": mem_mhz,
            "power": f"{power}%" if power else "N/A",
            "thermal": f"{thermal}°C" if thermal else "N/A",
            "fan": fan_label,
        })
    return results


# ── CLI handler ─────────────────────────────────────────────────────────

def cmd_import_msi(args) -> int:
    """Handle the 'import-msi' subcommand."""

    cfg_path = args.cfg

    # ── Auto-detect mode ──
    # If --auto is given (or no cfg path provided), find AB's Profiles folder
    # and discover per-GPU config files automatically.
    if args.auto or cfg_path is None:
        found = find_afterburner_profiles()
        if not found:
            print("Could not find MSI Afterburner Profiles folder.")
            print("Checked:")
            for p in _STANDARD_PATHS:
                print(f"  {p}")
            print("  Windows registry (HKLM\\SOFTWARE\\MSI\\Afterburner)")
            print("\nProvide the .cfg path manually instead:")
            print("  kingai-gpu import-msi path/to/VEN_10DE...cfg --list")
            return 1

        # Filter: show NVIDIA (VEN_10DE) configs. Flag AMD (VEN_1002) as unsupported.
        nvidia_cfgs = [f for f in found if "VEN_10DE" in f.name.upper()]
        amd_cfgs = [f for f in found if "VEN_1002" in f.name.upper()]

        if not nvidia_cfgs and not amd_cfgs:
            print(f"Found Profiles folder but no per-GPU configs: {found[0].parent}")
            return 1

        # If no specific cfg was given, just list what we found
        if cfg_path is None:
            print(f"Found {len(nvidia_cfgs)} NVIDIA config(s) in {found[0].parent}:")
            for f in nvidia_cfgs:
                print(f"  {f.name}")
            if amd_cfgs:
                print(f"\nAlso found {len(amd_cfgs)} AMD config(s) (not yet supported):")
                for f in amd_cfgs:
                    print(f"  {f.name}")
            # If exactly 1 NVIDIA config and --list or --section given, use it
            if len(nvidia_cfgs) == 1:
                cfg_path = str(nvidia_cfgs[0])
                print(f"\nAuto-selected: {nvidia_cfgs[0].name}")
            elif len(nvidia_cfgs) > 1:
                print("\nMultiple configs found. Specify which one:")
                print("  kingai-gpu import-msi <path> --list")
                if not args.list and not args.save:
                    return 0
            else:
                print("\nNo NVIDIA configs found (AMD not yet supported).")
                return 1

        if cfg_path is None:
            return 0

    if cfg_path is None:
        print("No config file specified. Use --auto or provide a path.")
        return 1

    # List mode — just show what's in the file
    if args.list:
        try:
            profiles = list_sections(cfg_path)
        except (FileNotFoundError, OSError) as e:
            print(f"Error: {e}")
            return 1
        if not profiles:
            print("No profile sections found.")
            return 1
        print(f"Profiles in {Path(cfg_path).name}:")
        print(f"  {'Section':<20} {'Core':>6} {'Mem':>6} {'Power':>7} {'Thermal':>8} {'Fan':>5}")
        print(f"  {'─' * 20} {'─' * 6} {'─' * 6} {'─' * 7} {'─' * 8} {'─' * 5}")
        for p in profiles:
            print(f"  {p['section']:<20} {p['core']:>6} {p['mem']:>6} "
                  f"{p['power']:>7} {p['thermal']:>8} {p['fan']:>5}")
        return 0

    # Extract the profile
    section = args.section
    try:
        profile = extract_profile(cfg_path, section)
    except (FileNotFoundError, KeyError, OSError) as e:
        print(f"Error: {e}")
        return 1

    # Save-only mode — convert to KingAi JSON without applying
    if args.save:
        save_path = Path(args.save).resolve()
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        print(f"Converted [{section}] → {save_path}")
        _print_profile_summary(profile)
        return 0

    # Apply mode — import and apply to GPU via NVAPI
    try:
        from kingai_gpu.lib.nvapi import (
            NvApiError,
            enable_oc,
            set_core_offset,
            set_fan_auto,
            set_fan_speed,
            set_mem_offset,
            set_power_limit,
            set_thermal_limit,
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("Applying OC settings requires Windows with NVIDIA drivers.")
        print("Use --save to convert the profile without applying.")
        return 1

    gpu = args.gpu
    try:
        enable_oc(gpu)
    except NvApiError as e:
        print(f"Failed to initialize NVAPI: {e}")
        return 1

    print(f"Importing [{section}] from {Path(cfg_path).name} → GPU {gpu}...")

    changes = []

    if "core_offset_mhz" in profile:
        val = profile["core_offset_mhz"]
        try:
            set_core_offset(val, gpu)
            changes.append(f"  Core offset:   {val:+d} MHz")
        except NvApiError as e:
            print(f"Failed to set core offset: {e}")
            return 1

    if "mem_offset_mhz" in profile:
        val = profile["mem_offset_mhz"]
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
        val = profile["thermal_c"]
        try:
            set_thermal_limit(val, gpu)
            changes.append(f"  Thermal limit: {val}°C")
        except NvApiError as e:
            print(f"Failed to set thermal limit: {e}")
            return 1

    if profile.get("fan_auto"):
        try:
            set_fan_auto(gpu)
            changes.append(f"  Fan:           auto")
        except NvApiError as e:
            print(f"Failed to set fan auto: {e}")
            return 1
    elif "fan_pct" in profile:
        val = profile["fan_pct"]
        try:
            set_fan_speed(val, gpu)
            changes.append(f"  Fan speed:     {val}%")
        except NvApiError as e:
            print(f"Failed to set fan speed: {e}")
            return 1

    if "_voltage_boost_mv" in profile:
        print(f"  ⚠ Voltage offset {profile['_voltage_boost_mv']}mV skipped (not yet supported)")

    if changes:
        print(f"Applied to GPU {gpu}:")
        for c in changes:
            print(c)
    else:
        print("No applicable settings found in that profile section.")

    return 0


def _print_profile_summary(profile: dict):
    """Print a readable summary of extracted profile values."""
    if "core_offset_mhz" in profile:
        print(f"  Core offset:   {profile['core_offset_mhz']:+d} MHz")
    if "mem_offset_mhz" in profile:
        print(f"  Memory offset: {profile['mem_offset_mhz']:+d} MHz")
    if "power_pct" in profile:
        print(f"  Power limit:   {profile['power_pct']}%")
    if "thermal_c" in profile:
        print(f"  Thermal limit: {profile['thermal_c']}°C")
    if profile.get("fan_auto"):
        print(f"  Fan:           auto")
    elif "fan_pct" in profile:
        print(f"  Fan speed:     {profile['fan_pct']}%")
    if "_voltage_boost_mv" in profile:
        print(f"  Voltage boost: {profile['_voltage_boost_mv']}mV (not applied)")