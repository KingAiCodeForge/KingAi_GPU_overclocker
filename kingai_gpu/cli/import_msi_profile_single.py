"""Read and safely plan an MSI Afterburner profile import.

MSI configuration files are never edited. Applying extracted settings follows
the same identity, range, ``--execute``, snapshot, and rollback gates as the
native ``oc`` command.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from kingai_gpu.cli.overclock import (
    PlanError,
    _apply_transaction,
    _print_plan,
    profile_from_state,
    save_profile,
    validate_changes,
)
from kingai_gpu.lib.backend import NvApiBackend, OcBackend


_STANDARD_PATHS = [
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
    / "MSI Afterburner"
    / "Profiles",
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    / "MSI Afterburner"
    / "Profiles",
]
_CFG_IDENTITY = re.compile(
    r"VEN_(?P<vendor>[0-9A-F]{4})&DEV_(?P<device>[0-9A-F]{4})"
    r"&SUBSYS_(?P<subsys>[0-9A-F]{8}).*?&BUS_(?P<bus>[0-9A-F]+)"
    r"&DEV_(?P<slot>[0-9A-F]+)&FN_(?P<function>[0-9A-F]+)",
    re.IGNORECASE,
)
_FIELD_MAP = {
    "CoreClkBoost": ("core_offset_mhz", lambda value: int(value) // 1000),
    "MemClkBoost": ("mem_offset_mhz", lambda value: int(value) // 1000),
    "PowerLimit": ("power_pct", int),
    "ThermalLimit": ("thermal_c", int),
    "FanSpeed": ("fan_pct", int),
}


def find_afterburner_profiles() -> list[Path]:
    """Return per-GPU Afterburner configs from standard/registered locations."""

    profiles_dir = next((path for path in _STANDARD_PATHS if path.is_dir()), None)
    if profiles_dir is None:
        profiles_dir = _find_via_registry()
    return sorted(profiles_dir.glob("VEN_*.cfg")) if profiles_dir else []


def _find_via_registry() -> Path | None:
    try:
        import winreg
    except ImportError:
        return None
    for key_path in (
        r"SOFTWARE\WOW6432Node\MSI\Afterburner",
        r"SOFTWARE\MSI\Afterburner",
    ):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                profiles = Path(install_path) / "Profiles"
                if profiles.is_dir():
                    return profiles
        except (FileNotFoundError, OSError):
            continue
    return None


def _parse_sections(text: str) -> dict[str, str]:
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
    for line in block.splitlines():
        if line.strip().startswith(key + "="):
            value = line.split("=", 1)[1].strip()
            return value or None
    return None


def extract_profile(cfg_path: str, section: str = "Startup") -> dict:
    """Extract supported settings and source metadata without changing hardware."""

    path = Path(cfg_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    sections = _parse_sections(path.read_text(encoding="ascii", errors="replace"))
    if section not in sections:
        available = [name for name in sections if name not in {"Defaults", "Settings"}]
        raise KeyError(
            f"Section [{section}] not found in {path.name}. Available: {', '.join(available)}"
        )

    block = sections[section]
    settings: dict[str, int | float | None] = {}
    for afterburner_name, (our_name, convert) in _FIELD_MAP.items():
        raw = _get_value(block, afterburner_name)
        if raw is None:
            continue
        try:
            settings[our_name] = convert(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {afterburner_name} value in [{section}]: {raw!r}")

    if _get_value(block, "FanMode") == "0":
        settings["fan_pct"] = None

    source_identity = _CFG_IDENTITY.search(path.stem)
    source = {
        "kind": "msi_afterburner",
        "file": path.name,
        "section": section,
        "device_identity": source_identity.groupdict() if source_identity else {},
    }
    voltage = _get_value(block, "CoreVoltageBoost")
    if voltage:
        source["ignored_core_voltage_boost"] = int(voltage)
    return {"settings": settings, "source": source}


def list_sections(cfg_path: str) -> list[dict[str, str]]:
    path = Path(cfg_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    sections = _parse_sections(path.read_text(encoding="ascii", errors="replace"))
    results: list[dict[str, str]] = []
    for name, block in sections.items():
        if name in {"Defaults", "Settings"}:
            continue
        core = _get_value(block, "CoreClkBoost")
        memory = _get_value(block, "MemClkBoost")
        power = _get_value(block, "PowerLimit")
        thermal = _get_value(block, "ThermalLimit")
        fan_mode = _get_value(block, "FanMode")
        fan_speed = _get_value(block, "FanSpeed")
        results.append(
            {
                "section": name,
                "core": f"{int(core) // 1000:+d}" if core else "N/A",
                "mem": f"{int(memory) // 1000:+d}" if memory else "N/A",
                "power": f"{power}%" if power else "N/A",
                "thermal": f"{thermal}°C" if thermal else "N/A",
                "fan": f"{fan_speed}%" if fan_mode == "1" else "auto",
            }
        )
    return results


def _parse_msi_number(value: str) -> int:
    """Parse Afterburner identity numbers.

    Existing Afterburner filenames use unprefixed decimal numbers (for example
    ``BUS_10`` means decimal bus 10, matching NVML bus ``0A``). If an identifier
    contains A-F, accept it as hexadecimal for compatibility with hand-copied
    hardware IDs.
    """

    return int(value, 16 if re.search(r"[A-F]", value, re.IGNORECASE) else 10)


def _pci_bdf(pci_bus_id: str) -> tuple[int, int, int] | None:
    """Parse NVML PCI ``domain:bus:device.function`` into numeric BDF."""

    match = re.fullmatch(
        r"(?:[0-9A-F]{8}:)?(?P<bus>[0-9A-F]{2}):"
        r"(?P<device>[0-9A-F]{2})\.(?P<function>[0-7])",
        pci_bus_id,
        re.IGNORECASE,
    )
    if not match:
        return None
    return (
        int(match.group("bus"), 16),
        int(match.group("device"), 16),
        int(match.group("function"), 16),
    )


def _validate_source_device(source: dict, live_identity) -> None:
    identity = source.get("device_identity")
    if not isinstance(identity, dict) or not identity:
        raise PlanError("Afterburner filename lacks a complete VEN/DEV/SUBSYS/BUS/DEV/FN identity")
    if identity.get("vendor", "").upper() != "10DE":
        raise PlanError("Only NVIDIA (VEN_10DE) Afterburner profiles are supported")
    selected_bdf = _pci_bdf(live_identity.pci_bus_id)
    if selected_bdf is None:
        raise PlanError(
            "Selected GPU lacks a complete NVML PCI bus/device/function identity; "
            "bus-only matching is not sufficient for import"
        )
    source_bdf = (
        _parse_msi_number(identity["bus"]),
        _parse_msi_number(identity["slot"]),
        _parse_msi_number(identity["function"]),
    )
    if source_bdf != selected_bdf:
        raise PlanError(
            "Afterburner profile PCI BDF "
            f"{source_bdf[0]:02X}:{source_bdf[1]:02X}.{source_bdf[2]} does not match "
            "selected GPU BDF "
            f"{selected_bdf[0]:02X}:{selected_bdf[1]:02X}.{selected_bdf[2]}"
        )

    source_device_id = (identity["device"] + identity["vendor"]).upper()
    source_subsystem_id = identity["subsys"].upper()
    if not live_identity.pci_device_id or not live_identity.pci_subsystem_id:
        raise PlanError(
            "Selected GPU lacks complete NVML PCI device/subsystem IDs; "
            "exact Afterburner source matching is unavailable"
        )
    if source_device_id != live_identity.pci_device_id.upper():
        raise PlanError(
            f"Afterburner PCI device ID {source_device_id} does not match "
            f"selected GPU {live_identity.pci_device_id}"
        )
    if source_subsystem_id != live_identity.pci_subsystem_id.upper():
        raise PlanError(
            f"Afterburner subsystem ID {source_subsystem_id} does not match "
            f"selected GPU {live_identity.pci_subsystem_id}"
        )


def _print_profile_summary(settings: dict[str, int | float | None]) -> None:
    for name, label, suffix in (
        ("core_offset_mhz", "Core offset", " MHz"),
        ("mem_offset_mhz", "Memory offset", " MHz"),
        ("power_pct", "Power limit", "%"),
        ("thermal_c", "Thermal limit", "°C"),
    ):
        if name in settings:
            print(f"  {label}: {settings[name]}{suffix}")
    if "fan_pct" in settings:
        print("  Fan: auto" if settings["fan_pct"] is None else f"  Fan: {settings['fan_pct']}%")


def _resolve_cfg(args) -> tuple[str | None, int]:
    cfg_path = args.cfg
    if not args.auto and cfg_path is not None:
        return cfg_path, 0

    found = find_afterburner_profiles()
    nvidia = [path for path in found if "VEN_10DE" in path.name.upper()]
    if cfg_path is not None:
        return cfg_path, 0
    if len(nvidia) == 1:
        return str(nvidia[0]), 0
    if not nvidia:
        print("No NVIDIA Afterburner per-GPU config found.", file=sys.stderr)
        return None, 1
    print("Multiple NVIDIA configs found; provide one explicit path:", file=sys.stderr)
    for path in nvidia:
        print(f"  {path}", file=sys.stderr)
    return None, 2


def cmd_import_msi(args, backend: OcBackend | None = None) -> int:
    """List, convert, plan, or explicitly execute an MSI profile import."""

    if args.gpu < 0:
        print("GPU index must not be negative.", file=sys.stderr)
        return 2
    cfg_path, code = _resolve_cfg(args)
    if cfg_path is None:
        return code

    if args.list:
        try:
            profiles = list_sections(cfg_path)
        except (OSError, KeyError, ValueError) as exc:
            print(f"Unable to list profiles: {exc}", file=sys.stderr)
            return 1
        print(f"Profiles in {Path(cfg_path).name}:")
        for item in profiles:
            print(
                f"  {item['section']:<20} core={item['core']:>6} "
                f"mem={item['mem']:>6} power={item['power']:>7} "
                f"thermal={item['thermal']:>7} fan={item['fan']}"
            )
        return 0

    try:
        extracted = extract_profile(cfg_path, args.section)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Import refused: {exc}", file=sys.stderr)
        return 2

    selected = backend or NvApiBackend()
    try:
        live = selected.get_state(args.gpu)
        _validate_source_device(extracted["source"], live.identity)
        changes = validate_changes(extracted["settings"], live)
    except Exception as exc:
        print(f"Import refused before hardware access: {exc}", file=sys.stderr)
        return 2

    if args.save:
        profile = profile_from_state(live)
        profile["settings"] = changes
        profile["source"] = extracted["source"]
        try:
            output = save_profile(args.save, profile)
        except FileExistsError:
            print(f"Refusing to overwrite existing profile: {Path(args.save).resolve()}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"Unable to save converted profile: {exc}", file=sys.stderr)
            return 1
        print(f"Saved converted profile: {output}")
        _print_profile_summary(changes)
        return 0

    if not args.execute:
        _print_plan(live, changes, reset=False)
        print(f"  Source: {Path(cfg_path).name} [{args.section}]")
        return 0

    ok, error = _apply_transaction(selected, live, changes, args.gpu)
    if not ok:
        print(error, file=sys.stderr)
        return 1
    print(f"Applied [{args.section}] to GPU {args.gpu}: {live.identity.name}")
    _print_profile_summary(changes)
    ignored_voltage = extracted["source"].get("ignored_core_voltage_boost")
    if ignored_voltage:
        print(
            f"  Voltage boost {ignored_voltage} mV was not applied; this tool has no "
            "V/F-curve undervolting control.",
            file=sys.stderr,
        )
    return 0
