"""Safe CLI planning and execution for GPU clock/limit controls.

Every hardware-changing operation is a dry-run unless ``--execute`` is
present. Profiles are schema-versioned, bound to one exact GPU identity, and
validated against the live ranges before the first write.
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from kingai_gpu.lib.backend import (
    GpuIdentity,
    NvApiBackend,
    OcBackend,
    OcState,
    rollback_settings,
    state_value,
)


PROFILE_SCHEMA = "kingai-gpu-profile"
PROFILE_SCHEMA_VERSION = 3
SETTING_ORDER = (
    "core_offset_mhz",
    "mem_offset_mhz",
    "power_pct",
    "thermal_c",
    "fan_pct",
)
RANGE_BY_SETTING = {
    "core_offset_mhz": "core_mhz",
    "mem_offset_mhz": "mem_mhz",
    "power_pct": "power_pct",
    "thermal_c": "thermal_c",
}


class PlanError(ValueError):
    """A requested operation is unsafe or internally inconsistent."""


def _identity_dict(identity: GpuIdentity) -> dict[str, str]:
    return {
        "name": identity.name,
        "pci_bus_id": identity.pci_bus_id,
        "uuid": identity.uuid,
        "pci_device_id": identity.pci_device_id,
        "pci_subsystem_id": identity.pci_subsystem_id,
    }


def _ranges_dict(state: OcState) -> dict[str, list[int | float]]:
    return {
        "core_mhz": list(state.core_offset_range_mhz),
        "mem_mhz": list(state.mem_offset_range_mhz),
        "power_pct": list(state.power_range_pct),
        "thermal_c": list(state.thermal_range_c),
        "fan_pct": [0, 100],
    }


def profile_from_state(state: OcState) -> dict[str, Any]:
    """Create the complete v3 profile representation for a state snapshot."""

    if not state.identity.name:
        raise PlanError("Cannot save profile: GPU name is unavailable")
    full_bdf = re.fullmatch(
        r"(?:[0-9A-F]{8}:)?[0-9A-F]{2}:[0-9A-F]{2}\.[0-7]",
        state.identity.pci_bus_id,
        re.IGNORECASE,
    )
    if not state.identity.uuid and not (
        full_bdf and state.identity.pci_device_id and state.identity.pci_subsystem_id
    ):
        raise PlanError(
            "Cannot save profile: no UUID or complete PCI BDF/device/subsystem identity"
        )
    required = {"core_offset_mhz", "mem_offset_mhz", "power_pct", "thermal_c"}
    missing = required - state.valid_settings
    if missing:
        raise PlanError(
            f"Cannot save profile: live reads unavailable for {sorted(missing)}"
        )
    for name, value in (
        ("core_offset_mhz", state.core_offset_mhz),
        ("mem_offset_mhz", state.mem_offset_mhz),
    ):
        if not float(value).is_integer():
            raise PlanError(
                f"Cannot save {name}={value:g}: v3 profiles require whole MHz values"
            )
    settings: dict[str, int | float | None] = {
        "core_offset_mhz": int(state.core_offset_mhz),
        "mem_offset_mhz": int(state.mem_offset_mhz),
        "power_pct": round(state.power_pct, 1),
        "thermal_c": int(state.thermal_c),
    }
    if state.fan_control_known:
        settings["fan_pct"] = state.fan_pct
    return {
        "schema": PROFILE_SCHEMA,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "gpu": _identity_dict(state.identity),
        "settings": settings,
        "ranges": _ranges_dict(state),
    }


def save_profile(path_str: str, profile: dict[str, Any]) -> str:
    """Write a profile without overwriting an existing file."""

    path = Path(path_str).resolve()
    if path.suffix == "":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(profile, handle, indent=2)
        handle.write("\n")
    return str(path)


def load_profile(path_str: str) -> dict[str, Any]:
    """Load a JSON object; semantic/profile checks happen separately."""

    path = Path(path_str).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Profile not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PlanError(f"Invalid profile: expected object, got {type(data).__name__}")
    return data


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise PlanError(f"{label} must be finite")
    return number


def _validate_range(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise PlanError(f"Profile range {label!r} must contain [minimum, maximum]")
    minimum = _finite_number(value[0], f"{label} minimum")
    maximum = _finite_number(value[1], f"{label} maximum")
    if minimum > maximum:
        raise PlanError(f"Profile range {label!r} has minimum greater than maximum")
    return minimum, maximum


def _same_identity(expected: dict[str, Any], actual: GpuIdentity) -> None:
    required = {
        "name",
        "pci_bus_id",
        "uuid",
        "pci_device_id",
        "pci_subsystem_id",
    }
    if not isinstance(expected, dict) or set(expected) != required:
        raise PlanError(
            "Profile gpu identity must contain exactly name, pci_bus_id, uuid, "
            "pci_device_id, and pci_subsystem_id"
        )

    expected_name = expected["name"]
    if not isinstance(expected_name, str) or not expected_name:
        raise PlanError("Profile GPU name is missing")
    if expected_name != actual.name:
        raise PlanError(
            f"GPU name mismatch: profile={expected_name!r}, selected={actual.name!r}"
        )

    expected_uuid = expected["uuid"]
    expected_pci = expected["pci_bus_id"]
    expected_device = expected["pci_device_id"]
    expected_subsystem = expected["pci_subsystem_id"]
    if not all(
        isinstance(item, str)
        for item in (expected_uuid, expected_pci, expected_device, expected_subsystem)
    ):
        raise PlanError("Profile PCI/UUID identity fields must be strings")
    for label, saved, selected in (
        ("PCI device ID", expected_device, actual.pci_device_id),
        ("PCI subsystem ID", expected_subsystem, actual.pci_subsystem_id),
    ):
        if saved and (not selected or saved.upper() != selected.upper()):
            raise PlanError(f"{label} mismatch: profile={saved!r}, selected={selected!r}")

    if expected_uuid:
        if not actual.uuid or expected_uuid != actual.uuid:
            raise PlanError(
                f"GPU UUID mismatch: profile={expected_uuid!r}, selected={actual.uuid!r}"
            )
        return
    if expected_pci:
        full_bdf = re.fullmatch(
            r"(?:[0-9A-F]{8}:)?[0-9A-F]{2}:[0-9A-F]{2}\.[0-7]",
            expected_pci,
            re.IGNORECASE,
        )
        if not full_bdf:
            raise PlanError(
                f"Profile PCI identity is not a complete BDF: {expected_pci!r}"
            )
        if not actual.pci_bus_id or expected_pci.upper() != actual.pci_bus_id.upper():
            raise PlanError(
                "GPU PCI identity mismatch: "
                f"profile={expected_pci!r}, selected={actual.pci_bus_id!r}"
            )
        if not expected_device or not expected_subsystem:
            raise PlanError(
                "Profile without a UUID requires PCI device and subsystem IDs"
            )
        return
    raise PlanError("Profile lacks a stable UUID or PCI bus identity")


def require_stable_identity(identity: GpuIdentity) -> None:
    """Refuse write plans that cannot be bound to one exact physical GPU."""

    _same_identity(_identity_dict(identity), identity)


def validate_profile(profile: dict[str, Any], live: OcState) -> dict[str, int | float | None]:
    """Validate schema, identity, complete ranges, and values against live ranges."""

    if profile.get("schema") != PROFILE_SCHEMA:
        raise PlanError(f"Unsupported profile schema: {profile.get('schema')!r}")
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise PlanError(
            f"Unsupported profile version: {profile.get('schema_version')!r}; "
            f"expected {PROFILE_SCHEMA_VERSION}"
        )
    _same_identity(profile.get("gpu"), live.identity)

    ranges = profile.get("ranges")
    required_ranges = {"core_mhz", "mem_mhz", "power_pct", "thermal_c", "fan_pct"}
    if not isinstance(ranges, dict) or set(ranges) != required_ranges:
        raise PlanError(f"Profile ranges must contain exactly {sorted(required_ranges)}")
    profile_ranges = {name: _validate_range(ranges[name], name) for name in required_ranges}

    settings = profile.get("settings")
    if not isinstance(settings, dict) or not settings:
        raise PlanError("Profile settings must be a non-empty object")
    unknown = set(settings) - set(SETTING_ORDER)
    if unknown:
        raise PlanError(f"Unsupported profile settings: {sorted(unknown)}")

    validated: dict[str, int | float | None] = {}
    live_ranges = _ranges_dict(live)
    for name in SETTING_ORDER:
        if name not in settings:
            continue
        if name not in live.valid_settings:
            raise PlanError(
                f"Selected GPU live read for {name} is unavailable; refusing a write "
                "with guessed range or rollback state"
            )
        value = settings[name]
        if name == "fan_pct" and not live.fan_control_known:
            raise PlanError(
                "Selected GPU fan control mode is unreadable; refusing a change "
                "that cannot be rolled back to an exact prior mode"
            )
        if name == "fan_pct" and value is None:
            validated[name] = None
            continue
        number = _finite_number(value, name)
        range_name = RANGE_BY_SETTING.get(name, "fan_pct")
        profile_min, profile_max = profile_ranges[range_name]
        live_min, live_max = live_ranges[range_name]
        if not profile_min <= number <= profile_max:
            raise PlanError(
                f"{name}={number:g} is outside its saved range "
                f"[{profile_min:g}, {profile_max:g}]"
            )
        if not float(live_min) <= number <= float(live_max):
            raise PlanError(
                f"{name}={number:g} is outside the selected GPU range "
                f"[{live_min:g}, {live_max:g}]"
            )
        if name != "power_pct" and not number.is_integer():
            raise PlanError(f"{name} must be a whole-number value, got {number:g}")
        validated[name] = int(number) if name != "power_pct" else number
    return validated


def validate_changes(
    changes: dict[str, int | float | None],
    live: OcState,
) -> dict[str, int | float | None]:
    """Validate direct CLI values using the same rules as profile values."""

    synthetic = {
        "schema": PROFILE_SCHEMA,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "gpu": _identity_dict(live.identity),
        "settings": changes,
        "ranges": _ranges_dict(live),
    }
    return validate_profile(synthetic, live)


def _setting_label(name: str, value: int | float | None) -> str:
    if name == "core_offset_mhz":
        return f"Core clock offset: {value:+g} MHz"
    if name == "mem_offset_mhz":
        return f"Memory clock offset: {value:+g} MHz"
    if name == "power_pct":
        return f"Power limit: {value:g}%"
    if name == "thermal_c":
        return f"Thermal limit: {value:g}°C"
    return "Fan control: auto" if value is None else f"Fan speed: {value:g}%"


def _print_plan(live: OcState, changes: dict[str, int | float | None], reset: bool) -> None:
    print(f"PLAN ONLY — GPU: {live.identity.name}")
    print(f"  PCI: {live.identity.pci_bus_id or 'unavailable'}")
    print(f"  UUID: {live.identity.uuid or 'unavailable'}")
    if reset:
        print("  Reset core, memory, power, thermal, and fan controls to defaults")
    else:
        for name in SETTING_ORDER:
            if name in changes:
                print(f"  {_setting_label(name, changes[name])}")
    print("No hardware changes made. Re-run with --execute to apply this plan.")


def _show_status(state: OcState, gpu: int) -> None:
    print(f"GPU {gpu}: {state.identity.name}")
    print(f"  PCI / UUID:     {state.identity.pci_bus_id} / {state.identity.uuid or 'unavailable'}")
    if "core_offset_mhz" in state.valid_settings:
        print(
            f"  Core offset:    {state.core_offset_mhz:+.0f} MHz "
            f"(range {state.core_offset_range_mhz[0]:+.0f} to "
            f"{state.core_offset_range_mhz[1]:+.0f})"
        )
    else:
        print("  Core offset:    unavailable (writes refused)")
    if "mem_offset_mhz" in state.valid_settings:
        print(
            f"  Memory offset:  {state.mem_offset_mhz:+.0f} MHz "
            f"(range {state.mem_offset_range_mhz[0]:+.0f} to "
            f"{state.mem_offset_range_mhz[1]:+.0f})"
        )
    else:
        print("  Memory offset:  unavailable (writes refused)")
    if "power_pct" in state.valid_settings:
        print(
            f"  Power limit:    {state.power_pct:.0f}% "
            f"(range {state.power_range_pct[0]:.0f}% to {state.power_range_pct[1]:.0f}%)"
        )
    else:
        print("  Power limit:    unavailable (writes refused)")
    if "thermal_c" in state.valid_settings:
        print(
            f"  Thermal limit:  {state.thermal_c}°C "
            f"(range {state.thermal_range_c[0]}°C to {state.thermal_range_c[1]}°C)"
        )
    else:
        print("  Thermal limit:  unavailable (writes refused)")
    if state.fan_control_known:
        fan = "auto" if state.fan_pct is None else f"manual {state.fan_pct}%"
    else:
        fan = "unknown (changes refused)"
    print(f"  Fan control:    {fan}")


def _setting_equal(name: str, actual, expected) -> bool:
    if actual is None or expected is None:
        return actual is expected
    tolerance = 0.01 if name in {"core_offset_mhz", "mem_offset_mhz", "power_pct"} else 0
    return abs(float(actual) - float(expected)) <= tolerance


def _verify_settings(
    backend: OcBackend,
    expected: dict[str, int | float | None],
    gpu: int,
) -> list[str]:
    """Read back affected controls and return every mismatch/read failure."""

    try:
        observed = backend.get_state(gpu)
    except Exception as exc:
        return [f"read-back failed: {exc}"]
    failures: list[str] = []
    for name, wanted in expected.items():
        if name not in observed.valid_settings:
            failures.append(f"{name}: read-back unavailable")
            continue
        actual = state_value(observed, name)
        if not _setting_equal(name, actual, wanted):
            failures.append(f"{name}: expected {wanted!r}, observed {actual!r}")
    return failures


def _apply_transaction(
    backend: OcBackend,
    prior: OcState,
    changes: dict[str, int | float | None],
    gpu: int,
) -> tuple[bool, str]:
    attempted: list[str] = []
    try:
        backend.prepare_write(gpu, execute=True)
        for name in SETTING_ORDER:
            if name not in changes:
                continue
            # Include the current operation in rollback: a failed low-level call
            # may still have made a partial driver change.
            attempted.append(name)
            backend.set_setting(name, changes[name], gpu, execute=True)
        verification = _verify_settings(
            backend,
            {name: changes[name] for name in attempted},
            gpu,
        )
        if verification:
            raise RuntimeError(f"post-apply verification failed: {verification}")
    except Exception as exc:
        rollback_failures = rollback_settings(
            backend,
            prior,
            attempted,
            gpu,
            execute=True,
        )
        rollback_verification = (
            _verify_settings(
                backend,
                {name: state_value(prior, name) for name in attempted},
                gpu,
            )
            if attempted
            else []
        )
        message = f"Apply failed: {exc}. Rollback attempted for {attempted or ['none']}."
        if rollback_failures:
            message += f" Rollback failures: {rollback_failures}"
        if rollback_verification:
            message += f" Rollback verification failures: {rollback_verification}"
        return False, message
    return True, ""


def _reset_transaction(backend: OcBackend, prior: OcState, gpu: int) -> tuple[bool, str]:
    names = list(SETTING_ORDER)
    try:
        backend.prepare_write(gpu, execute=True)
    except Exception as exc:
        return False, f"Reset preparation failed before any reset call: {exc}"
    try:
        backend.reset_all(gpu, execute=True)
        expected = {
            "core_offset_mhz": 0,
            "mem_offset_mhz": 0,
            "power_pct": prior.power_default_pct,
            "thermal_c": prior.thermal_default_c,
            "fan_pct": None,
        }
        verification = _verify_settings(backend, expected, gpu)
        if verification:
            raise RuntimeError(f"post-reset verification failed: {verification}")
    except Exception as exc:
        rollback_failures = rollback_settings(
            backend,
            prior,
            names,
            gpu,
            execute=True,
        )
        rollback_verification = _verify_settings(
            backend,
            {name: state_value(prior, name) for name in names},
            gpu,
        )
        message = f"Reset failed: {exc}. Prior-state rollback attempted."
        if rollback_failures:
            message += f" Rollback failures: {rollback_failures}"
        if rollback_verification:
            message += f" Rollback verification failures: {rollback_verification}"
        return False, message
    return True, ""


def cmd_overclock(args, backend: OcBackend | None = None) -> int:
    """Handle ``oc`` with read-only planning by default."""

    gpu = args.gpu
    if gpu < 0:
        print("GPU index must not be negative.", file=sys.stderr)
        return 2
    direct = {
        name: value
        for name, value in (
            ("core_offset_mhz", args.core),
            ("mem_offset_mhz", args.mem),
            ("power_pct", args.power),
            ("thermal_c", args.thermal),
            ("fan_pct", args.fan),
        )
        if value is not None
    }
    if args.fan_auto:
        if "fan_pct" in direct:
            print("Error: --fan and --fan-auto are mutually exclusive", file=sys.stderr)
            return 2
        direct["fan_pct"] = None

    # Keep the explicit status command strictly read-only. This validation must
    # happen before constructing the real backend: backend construction may
    # load NVAPI, while a contradictory command is a CLI error that needs no
    # hardware access at all.
    if args.status and (direct or args.reset or args.load):
        print(
            "Error: --status cannot be combined with settings, --reset, or --load",
            file=sys.stderr,
        )
        return 2

    action_count = sum(bool(item) for item in (direct, args.reset, args.load))
    if action_count > 1:
        print("Error: use only one of direct settings, --reset, or --load", file=sys.stderr)
        return 2
    if args.save and action_count:
        print("Error: --save cannot be combined with a hardware-change plan", file=sys.stderr)
        return 2

    selected = backend or NvApiBackend()

    try:
        live = selected.get_state(gpu)
    except Exception as exc:
        print(f"Unable to read GPU {gpu} state: {exc}", file=sys.stderr)
        return 1

    if args.save:
        try:
            saved = save_profile(args.save, profile_from_state(live))
        except FileExistsError:
            print(f"Refusing to overwrite existing profile: {Path(args.save).resolve()}", file=sys.stderr)
            return 2
        except (OSError, ValueError) as exc:
            print(f"Failed to save profile: {exc}", file=sys.stderr)
            return 1
        print(f"Saved profile: {saved}")
        return 0

    if args.load:
        try:
            changes = validate_profile(load_profile(args.load), live)
        except (OSError, json.JSONDecodeError, PlanError) as exc:
            print(f"Profile refused: {exc}", file=sys.stderr)
            return 2
    elif direct:
        try:
            changes = validate_changes(direct, live)
        except PlanError as exc:
            print(f"Plan refused: {exc}", file=sys.stderr)
            return 2
    else:
        changes = {}

    reset_required = set(SETTING_ORDER)
    reset_missing = reset_required - live.valid_settings
    if not live.fan_control_known:
        reset_missing.add("fan_pct")
    if args.reset and reset_missing:
        print(
            f"Reset refused: live reads unavailable for {sorted(reset_missing)}, "
            "so an exact prior-state rollback is impossible.",
            file=sys.stderr,
        )
        return 2
    if args.reset:
        try:
            require_stable_identity(live.identity)
        except PlanError as exc:
            print(f"Reset refused: {exc}", file=sys.stderr)
            return 2

    if not changes and not args.reset:
        _show_status(live, gpu)
        return 0

    if not args.execute:
        _print_plan(live, changes, args.reset)
        return 0

    if args.reset:
        ok, error = _reset_transaction(selected, live, gpu)
    else:
        ok, error = _apply_transaction(selected, live, changes, gpu)
    if not ok:
        print(error, file=sys.stderr)
        return 1

    print(f"Applied to GPU {gpu}: {live.identity.name}")
    if args.reset:
        print("  Reset controls to defaults")
    else:
        for name in SETTING_ORDER:
            if name in changes:
                print(f"  {_setting_label(name, changes[name])}")
    return 0
