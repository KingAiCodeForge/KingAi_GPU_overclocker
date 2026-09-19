"""Backend boundary for safe, testable GPU control.

The CLI plans and validates changes against this small interface.  The real
implementation delegates to NVAPI/NVML, while tests inject an in-memory fake.
Importing this module never initializes NVAPI and never changes hardware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class WriteAuthorizationError(PermissionError):
    """A hardware-changing API was called without explicit authorization."""


def require_execute(execute: bool) -> None:
    """Require the caller to opt in with the literal boolean ``True``."""

    if execute is not True:
        raise WriteAuthorizationError(
            "Hardware writes require the keyword-only argument execute=True"
        )


@dataclass(frozen=True)
class GpuIdentity:
    """Stable identity fields used to bind a profile to one physical GPU."""

    name: str
    pci_bus_id: str = ""
    uuid: str = ""
    pci_device_id: str = ""
    pci_subsystem_id: str = ""


def parse_pci_bdf(value: str) -> tuple[int, int, int, int] | None:
    """Parse a complete PCI domain:bus:device.function identifier.

    NVML commonly reports an eight-digit domain while CUDA commonly reports a
    four-digit domain.  Returning numeric components makes those equivalent
    forms directly comparable without weakening the device/function match.
    """

    match = re.fullmatch(
        r"(?P<domain>[0-9A-F]{4,8}):(?P<bus>[0-9A-F]{2}):"
        r"(?P<device>[0-9A-F]{2})\.(?P<function>[0-7])",
        value,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return tuple(int(match.group(name), 16) for name in ("domain", "bus", "device", "function"))


@dataclass
class OcState:
    """Point-in-time snapshot of every setting the CLI can change."""

    identity: GpuIdentity
    core_offset_mhz: float
    core_offset_range_mhz: tuple[float, float]
    mem_offset_mhz: float
    mem_offset_range_mhz: tuple[float, float]
    power_pct: float
    power_range_pct: tuple[float, float]
    thermal_c: int
    thermal_range_c: tuple[int, int]
    fan_pct: int | None = None
    fan_control_known: bool = False
    power_default_pct: float = 100.0
    thermal_default_c: int = 83
    valid_settings: frozenset[str] = field(default_factory=frozenset)


@runtime_checkable
class OcBackend(Protocol):
    """Read/write interface implemented by NVAPI and deterministic test fakes."""

    def get_state(self, gpu: int = 0) -> OcState: ...

    def prepare_write(self, gpu: int = 0, *, execute: bool = False) -> None: ...

    def set_setting(
        self,
        name: str,
        value: int | float | None,
        gpu: int = 0,
        *,
        execute: bool = False,
    ) -> None: ...

    def reset_all(self, gpu: int = 0, *, execute: bool = False) -> None: ...


class NvApiBackend:
    """Production backend. All imports and hardware access are lazy."""

    @staticmethod
    def _load_nvapi():
        """Import the Windows-only write layer after authorization checks."""

        from kingai_gpu.lib import nvapi

        return nvapi

    def get_state(self, gpu: int = 0) -> OcState:
        from kingai_gpu.lib import nvapi

        status = nvapi.get_oc_status(gpu)
        pci_bus_id = f"bus:{status.bus_id}"
        uuid = ""

        # NVML supplies stronger identity when available. A monitoring failure
        # must not erase the NVAPI state, but profiles made without a stable
        # identifier will later be rejected for execution.
        try:
            from kingai_gpu.lib.nvml import snapshot_all

            # NVAPI and NVML do not promise the same enumeration order. Bind
            # the NVAPI handle to the unique NVML device on its reported PCI
            # bus; an absent/ambiguous match deliberately leaves only the weak
            # ``bus:N`` fallback, which the validated write paths reject.
            candidates = []
            for candidate in snapshot_all():
                bdf = parse_pci_bdf(str(candidate.pci_bus_id or ""))
                if bdf is not None and bdf[1] == int(status.bus_id):
                    candidates.append(candidate)
            if len(candidates) != 1:
                raise RuntimeError(
                    "NVAPI PCI bus did not resolve to exactly one NVML device"
                )
            observed = candidates[0]
            pci_bus_id = str(observed.pci_bus_id or pci_bus_id)
            uuid = str(observed.uuid or "")
            pci_device_id = str(observed.pci_device_id or "")
            pci_subsystem_id = str(observed.pci_subsystem_id or "")
            name = str(observed.name or status.gpu_name)
        except Exception:
            name = str(status.gpu_name)
            pci_device_id = ""
            pci_subsystem_id = ""

        valid_settings = set()
        if status.core_read_valid:
            valid_settings.add("core_offset_mhz")
        if status.mem_read_valid:
            valid_settings.add("mem_offset_mhz")
        if status.power_read_valid:
            valid_settings.add("power_pct")
        if status.thermal_read_valid:
            valid_settings.add("thermal_c")
        if status.fan_control_known:
            valid_settings.add("fan_pct")

        return OcState(
            identity=GpuIdentity(
                name=name,
                pci_bus_id=pci_bus_id,
                uuid=uuid,
                pci_device_id=pci_device_id,
                pci_subsystem_id=pci_subsystem_id,
            ),
            core_offset_mhz=status.core_offset_mhz,
            core_offset_range_mhz=status.core_offset_range_mhz,
            mem_offset_mhz=status.mem_offset_mhz,
            mem_offset_range_mhz=status.mem_offset_range_mhz,
            power_pct=status.power_pct,
            power_range_pct=status.power_range_pct,
            thermal_c=status.thermal_c,
            thermal_range_c=status.thermal_range_c,
            fan_pct=status.fan_pct,
            fan_control_known=status.fan_control_known,
            power_default_pct=status.power_default_pct,
            thermal_default_c=status.thermal_default_c,
            valid_settings=frozenset(valid_settings),
        )

    def prepare_write(self, gpu: int = 0, *, execute: bool = False) -> None:
        require_execute(execute)
        nvapi = self._load_nvapi()

        nvapi.enable_oc(gpu, execute=True)

    def set_setting(
        self,
        name: str,
        value: int | float | None,
        gpu: int = 0,
        *,
        execute: bool = False,
    ) -> None:
        require_execute(execute)
        nvapi = self._load_nvapi()

        setters = {
            "core_offset_mhz": nvapi.set_core_offset,
            "mem_offset_mhz": nvapi.set_mem_offset,
            "power_pct": nvapi.set_power_limit,
            "thermal_c": nvapi.set_thermal_limit,
            "fan_pct": nvapi.set_fan_auto if value is None else nvapi.set_fan_speed,
        }
        try:
            setter = setters[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported GPU setting: {name}") from exc
        if name == "fan_pct" and value is None:
            setter(gpu, execute=True)
        else:
            setter(value, gpu, execute=True)

    def reset_all(self, gpu: int = 0, *, execute: bool = False) -> None:
        require_execute(execute)
        nvapi = self._load_nvapi()

        nvapi.reset_all(gpu, execute=True)


def state_value(state: OcState, name: str) -> int | float | None:
    """Return a setting from a prior state for rollback."""

    if name not in {
        "core_offset_mhz",
        "mem_offset_mhz",
        "power_pct",
        "thermal_c",
        "fan_pct",
    }:
        raise ValueError(f"Unsupported GPU setting: {name}")
    return getattr(state, name)


def rollback_settings(
    backend: OcBackend,
    prior: OcState,
    names: list[str],
    gpu: int = 0,
    *,
    execute: bool = False,
) -> list[str]:
    """Best-effort rollback in reverse order; return human-readable failures."""

    require_execute(execute)
    failures: list[str] = []
    for name in reversed(names):
        try:
            backend.set_setting(
                name,
                state_value(prior, name),
                gpu,
                execute=True,
            )
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    return failures
