"""Deterministic test doubles. No NVIDIA API is imported or initialized."""

from __future__ import annotations

from copy import deepcopy

from kingai_gpu.lib.backend import GpuIdentity, OcState, require_execute


def sample_state() -> OcState:
    return OcState(
        identity=GpuIdentity(
            name="NVIDIA GeForce RTX 3070 Ti",
            pci_bus_id="00000000:01:00.0",
            uuid="GPU-test-3070ti",
            pci_device_id="248210DE",
            pci_subsystem_id="40901458",
        ),
        core_offset_mhz=25,
        core_offset_range_mhz=(-500, 500),
        mem_offset_mhz=200,
        mem_offset_range_mhz=(-1000, 2000),
        power_pct=100,
        power_range_pct=(50, 110),
        thermal_c=83,
        thermal_range_c=(65, 90),
        fan_pct=None,
        fan_control_known=True,
        power_default_pct=100,
        thermal_default_c=83,
        valid_settings=frozenset(
            {
                "core_offset_mhz",
                "mem_offset_mhz",
                "power_pct",
                "thermal_c",
                "fan_pct",
            }
        ),
    )


class FakeBackend:
    def __init__(self, state: OcState | None = None):
        self.state = deepcopy(state or sample_state())
        self.initial = deepcopy(self.state)
        self.prepare_count = 0
        self.writes: list[tuple[str, int | float | None, int]] = []
        self.fail_once_on: str | None = None
        self.fail_prepare = False
        self.ignore_writes_on: set[str] = set()
        self.reset_count = 0

    def get_state(self, gpu: int = 0) -> OcState:
        return deepcopy(self.state)

    def prepare_write(self, gpu: int = 0, *, execute: bool = False) -> None:
        require_execute(execute)
        self.prepare_count += 1
        if self.fail_prepare:
            raise RuntimeError("injected preparation failure")

    def set_setting(
        self,
        name,
        value,
        gpu: int = 0,
        *,
        execute: bool = False,
    ) -> None:
        require_execute(execute)
        self.writes.append((name, value, gpu))
        if name not in self.ignore_writes_on:
            setattr(self.state, name, value)
        if self.fail_once_on == name:
            self.fail_once_on = None
            raise RuntimeError(f"injected {name} failure")

    def reset_all(self, gpu: int = 0, *, execute: bool = False) -> None:
        require_execute(execute)
        self.reset_count += 1
        self.state.core_offset_mhz = 0
        self.state.mem_offset_mhz = 0
        self.state.power_pct = self.state.power_default_pct
        self.state.thermal_c = self.state.thermal_default_c
        self.state.fan_pct = None


class FakeGpuMemory:
    is_gpu_memory = True

    def __init__(
        self,
        bandwidths=None,
        fail_measure=False,
        pci_bus_id="0000:01:00.0",
    ):
        self.bandwidths = iter(bandwidths or [100.0, 105.0, 103.0])
        self.fail_measure = fail_measure
        self._pci_bus_id = pci_bus_id
        self.test_durations: list[float] = []

    def pci_bus_id(self, gpu_index: int) -> str:
        return self._pci_bus_id

    def measure(self, size_mb: int, iterations: int, gpu_index: int) -> float:
        if self.fail_measure:
            raise RuntimeError("injected GPU-memory workload failure")
        return next(self.bandwidths)

    def test(self, size_mb: int, gpu_index: int, duration_sec: float = 0):
        from kingai_gpu.cli.memtest import MemtestResult

        self.test_durations.append(duration_sec)
        return MemtestResult(errors_detected=0, passed=True, backend="fake-gpu-memory")
