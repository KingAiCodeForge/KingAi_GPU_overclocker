from __future__ import annotations

import ctypes
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kingai_gpu.lib.backend import (
    NvApiBackend,
    WriteAuthorizationError,
    parse_pci_bdf,
    require_execute,
    rollback_settings,
)


class _FakeQueryInterface:
    """Enough of nvapi64.dll for importing the low-level module without a GPU."""

    restype = None
    argtypes = None

    def __call__(self, _query_id):
        return 0


@pytest.fixture
def fake_nvapi_module(monkeypatch):
    """Import nvapi.py against a fake DLL without calling NvAPI_Initialize."""

    module_name = "kingai_gpu.lib.nvapi"
    package = importlib.import_module("kingai_gpu.lib")
    missing = object()
    original_module = sys.modules.pop(module_name, None)
    original_attribute = getattr(package, "nvapi", missing)
    if original_attribute is not missing:
        delattr(package, "nvapi")

    fake_dll = SimpleNamespace(nvapi_QueryInterface=_FakeQueryInterface())
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(sys, "platform", "win32")
            scoped.setattr(ctypes, "CDLL", lambda _name: fake_dll)
            module = importlib.import_module(module_name)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
        if original_attribute is missing:
            if getattr(package, "nvapi", None) is module:
                delattr(package, "nvapi")
        else:
            setattr(package, "nvapi", original_attribute)


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("enable_oc", ()),
        ("set_core_offset", (50,)),
        ("set_mem_offset", (100,)),
        ("set_power_limit", (90,)),
        ("set_thermal_limit", (80,)),
        ("set_fan_speed", (60,)),
        ("set_fan_auto", ()),
        ("reset_all", ()),
    ],
)
def test_low_level_writers_refuse_before_init_or_write(
    fake_nvapi_module,
    monkeypatch,
    method_name,
    args,
):
    reached: list[str] = []

    def forbidden(*_args, **_kwargs):
        reached.append("hardware")
        raise AssertionError("authorization check ran too late")

    for internal_name in (
        "_init",
        "_set_clock_offset",
        "_set_power_status",
        "_set_thermal_limit",
        "_get_cached_fan_api",
        "_get_power_info",
        "_get_thermal_info",
    ):
        monkeypatch.setattr(fake_nvapi_module, internal_name, forbidden)

    with pytest.raises(WriteAuthorizationError, match="execute=True"):
        getattr(fake_nvapi_module, method_name)(*args)
    assert reached == []


def test_low_level_writer_accepts_literal_execute_true(fake_nvapi_module, monkeypatch):
    set_clock = Mock()
    monkeypatch.setattr(fake_nvapi_module, "_set_clock_offset", set_clock)

    fake_nvapi_module.set_core_offset(12.5, 3, execute=True)

    set_clock.assert_called_once_with(domain_id=0, offset_khz=12500, gpu=3)


@pytest.mark.parametrize("status", [None, -3])
def test_unavailable_bus_id_does_not_impersonate_bus_zero(
    fake_nvapi_module, monkeypatch, status,
):
    monkeypatch.setattr(fake_nvapi_module, "_handle", lambda _gpu: None)
    get_bus = None if status is None else lambda *_args: status
    monkeypatch.setitem(fake_nvapi_module._fn, "GPU_GetBusId", get_bus)
    assert fake_nvapi_module._get_bus_id() == -1


def test_real_bus_zero_remains_valid(fake_nvapi_module, monkeypatch):
    monkeypatch.setattr(fake_nvapi_module, "_handle", lambda _gpu: None)
    monkeypatch.setitem(fake_nvapi_module._fn, "GPU_GetBusId", lambda *_args: 0)
    assert fake_nvapi_module._get_bus_id() == 0


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("reason", ["missing", "get-error", "count-zero", "entry-size"])
def test_fan_fallback_preserves_requested_mode(
    fake_nvapi_module, monkeypatch, manual, reason,
):
    module = fake_nvapi_module
    monkeypatch.setattr(module, "_handle", lambda _gpu: None)

    def fake_get(_handle, pointer):
        if reason == "get-error":
            return -3
        if reason == "entry-size":
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32))[2] = 1
        return 0

    monkeypatch.setitem(module._fn, "GPU_ClientFanCoolersGetControl",
                        None if reason == "missing" else fake_get)
    monkeypatch.setitem(module._fn, "GPU_ClientFanCoolersSetControl", Mock())
    level, auto = Mock(), Mock()
    monkeypatch.setattr(module, "_set_cooler_level", level)
    monkeypatch.setattr(module, "_set_cooler_auto", auto)
    module._set_fan_new_api(0, manual=manual, gpu=3)
    if manual:
        level.assert_called_once_with(0, 0, 3)
        auto.assert_not_called()
    else:
        auto.assert_called_once_with(0, 3)
        level.assert_not_called()


@pytest.mark.parametrize(
    "invoke",
    [
        lambda backend: backend.prepare_write(),
        lambda backend: backend.set_setting("core_offset_mhz", 50),
        lambda backend: backend.reset_all(),
    ],
)
def test_backend_writers_refuse_before_loading_nvapi(monkeypatch, invoke):
    backend = NvApiBackend()
    loads: list[str] = []

    def forbidden_load():
        loads.append("nvapi")
        raise AssertionError("authorization check ran too late")

    monkeypatch.setattr(backend, "_load_nvapi", forbidden_load)

    with pytest.raises(WriteAuthorizationError, match="execute=True"):
        invoke(backend)
    assert loads == []


def test_backend_forwards_explicit_authorization_to_low_level(monkeypatch):
    nvapi = SimpleNamespace(
        enable_oc=Mock(),
        set_core_offset=Mock(),
        set_mem_offset=Mock(),
        set_power_limit=Mock(),
        set_thermal_limit=Mock(),
        set_fan_speed=Mock(),
        set_fan_auto=Mock(),
        reset_all=Mock(),
    )
    backend = NvApiBackend()
    monkeypatch.setattr(backend, "_load_nvapi", lambda: nvapi)

    backend.prepare_write(2, execute=True)
    backend.set_setting("core_offset_mhz", 75, 2, execute=True)
    backend.set_setting("fan_pct", None, 2, execute=True)
    backend.reset_all(2, execute=True)

    nvapi.enable_oc.assert_called_once_with(2, execute=True)
    nvapi.set_core_offset.assert_called_once_with(75, 2, execute=True)
    nvapi.set_fan_auto.assert_called_once_with(2, execute=True)
    nvapi.reset_all.assert_called_once_with(2, execute=True)


def test_write_authorization_rejects_truthy_non_boolean():
    with pytest.raises(WriteAuthorizationError):
        require_execute(1)


def test_rollback_helper_refuses_before_any_backend_call():
    backend = Mock()

    with pytest.raises(WriteAuthorizationError, match="execute=True"):
        rollback_settings(backend, Mock(), ["core_offset_mhz"])

    backend.set_setting.assert_not_called()


def _status_on_bus(bus_id: int):
    return SimpleNamespace(
        gpu_name="NVAPI GPU",
        bus_id=bus_id,
        core_offset_mhz=0,
        core_offset_range_mhz=(-500, 500),
        mem_offset_mhz=0,
        mem_offset_range_mhz=(-1000, 2000),
        power_pct=100,
        power_range_pct=(50, 110),
        thermal_c=83,
        thermal_range_c=(65, 90),
        fan_pct=None,
        fan_control_known=False,
        power_default_pct=100,
        thermal_default_c=83,
        core_read_valid=True,
        mem_read_valid=True,
        power_read_valid=True,
        thermal_read_valid=True,
    )


def _nvml_device(name: str, pci_bus_id: str):
    return SimpleNamespace(
        name=name,
        pci_bus_id=pci_bus_id,
        uuid=f"GPU-{name}",
        pci_device_id="248210DE",
        pci_subsystem_id="40901458",
    )


def test_pci_parser_normalizes_nvml_and_cuda_domain_widths():
    assert parse_pci_bdf("00000000:01:00.0") == parse_pci_bdf("0000:01:00.0")
    assert parse_pci_bdf("bus:1") is None


def test_backend_correlates_nvapi_to_nvml_by_bus_not_ordinal(monkeypatch):
    import kingai_gpu.lib as lib_package
    from kingai_gpu.lib import nvml as nvml_module

    monkeypatch.setattr(
        lib_package,
        "nvapi",
        SimpleNamespace(get_oc_status=lambda _gpu: _status_on_bus(2)),
        raising=False,
    )
    monkeypatch.setattr(
        nvml_module,
        "snapshot_all",
        lambda: [
            _nvml_device("ordinal-zero", "00000000:01:00.0"),
            _nvml_device("bus-match", "00000000:02:00.0"),
        ],
    )

    state = NvApiBackend().get_state(0)

    assert state.identity.name == "bus-match"
    assert state.identity.pci_bus_id == "00000000:02:00.0"
    assert state.identity.uuid == "GPU-bus-match"


def test_backend_rejects_ambiguous_nvml_bus_as_stable_identity(monkeypatch):
    import kingai_gpu.lib as lib_package
    from kingai_gpu.lib import nvml as nvml_module

    monkeypatch.setattr(
        lib_package,
        "nvapi",
        SimpleNamespace(get_oc_status=lambda _gpu: _status_on_bus(2)),
        raising=False,
    )
    monkeypatch.setattr(
        nvml_module,
        "snapshot_all",
        lambda: [
            _nvml_device("domain-zero", "00000000:02:00.0"),
            _nvml_device("domain-one", "00000001:02:00.0"),
        ],
    )

    state = NvApiBackend().get_state(0)

    assert state.identity.name == "NVAPI GPU"
    assert state.identity.pci_bus_id == "bus:2"
    assert state.identity.uuid == ""
