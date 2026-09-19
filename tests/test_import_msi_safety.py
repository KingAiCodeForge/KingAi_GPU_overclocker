from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from kingai_gpu.cli.import_msi_profile_single import cmd_import_msi
from tests.test_fakes import FakeBackend, sample_state


def _args(path, **overrides):
    values = {
        "cfg": str(path),
        "auto": False,
        "section": "Profile3",
        "list": False,
        "save": None,
        "gpu": 0,
        "execute": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_cfg(
    tmp_path,
    bus="1",
    slot="0",
    function="0",
    device="2482",
    subsystem="40901458",
):
    path = tmp_path / (
        f"VEN_10DE&DEV_{device}&SUBSYS_{subsystem}&REV_A1"
        f"&BUS_{bus}&DEV_{slot}&FN_{function}.cfg"
    )
    path.write_text(
        "[Profile3]\n"
        "CoreClkBoost=75000\n"
        "MemClkBoost=200000\n"
        "PowerLimit=100\n"
        "ThermalLimit=83\n"
        "FanMode=0\n",
        encoding="ascii",
    )
    return path


def test_import_is_plan_only_by_default(tmp_path):
    backend = FakeBackend()
    path = _write_cfg(tmp_path)
    assert cmd_import_msi(_args(path), backend) == 0
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_import_bus_mismatch_refused_before_write(tmp_path):
    backend = FakeBackend()
    path = _write_cfg(tmp_path, bus="2")
    assert cmd_import_msi(_args(path, execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_import_slot_or_function_mismatch_refused_before_write(tmp_path):
    for slot, function in (("1", "0"), ("0", "1")):
        backend = FakeBackend()
        path = _write_cfg(tmp_path, slot=slot, function=function)
        assert cmd_import_msi(_args(path, execute=True), backend) == 2
        assert backend.prepare_count == 0
        assert backend.writes == []


def test_import_device_or_subsystem_mismatch_refused_before_write(tmp_path):
    for device, subsystem in (("2484", "40901458"), ("2482", "40911458")):
        backend = FakeBackend()
        path = _write_cfg(tmp_path, device=device, subsystem=subsystem)
        assert cmd_import_msi(_args(path, execute=True), backend) == 2
        assert backend.prepare_count == 0
        assert backend.writes == []


def test_import_refuses_bus_only_live_identity(tmp_path):
    state = sample_state()
    state.identity = replace(state.identity, pci_bus_id="bus:1")
    backend = FakeBackend(state)
    path = _write_cfg(tmp_path)
    assert cmd_import_msi(_args(path, execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_msi_numeric_bus_is_decimal_and_matches_hex_nvml_bdf(tmp_path):
    state = sample_state()
    state.identity = replace(state.identity, pci_bus_id="00000000:0A:00.0")
    backend = FakeBackend(state)
    path = _write_cfg(tmp_path, bus="10")
    assert cmd_import_msi(_args(path), backend) == 0
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_converted_import_is_v3_bound_profile_and_no_overwrite(tmp_path):
    backend = FakeBackend()
    path = _write_cfg(tmp_path)
    output = tmp_path / "converted.json"
    args = _args(path, save=str(output))
    assert cmd_import_msi(args, backend) == 0
    first = output.read_text(encoding="utf-8")
    assert '"schema_version": 3' in first
    assert '"GPU-test-3070ti"' in first
    assert cmd_import_msi(args, backend) == 2
    assert output.read_text(encoding="utf-8") == first
    assert backend.prepare_count == 0


def test_import_negative_gpu_index_refused_before_file_or_backend_access(tmp_path):
    backend = FakeBackend()
    missing = tmp_path / "does-not-exist.cfg"
    assert cmd_import_msi(_args(missing, gpu=-1, execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []
