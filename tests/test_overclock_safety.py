from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kingai_gpu.cli import overclock
from kingai_gpu.cli.overclock import (
    PROFILE_SCHEMA_VERSION,
    cmd_overclock,
    profile_from_state,
    save_profile,
)
from tests.test_fakes import FakeBackend, sample_state


def _args(**overrides):
    values = {
        "gpu": 0,
        "core": None,
        "mem": None,
        "power": None,
        "thermal": None,
        "fan": None,
        "fan_auto": False,
        "reset": False,
        "save": None,
        "load": None,
        "status": False,
        "execute": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_status_never_prepares_or_writes(capsys):
    backend = FakeBackend()
    assert cmd_overclock(_args(status=True), backend) == 0
    assert backend.prepare_count == 0
    assert backend.writes == []
    assert "NVIDIA GeForce RTX 3070 Ti" in capsys.readouterr().out


@pytest.mark.parametrize(
    "conflict",
    [
        {"core": 75},
        {"mem": 200},
        {"power": 105},
        {"thermal": 85},
        {"fan": 70},
        {"fan_auto": True},
        {"reset": True},
        {"load": "profile.json"},
    ],
)
def test_status_conflicts_are_refused_before_backend_construction(
    conflict,
    monkeypatch,
    capsys,
):
    constructed = 0

    def fail_if_constructed():
        nonlocal constructed
        constructed += 1
        raise AssertionError("backend must not be constructed for a CLI conflict")

    monkeypatch.setattr(overclock, "NvApiBackend", fail_if_constructed)
    assert cmd_overclock(_args(status=True, execute=True, **conflict)) == 2
    assert constructed == 0
    assert "--status cannot be combined" in capsys.readouterr().err


def test_direct_change_is_plan_only_by_default():
    backend = FakeBackend()
    assert cmd_overclock(_args(core=75), backend) == 0
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_execute_reads_back_confirmed_change():
    backend = FakeBackend()
    assert cmd_overclock(_args(core=75, execute=True), backend) == 0
    assert backend.state.core_offset_mhz == 75
    assert backend.prepare_count == 1


def test_out_of_range_refused_before_first_write():
    backend = FakeBackend()
    assert cmd_overclock(_args(core=999, execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_unreadable_affected_setting_refused_before_first_write():
    state = sample_state()
    state.valid_settings = state.valid_settings - {"mem_offset_mhz"}
    backend = FakeBackend(state)
    assert cmd_overclock(_args(mem=400, execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_post_apply_mismatch_is_failure_and_rolls_back():
    backend = FakeBackend()
    backend.ignore_writes_on.add("core_offset_mhz")
    assert cmd_overclock(_args(core=75, execute=True), backend) == 1
    assert backend.state.core_offset_mhz == 25
    assert backend.prepare_count == 1


def test_fan_change_refused_when_prior_mode_cannot_be_read():
    state = sample_state()
    state.fan_control_known = False
    backend = FakeBackend(state)
    assert cmd_overclock(_args(fan=70, execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_reset_refused_when_prior_fan_mode_cannot_be_read():
    state = sample_state()
    state.fan_control_known = False
    backend = FakeBackend(state)
    assert cmd_overclock(_args(reset=True, execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.reset_count == 0


def test_reset_refused_when_any_authoritative_prior_read_is_missing():
    state = sample_state()
    state.valid_settings = state.valid_settings - {"power_pct"}
    backend = FakeBackend(state)
    assert cmd_overclock(_args(reset=True, execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.reset_count == 0


def test_reset_prepare_failure_does_not_call_reset_or_rollback_setters():
    backend = FakeBackend()
    backend.fail_prepare = True
    assert cmd_overclock(_args(reset=True, execute=True), backend) == 1
    assert backend.prepare_count == 1
    assert backend.reset_count == 0
    assert backend.writes == []


def test_reset_refuses_ambiguous_bus_only_identity_before_write():
    backend = FakeBackend()
    backend.state.identity = replace(
        backend.state.identity,
        pci_bus_id="bus:1",
        uuid="",
        pci_device_id="",
        pci_subsystem_id="",
    )
    assert cmd_overclock(_args(reset=True, execute=True), backend=backend) == 2
    assert backend.prepare_count == 0
    assert backend.reset_count == 0
    assert backend.writes == []


def test_partial_failure_restores_exact_prior_state():
    backend = FakeBackend()
    prior = deepcopy(backend.state)
    backend.fail_once_on = "mem_offset_mhz"
    assert cmd_overclock(_args(core=75, mem=400, execute=True), backend) == 1
    assert backend.state == prior
    assert backend.writes == [
        ("core_offset_mhz", 75, 0),
        ("mem_offset_mhz", 400, 0),
        ("mem_offset_mhz", 200, 0),
        ("core_offset_mhz", 25, 0),
    ]


def test_profile_identity_mismatch_refused_before_write(tmp_path):
    backend = FakeBackend()
    profile = profile_from_state(sample_state())
    profile["gpu"]["uuid"] = "GPU-other"
    path = tmp_path / "mismatch.json"
    save_profile(str(path), profile)
    assert cmd_overclock(_args(load=str(path), execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_profile_requires_complete_ranges_and_current_schema(tmp_path):
    backend = FakeBackend()
    profile = profile_from_state(sample_state())
    profile["schema_version"] = PROFILE_SCHEMA_VERSION - 1
    path = tmp_path / "old.json"
    save_profile(str(path), profile)
    assert cmd_overclock(_args(load=str(path), execute=True), backend) == 2
    assert backend.prepare_count == 0


def test_profile_missing_range_is_refused_before_write(tmp_path):
    backend = FakeBackend()
    profile = profile_from_state(sample_state())
    del profile["ranges"]["mem_mhz"]
    path = tmp_path / "incomplete.json"
    save_profile(str(path), profile)
    assert cmd_overclock(_args(load=str(path), execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_fractional_clock_value_is_refused_instead_of_truncated(tmp_path):
    backend = FakeBackend()
    profile = profile_from_state(sample_state())
    profile["settings"]["core_offset_mhz"] = 75.9
    path = tmp_path / "fractional.json"
    save_profile(str(path), profile)
    assert cmd_overclock(_args(load=str(path), execute=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_profile_save_refuses_overwrite(tmp_path):
    backend = FakeBackend()
    path = tmp_path / "profile.json"
    assert cmd_overclock(_args(save=str(path)), backend) == 0
    original = path.read_text(encoding="utf-8")
    assert cmd_overclock(_args(save=str(path)), backend) == 2
    assert path.read_text(encoding="utf-8") == original
    assert backend.prepare_count == 0


def test_negative_gpu_index_refused_without_backend_access():
    backend = FakeBackend()
    assert cmd_overclock(_args(gpu=-1, status=True), backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []
