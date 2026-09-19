from __future__ import annotations

from types import SimpleNamespace

from kingai_gpu.cli import memtest
from tests.test_fakes import FakeBackend, FakeGpuMemory


def _args(**overrides):
    values = {
        "sweep": True,
        "duration": 1,
        "start": 0,
        "stop": 100,
        "step": 50,
        "gpu": 0,
        "size": 1,
        "execute": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sensors(_gpu):
    return SimpleNamespace(temp_gpu=50, clock_mem=7000, power_draw=100.0)


def test_sweep_refuses_without_gpu_memory_backend(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(memtest, "get_gpu_memory_backend", lambda _gpu: None)
    assert memtest.cmd_memtest(_args(execute=True), backend=backend) == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_sweep_refuses_mismatched_cuda_device_before_first_write():
    backend = FakeBackend()
    result = memtest.cmd_memtest(
        _args(execute=True),
        backend=backend,
        memory_backend=FakeGpuMemory(pci_bus_id="0000:02:00.0"),
    )
    assert result == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_sweep_refuses_memory_backend_without_pci_identity_before_first_write():
    backend = FakeBackend()
    memory_backend = FakeGpuMemory()
    memory_backend.pci_bus_id = None
    result = memtest.cmd_memtest(
        _args(execute=True),
        backend=backend,
        memory_backend=memory_backend,
    )
    assert result == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_sweep_invalid_step_refused_before_first_write():
    backend = FakeBackend()
    result = memtest.cmd_memtest(
        _args(step=0, execute=True),
        backend=backend,
        memory_backend=FakeGpuMemory(),
    )
    assert result == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_sweep_unknown_prior_offset_refused_before_first_write():
    backend = FakeBackend()
    backend.state.valid_settings = backend.state.valid_settings - {"mem_offset_mhz"}
    result = memtest.cmd_memtest(
        _args(execute=True),
        backend=backend,
        memory_backend=FakeGpuMemory(),
    )
    assert result == 2
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_sweep_is_plan_only_without_execute():
    backend = FakeBackend()
    result = memtest.cmd_memtest(
        _args(),
        backend=backend,
        memory_backend=FakeGpuMemory(),
    )
    assert result == 0
    assert backend.prepare_count == 0
    assert backend.writes == []


def test_sweep_restores_exact_prior_offset_on_success():
    backend = FakeBackend()
    gpu_memory = FakeGpuMemory([100.0, 105.0])
    result = memtest.run_sweep(
        start_mhz=0,
        stop_mhz=50,
        step_mhz=50,
        test_duration=1,
        size_mb=1,
        gpu_index=0,
        execute=True,
        backend=backend,
        memory_backend=gpu_memory,
        settle_seconds=0,
        sensor_reader=_sensors,
    )
    assert result.crash_offset_mhz == -1
    assert backend.state.mem_offset_mhz == 200
    assert backend.writes[-1] == ("mem_offset_mhz", 200, 0)
    assert gpu_memory.test_durations == [1, 1]


def test_sweep_restores_exact_prior_offset_after_workload_exception():
    backend = FakeBackend()
    result = memtest.run_sweep(
        start_mhz=0,
        stop_mhz=50,
        step_mhz=50,
        test_duration=1,
        size_mb=1,
        gpu_index=0,
        execute=True,
        backend=backend,
        memory_backend=FakeGpuMemory(fail_measure=True),
        settle_seconds=0,
        sensor_reader=_sensors,
    )
    assert result.crash_offset_mhz == 0
    assert backend.state.mem_offset_mhz == 200
    assert backend.writes[-1] == ("mem_offset_mhz", 200, 0)


def test_sweep_prepare_failure_does_not_call_any_setter():
    backend = FakeBackend()
    backend.fail_prepare = True
    try:
        memtest.run_sweep(
            start_mhz=0,
            stop_mhz=50,
            step_mhz=50,
            test_duration=1,
            size_mb=1,
            gpu_index=0,
            execute=True,
            backend=backend,
            memory_backend=FakeGpuMemory(),
            settle_seconds=0,
            sensor_reader=_sensors,
        )
    except RuntimeError as exc:
        assert "preparation" in str(exc)
    else:
        raise AssertionError("expected injected preparation failure")
    assert backend.writes == []


def test_memtest_negative_gpu_index_refused_without_backend_access():
    backend = FakeBackend()
    result = memtest.cmd_memtest(
        _args(gpu=-1, execute=True),
        backend=backend,
        memory_backend=FakeGpuMemory(),
    )
    assert result == 2
    assert backend.prepare_count == 0
    assert backend.writes == []
