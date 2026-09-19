from __future__ import annotations

import csv
import io
import json
from types import SimpleNamespace

from kingai_gpu.cli import main, monitor
from kingai_gpu.lib.nvml import GpuSnapshot


def _snapshot(name: str = "GPU, Test") -> GpuSnapshot:
    return GpuSnapshot(
        index=0,
        name=name,
        driver_version="1",
        pci_bus_id="00000000:01:00.0",
        uuid="GPU-test",
        timestamp=1.0,
        throttle_reasons=["GPU_IDLE"],
    )


def test_single_json_is_one_compact_json_value():
    rendered = monitor.output_json(_snapshot())
    assert "\n" not in rendered
    assert json.loads(rendered)["name"] == "GPU, Test"


def test_csv_escapes_device_names():
    row = monitor.output_csv_row(_snapshot())
    parsed = next(csv.reader(io.StringIO(row)))
    assert parsed[1] == "GPU, Test"


def test_tee_mirrors_bytes_without_changing_stream_content():
    stream = io.StringIO()
    log = io.StringIO()
    tee = main._Tee(stream, log)
    tee.write('{"sample":1}\n')
    tee.flush()
    assert stream.getvalue() == '{"sample":1}\n'
    assert log.getvalue() == stream.getvalue()


def test_continuous_json_is_jsonl_and_interrupt_notice_is_stderr(monkeypatch, capsys):
    monkeypatch.setattr(monitor, "gpu_count", lambda: 1)
    monkeypatch.setattr(monitor, "snapshot", lambda _gpu: _snapshot("GPU"))

    def stop_after_first(_interval):
        raise KeyboardInterrupt

    monkeypatch.setattr(monitor.time, "sleep", stop_after_first)
    args = SimpleNamespace(
        once=False,
        json=True,
        csv=False,
        compact=False,
        interval=1,
        gpu=0,
    )
    assert monitor.cmd_monitor(args) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["name"] == "GPU"
    assert "---" not in captured.out
    assert "Monitoring stopped" in captured.err


def test_launcher_banner_goes_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr(main, "_init_log", lambda _command: "test.log")

    def fake_monitor(_args):
        print('{"ok":true}')
        return 0

    monkeypatch.setattr(monitor, "cmd_monitor", fake_monitor)
    assert main.main(["monitor", "--once", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"ok": True}
    assert "log: test.log" in captured.err


def test_log_creation_failure_does_not_contaminate_machine_stdout(monkeypatch, capsys):
    def fail_log(_command):
        raise OSError("read-only log location")

    monkeypatch.setattr(main, "_init_log", fail_log)

    def fake_monitor(_args):
        print('{"ok":true}')
        return 0

    monkeypatch.setattr(monitor, "cmd_monitor", fake_monitor)
    assert main.main(["monitor", "--once", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"ok": True}
    assert "unable to create diagnostic log" in captured.err


def _monitor_args(**overrides):
    values = {
        "once": True,
        "json": False,
        "csv": False,
        "compact": False,
        "interval": 1,
        "gpu": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_monitor_nvml_initialization_error_is_friendly(monkeypatch, capsys):
    def fail_init():
        raise RuntimeError("driver unavailable")

    monkeypatch.setattr(monitor, "gpu_count", fail_init)
    assert monitor.cmd_monitor(_monitor_args()) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        "Unable to initialize NVIDIA monitoring: driver unavailable"
    )


def test_monitor_invalid_index_is_refused_without_snapshot(monkeypatch, capsys):
    called = False

    def fail_if_called(_index):
        nonlocal called
        called = True
        raise AssertionError("snapshot must not be read for an invalid index")

    monkeypatch.setattr(monitor, "gpu_count", lambda: 1)
    monkeypatch.setattr(monitor, "snapshot", fail_if_called)
    assert monitor.cmd_monitor(_monitor_args(gpu=2)) == 2
    assert called is False
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        "GPU index 2 is out of range; available indices are 0 through 0."
    )


def test_monitor_snapshot_error_is_friendly(monkeypatch, capsys):
    monkeypatch.setattr(monitor, "gpu_count", lambda: 1)

    def fail_snapshot(_index):
        raise RuntimeError("GPU is lost")

    monkeypatch.setattr(monitor, "snapshot", fail_snapshot)
    assert monitor.cmd_monitor(_monitor_args()) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "Unable to read GPU 0: GPU is lost"


def test_info_nvml_initialization_error_is_friendly(monkeypatch, capsys):
    def fail_init():
        raise RuntimeError("NVML library not found")

    monkeypatch.setattr(monitor, "gpu_count", fail_init)
    assert monitor.cmd_info(SimpleNamespace()) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        "Unable to initialize NVIDIA monitoring: NVML library not found"
    )


def test_info_snapshot_error_is_friendly(monkeypatch, capsys):
    monkeypatch.setattr(monitor, "gpu_count", lambda: 1)

    def fail_snapshot(_index):
        raise RuntimeError("Invalid Argument")

    monkeypatch.setattr(monitor, "snapshot", fail_snapshot)
    assert monitor.cmd_info(SimpleNamespace()) == 1
    captured = capsys.readouterr()
    assert captured.out == "Found 1 NVIDIA GPU(s)\n\n"
    assert captured.err.strip() == "Unable to read GPU 0: Invalid Argument"
