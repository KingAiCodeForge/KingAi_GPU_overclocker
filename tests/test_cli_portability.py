"""Offline launcher and validation regressions; no real GPU API is loaded."""

import io
import sys

import pytest

from kingai_gpu.cli import main, memtest, monitor
from tests.test_monitor_output import _monitor_args


@pytest.mark.parametrize("interval", [float("nan"), float("inf"), -float("inf"), 0, -1])
def test_invalid_interval_fails_before_nvml(monkeypatch, capsys, interval):
    def forbidden():
        pytest.fail("NVML must not be reached for an invalid interval")

    monkeypatch.setattr(monitor, "gpu_count", forbidden)
    assert monitor.cmd_monitor(_monitor_args(interval=interval)) == 2
    assert "finite and greater than zero" in capsys.readouterr().err


def test_launcher_restores_redirected_streams_and_closes_log(monkeypatch, tmp_path):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setenv("KINGAI_GPU_LOG_DIR", str(tmp_path / "logs with spaces \u03a9"))
    logs = []

    def fake_monitor(_args):
        logs.append(sys.stdout._log)
        print('{"ok":true}')
        return 0

    monkeypatch.setattr(monitor, "cmd_monitor", fake_monitor)
    for _ in range(2):
        assert main.main(["monitor", "--once", "--json"]) == 0
        assert sys.stdout is out
        assert sys.stderr is err
    assert out.getvalue() == '{"ok":true}\n' * 2
    assert len(list(tmp_path.rglob("*.log"))) == 2
    assert all(log.closed for log in logs)


def test_missing_dependency_is_a_concise_error(monkeypatch, capsys):
    def missing(*_args):
        raise ModuleNotFoundError("No module named 'pynvml'", name="pynvml")

    monkeypatch.setattr(main, "_run_command", missing)
    assert main.main(["monitor", "--once"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pynvml" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_failed_optional_log_does_not_abort_command(monkeypatch, capsys, failure):
    class FullLog(io.StringIO):
        def write(self, data):
            if failure == "write":
                raise OSError("synthetic disk full")
            return super().write(data)

        def flush(self):
            if failure == "flush":
                raise OSError("synthetic disk full")
            return super().flush()

    log = FullLog()
    original_out, original_err = sys.stdout, sys.stderr

    def init(_command):
        sys.stdout = main._Tee(original_out, log, original_err)
        sys.stderr = main._Tee(original_err, log, original_err)
        return "synthetic.log"

    def fake_monitor(_args):
        print('{"ok":true}')
        sys.stdout.flush()
        return 0

    monkeypatch.setattr(main, "_init_log", init)
    monkeypatch.setattr(monitor, "cmd_monitor", fake_monitor)
    assert main.main(["monitor", "--once", "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.out == '{"ok":true}\n'
    assert "diagnostic log disabled" in captured.err
    assert log.closed


def test_memory_workload_failure_is_a_concise_error(monkeypatch, capsys):
    monkeypatch.setattr(memtest, "get_gpu_memory_backend", lambda _gpu: None)

    def fail(**_kwargs):
        raise MemoryError("synthetic buffer allocation failure")

    monkeypatch.setattr(memtest, "run_memtest", fail)
    args = main.build_parser().parse_args(["memtest", "--duration", "1", "--size", "1"])
    assert memtest.cmd_memtest(args) == 1
    captured = capsys.readouterr()
    assert "Memory test failed: synthetic buffer allocation failure" in captured.err
    assert "Traceback" not in captured.err
