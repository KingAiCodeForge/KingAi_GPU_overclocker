"""
CLI entry point — kingai-gpu command dispatcher.

This is the main entry point for the entire application. It uses argparse
subcommands to route to the appropriate handler module:

  kingai-gpu monitor     →  cli/monitor.py     (real-time GPU monitoring)
  kingai-gpu oc          →  cli/overclock.py    (set clock offsets, power, fan)
  kingai-gpu memtest     →  cli/memtest.py      (VRAM stability + OC sweep)
  kingai-gpu info        →  cli/monitor.py      (static GPU info dump)
  kingai-gpu import-msi  →  cli/import_msi_profile_single.py (AB profile import)

Usage examples:
    kingai-gpu monitor [--once] [--json] [--interval N] [--gpu N]
    kingai-gpu oc [--core N] [--mem N] [--power N] [--fan N] [--reset]
    kingai-gpu oc --save profile.json          # snapshot current settings
    kingai-gpu oc --load profile.json          # plan saved profile
    kingai-gpu memtest [--sweep] [--duration N]
    kingai-gpu import-msi path/to/VEN_10DE...cfg --list
    kingai-gpu import-msi path/to/VEN_10DE...cfg --section Profile3
    kingai-gpu info

Design: Imports are deferred inside each branch so that:
  - 'monitor' and 'info' work on Linux (no NVAPI import)
  - 'oc' only imports nvapi.py on Windows
  - 'memtest' only imports cupy if actually running a GPU test
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from kingai_gpu import __version__

# ── Logging tee ─────────────────────────────────────────────────────────────
class _Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        written = self._stream.write(data)
        self._log.write(data)
        return written

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _init_log(command: str) -> str:
    """Set up file logging. Returns the log file path.

    Writes to a per-user state directory (or ``KINGAI_GPU_LOG_DIR``).
    stdout and stderr are mirrored to the file without modifying the console
    bytes. The launcher banner itself is emitted only on stderr, so JSON,
    JSONL, CSV, and other machine-readable stdout modes remain parseable.
    """
    configured = os.environ.get("KINGAI_GPU_LOG_DIR")
    if configured:
        logs_dir = Path(configured).expanduser()
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        logs_dir = base / "KingAiGPU" / "logs"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        logs_dir = base / "kingai-gpu"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = logs_dir / f"{command}_{stamp}.log"
    log_file = open(log_path, "x", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return str(log_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands.

    Each subcommand maps 1:1 to a handler function in its own module.
    Aliases ('mon', 'm', 'mem') are provided for faster typing in terminal.
    """
    p = argparse.ArgumentParser(
        prog="kingai-gpu",
        description="KingAi GPU Overclocker — monitoring, OC control, and stability testing",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", help="Command to run")

    # ── monitor ── Real-time GPU monitoring dashboard
    # Supports multiple output formats for different use cases:
    #   default = full box-drawing dashboard (human viewing)
    #   --json  = machine-readable for piping to other tools
    #   --csv   = append to log file for time-series analysis
    #   --compact = single-line for narrow terminals
    mon = sub.add_parser("monitor", aliases=["mon", "m"], help="Real-time GPU monitoring")
    mon.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    mon_format = mon.add_mutually_exclusive_group()
    mon_format.add_argument("--json", action="store_true", help="Output JSON (JSONL continuously)")
    mon_format.add_argument("--csv", action="store_true", help="Output as CSV (for logging)")
    mon.add_argument("--interval", "-i", type=float, default=1.0, help="Poll interval (seconds)")
    mon.add_argument("--gpu", "-g", type=int, default=0, help="GPU index")
    mon.add_argument("--compact", "-c", action="store_true", help="Single-line compact output")

    # ── oc ── Clock-offset and limit control (Windows-only via NVAPI)
    # All values are offsets from stock (e.g., --core +150 = +150 MHz above base).
    # If no Set flags are given, shows current OC status (read-only).
    oc = sub.add_parser("oc", help="Clock-offset and limit control")
    oc.add_argument(
        "--core",
        type=int,
        default=None,
        help="Core clock offset (MHz); a negative offset is not voltage control",
    )
    oc.add_argument("--mem", type=int, default=None, help="Memory clock offset (MHz)")
    oc.add_argument("--power", type=int, default=None, help="Power limit (%%)")
    oc.add_argument("--thermal", type=int, default=None, help="Thermal limit (°C)")
    oc.add_argument("--fan", type=int, default=None, help="Fan speed (%%)")
    oc.add_argument("--fan-auto", action="store_true", help="Reset fan to auto")
    oc.add_argument("--reset", action="store_true", help="Reset all to stock")
    oc.add_argument("--save", type=str, default=None, metavar="PATH",
                    help="Save current OC settings to JSON profile")
    oc.add_argument("--load", type=str, default=None, metavar="PATH",
                    help="Load and plan settings from a JSON profile")
    oc.add_argument("--gpu", "-g", type=int, default=0, help="GPU index")
    oc.add_argument("--status", "-s", action="store_true", help="Show current OC status")
    oc.add_argument(
        "--execute",
        action="store_true",
        help="Apply the planned hardware change (default is dry-run)",
    )

    # ── memtest ── VRAM stability testing
    # Default = single pass pattern test (write/read/verify VRAM patterns).
    # --sweep = automated memory OC sweep (set offset, measure BW, detect cliff).
    mt = sub.add_parser("memtest", aliases=["mem"], help="VRAM stability testing")
    mt.add_argument("--sweep", action="store_true", help="Automated memory OC sweep")
    mt.add_argument("--duration", "-d", type=int, default=10, help="Test duration per step (sec)")
    mt.add_argument("--start", type=int, default=0, help="Sweep start offset (MHz)")
    mt.add_argument("--stop", type=int, default=1500, help="Sweep stop offset (MHz)")
    mt.add_argument("--step", type=int, default=50, help="Sweep step size (MHz)")
    mt.add_argument("--gpu", "-g", type=int, default=0, help="GPU index")
    mt.add_argument("--size", type=int, default=256, help="Test buffer size (MB)")
    mt.add_argument(
        "--execute",
        action="store_true",
        help="Apply memory offsets during --sweep (default is dry-run)",
    )

    # ── info ── Static GPU information (no continuous monitoring)
    sub.add_parser("info", help="Detailed GPU information")

    # ── import-msi ── Import MSI Afterburner per-GPU config profiles
    # Reads a VEN_10DE&DEV_xxxx...cfg file and applies a [ProfileN] or [Startup]
    # section. Can also convert to KingAi JSON format (--save) or list all
    # sections (--list) without applying anything.
    imp = sub.add_parser("import-msi", help="Import MSI Afterburner profile")
    imp.add_argument("cfg", nargs="?", default=None,
                     help="Path to per-GPU .cfg file (or use --auto)")
    imp.add_argument("--auto", "-a", action="store_true",
                     help="Auto-detect Afterburner install and find per-GPU configs")
    imp.add_argument("--section", "-S", default="Startup",
                     help="Section to import (default: Startup)")
    imp.add_argument("--list", "-l", action="store_true",
                     help="List all profile sections in the .cfg")
    imp.add_argument("--save", type=str, default=None, metavar="PATH",
                     help="Convert to KingAi JSON profile (don't apply)")
    imp.add_argument("--gpu", "-g", type=int, default=0, help="GPU index")
    imp.add_argument(
        "--execute",
        action="store_true",
        help="Apply the imported settings (default is dry-run)",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch to the correct subcommand handler.

    Returns 0 on success, non-zero on error. Designed to be called
    from the console_scripts entry point defined in pyproject.toml.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # No subcommand given — show help
    if args.command is None:
        parser.print_help()
        return 0

    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        return _run_command(args, parser)
    except ModuleNotFoundError as exc:
        print(
            f"Missing runtime dependency {exc.name!r}. "
            "Install this package with its declared dependencies using pip.",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        return 1
    finally:
        # main() is also a public Python entry point. Each invocation owns its
        # log; restore callers' redirected streams even when dispatch fails.
        logs = []
        for stream, original in ((sys.stdout, original_stdout), (sys.stderr, original_stderr)):
            if stream is not original and isinstance(stream, _Tee):
                if stream._log not in logs:
                    logs.append(stream._log)
        sys.stdout, sys.stderr = original_stdout, original_stderr
        for log in logs:
            try:
                log.close()
            except OSError:
                pass


def _run_command(args, parser) -> int:
    """Dispatch one parsed command within its logging lifetime."""

    # Per-user diagnostics are separate from machine-readable stdout.
    cmd_name = args.command if args.command not in ("mon", "m") else "monitor"
    cmd_name = cmd_name if cmd_name != "mem" else "memtest"
    try:
        log_path = _init_log(cmd_name)
    except OSError as exc:
        log_path = "disabled"
        print(f"Warning: unable to create diagnostic log: {exc}", file=sys.stderr)
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{ts} kingai-gpu {cmd_name} — log: {log_path}", file=sys.stderr)

    # Deferred imports: each subcommand only imports what it needs.
    # This keeps 'monitor' working on Linux where nvapi.py would fail to import.
    if args.command in ("monitor", "mon", "m"):
        from kingai_gpu.cli.monitor import cmd_monitor
        return cmd_monitor(args)

    elif args.command == "oc":
        from kingai_gpu.cli.overclock import cmd_overclock
        return cmd_overclock(args)

    elif args.command in ("memtest", "mem"):
        from kingai_gpu.cli.memtest import cmd_memtest
        return cmd_memtest(args)

    elif args.command == "info":
        from kingai_gpu.cli.monitor import cmd_info
        return cmd_info(args)

    elif args.command == "import-msi":
        from kingai_gpu.cli.import_msi_profile_single import cmd_import_msi
        return cmd_import_msi(args)

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
