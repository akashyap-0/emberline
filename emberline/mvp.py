"""One-command Emberline MVP run.

    make mvp            (or)   python -m emberline.mvp [--summary-only] [--skip-verify]

Runs, in order:

1. ``make verify`` (lint + full test suite + fast end-to-end smoke), or
   ``bash verify.sh`` when ``make`` is not installed (identical steps);
2. the canonical instrumented demo
   ``python -m emberline.demo --scenario ridgeline --fast --wind-shift 40 --kill-node N3``;
3. ``python -m emberline.demo.pitch_assets`` (six 1920x1080 PNGs);
4. ``docs/MVP_RUN.md``: a dated summary built ONLY from artifacts the run
   produced (``metrics/demo_last_run.json``, ``metrics/*.json``, the demo
   scoreboard text, the files on disk) plus the pass/fail of the test suite.

The summary is refused (``MvpError``) if the demo did not produce a fresh
``metrics/demo_last_run.json``: nothing here re-invents a number. Re-running
is idempotent - every step overwrites its own outputs.

Everything this command measures is synthetic simulation; the summary says
so at the top and the bottom.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import repo_root

DEMO_ARGS = ["--scenario", "ridgeline", "--fast", "--wind-shift", "40", "--kill-node", "N3"]
PITCH_ASSETS = ["system_architecture", "cone_evacuation_before_after", "mesh_topology",
                "detection_confusion", "calibration_before_after", "warning_timeline"]
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class MvpError(RuntimeError):
    """Raised when the summary cannot be built from real artifacts."""


@dataclass
class StepResult:
    name: str
    command: str
    returncode: int | None
    seconds: float
    output: str = ""
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.skipped or self.returncode == 0


@dataclass
class VerifySummary:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    all_green: bool = False
    details: list[str] = field(default_factory=list)


# --------------------------------------------------------------- helpers ----
def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _find_bash() -> str | None:
    """Locate a POSIX bash that can run verify.sh.

    On Windows ``shutil.which('bash')`` usually returns the WSL launcher in
    System32, which cannot run a repo script from a Windows path; prefer the
    bash shipped with Git, located via ``git --exec-path``. Override with the
    ``EMBERLINE_BASH`` environment variable.
    """
    env = os.environ.get("EMBERLINE_BASH")
    if env and pathlib.Path(env).exists():
        return env
    if os.name != "nt":
        return shutil.which("bash")
    candidates: list[pathlib.Path] = []
    try:
        exec_path = subprocess.run(["git", "--exec-path"], capture_output=True, text=True,
                                   check=True).stdout.strip()
        git_root = pathlib.Path(exec_path).resolve().parents[2]  # .../Git
        candidates += [git_root / "bin" / "bash.exe", git_root / "usr" / "bin" / "bash.exe"]
    except Exception:
        pass
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                 str(pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs")):
        candidates.append(pathlib.Path(base) / "Git" / "bin" / "bash.exe")
    for c in candidates:
        if c.exists():
            return str(c)
    which = shutil.which("bash")
    if which and "system32" not in which.lower() and "windowsapps" not in which.lower():
        return which
    return None


def verify_command() -> list[str]:
    """``make verify`` if make exists, else ``bash verify.sh`` (same steps)."""
    if shutil.which("make"):
        return ["make", "verify"]
    bash = _find_bash()
    if bash is None:
        raise MvpError("neither `make` nor a usable `bash` was found; install GNU make "
                       "or Git for Windows, or set EMBERLINE_BASH to a bash.exe")
    return [bash, "verify.sh"]


def _run_step(name: str, cmd: list[str], cwd: pathlib.Path, echo: bool = True) -> StepResult:
    """Run a subprocess, streaming its output, and capture it."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    shown = " ".join([pathlib.Path(cmd[0]).name, *cmd[1:]])  # basename keeps the table readable
    if echo:
        print(f"\n== mvp: {name} :: {shown}", flush=True)
    t0 = time.perf_counter()
    lines: list[str] = []
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    except OSError as e:
        return StepResult(name, shown, None, 0.0, output=f"could not start: {e}")
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        lines.append(line)
        if echo:
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode(), flush=True)
    rc = proc.wait()
    return StepResult(name, shown, rc, time.perf_counter() - t0, output="\n".join(lines))


def parse_verify(output: str, returncode: int | None) -> VerifySummary:
    """Pass/fail counts from verify.sh output.

    pytest runs with ``-q`` twice over (pyproject addopts + verify.sh), so the
    only per-test signal is the progress glyph line: ``.`` pass, ``F`` fail,
    ``E`` error, ``s`` skip, ``x``/``X`` xfail/xpass. We count those between
    the ``== unit tests ==`` and ``== end-to-end`` banners; a ``N passed``
    summary line, when present, takes precedence.
    """
    text = _strip_ansi(output)
    vs = VerifySummary(all_green=("VERIFY: ALL GREEN" in text and returncode == 0))
    block = text
    if "== unit tests ==" in text:
        block = text.split("== unit tests ==", 1)[1]
        block = block.split("== end-to-end", 1)[0]
    m = re.search(r"(\d+) passed", block)
    if m:
        vs.passed = int(m.group(1))
        for key, attr in (("failed", "failed"), ("error", "errors"), ("skipped", "skipped")):
            mm = re.search(rf"(\d+) {key}", block)
            if mm:
                setattr(vs, attr, int(mm.group(1)))
    else:
        glyphs = "".join(re.sub(r"\[\s*\d+%\]", "", ln).strip()
                         for ln in block.splitlines()
                         if ln.strip() and re.fullmatch(r"[.FEsxX ]+(\[\s*\d+%\])?", ln.strip()))
        vs.passed = glyphs.count(".")
        vs.failed = glyphs.count("F")
        vs.errors = glyphs.count("E")
        vs.skipped = glyphs.count("s")
    for ln in text.splitlines():
        if ln.startswith(("FAILED", "ERROR")):
            vs.details.append(ln.strip())
    return vs


def _load_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# --------------------------------------------------------- collection ----
def collect(root: pathlib.Path, demo_started_utc: datetime | None) -> dict[str, Any]:
    """Gather every artifact the summary reports. Raises MvpError if the demo
    did not produce ``metrics/demo_last_run.json`` (or produced it before
    ``demo_started_utc``)."""
    run_path = root / "metrics" / "demo_last_run.json"
    run = _load_json(run_path)
    if run is None:
        raise MvpError(f"refusing to write the MVP summary: {run_path} does not exist - "
                       "the demo did not produce its metrics")
    gen = run.get("_generated_utc")
    if demo_started_utc is not None:
        if not gen:
            raise MvpError("refusing to write the MVP summary: demo_last_run.json has no "
                           "_generated_utc stamp")
        if _parse_iso(gen) < demo_started_utc.replace(microsecond=0):
            raise MvpError("refusing to write the MVP summary: metrics/demo_last_run.json "
                           f"is stale (generated {gen}, demo started "
                           f"{demo_started_utc.isoformat(timespec='seconds')})")
    if run.get("scenario") != "ridgeline" or run.get("windshift_deg") != 40.0 \
            or run.get("killed_node") != "N3":
        raise MvpError("refusing to write the MVP summary: demo_last_run.json was not "
                       "produced by the canonical run (ridgeline, --wind-shift 40, "
                       f"--kill-node N3); got {run.get('scenario')!r}, "
                       f"{run.get('windshift_deg')!r}, {run.get('killed_node')!r}")

    out = root / "demo" / "out"
    frames_dir = out / "demo_frames"
    pitch = {name: out / "pitch_assets" / f"{name}.png" for name in PITCH_ASSETS}
    outbox = root / "outbox"
    caps = sorted(outbox.glob("cap_*.json"), key=lambda p: p.stat().st_mtime) if outbox.exists() else []
    cap_path = caps[-1] if caps else None
    cap_doc = _load_json(cap_path) if cap_path else None

    return {
        "run": run,
        "surrogate": _load_json(root / "metrics" / "surrogate.json"),
        "detect": _load_json(root / "metrics" / "detect.json"),
        "calibration": _load_json(root / "metrics" / "calibration.json"),
        "hindcast": _load_json(root / "metrics" / "hindcast.json"),
        "artifacts": {
            "gif": out / "demo.gif",
            "frames_dir": frames_dir,
            "n_frames": len(list(frames_dir.glob("frame_*.png"))) if frames_dir.exists() else 0,
            "npz": out / "last_run_artifacts.npz",
            "json": out / "last_run_artifacts.json",
            "pitch": pitch,
            "cap": cap_path,
            "cap_status": (cap_doc or {}).get("status"),
            "metrics_run": run_path,
        },
    }


def _git(root: pathlib.Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _rel(root: pathlib.Path, p: pathlib.Path | None) -> str:
    if p is None:
        return "—"
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def report_crosscheck(root: pathlib.Path, info: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """Does REPORT.md still quote the numbers the metrics JSON hold?"""
    rp = root / "REPORT.md"
    text = rp.read_text(encoding="utf-8") if rp.exists() else ""
    checks: list[tuple[str, str, bool]] = []
    sur, det = info.get("surrogate"), info.get("detect")
    if sur:
        for label, val in (("val IoU@+30", sur["val"]["iou_30"]),
                           ("held-out IoU@+30", sur["holdout_wind_regime"]["iou_30"]),
                           ("p5 IoU@+30", sur["worst_case_p5_iou_30"])):
            s = f"{val:.3f}"
            checks.append((label, s, s in text))
        s = f"{sur['speedup']['speedup_x']:.1f}x"
        checks.append(("speedup", s, s in text))
    if det:
        for label, val in (("CNN F1", det["cnn"]["f1"]), ("GBM F1", det["gbm"]["f1"])):
            s = f"{val:.3f}"
            checks.append((label, s, s in text))
        s = f"{det['ambient_day']['fp_per_node_day_cnn']:.2f}"
        checks.append(("FP/node-day (CNN)", s, s in text))
    cal = info.get("calibration")
    if cal:
        s = f"{cal['ece']:.3f}"
        checks.append(("ECE raw", s, s in text))
    return checks


# ------------------------------------------------------------ summary ----
def build_summary(info: dict[str, Any], steps: list[StepResult], verify: VerifySummary,
                  root: pathlib.Path, started: datetime, finished: datetime,
                  scoreboard_text: str = "") -> str:
    run = info["run"]
    art = info["artifacts"]
    sur, det = info.get("surrogate"), info.get("detect")

    def rv(key: str, fmt: str = "{}") -> str:
        v = run.get(key)
        return "—" if v is None else fmt.format(v)

    lines: list[str] = []
    a = lines.append
    a("# Emberline MVP run summary")
    a("")
    a(f"Generated {finished.isoformat(timespec='seconds')} by `python -m emberline.mvp` "
      f"(started {started.isoformat(timespec='seconds')}, "
      f"{(finished - started).total_seconds():.0f} s wall).")
    a("")
    a("> Every value on this page was read from an artifact this run produced or from "
      "`metrics/*.json`; none was typed in. **Everything is synthetic simulation** - see "
      "[03_RESULTS.md](03_RESULTS.md) and [04_GAP_REGISTER.md](04_GAP_REGISTER.md).")
    a("")
    a("## Environment")
    a("")
    a("| item | value |")
    a("|---|---|")
    a(f"| commit | `{_git(root, 'rev-parse', '--short', 'HEAD')}` on `{_git(root, 'branch', '--show-current')}` |")
    a(f"| platform | {platform.platform()} |")
    a(f"| python | {platform.python_version()} |")
    try:
        import torch  # noqa: WPS433

        a(f"| torch | {torch.__version__} |")
    except Exception:
        pass
    a(f"| cpu count | {os.cpu_count()} |")
    a("")
    a("## Steps")
    a("")
    a("| step | command | result | wall |")
    a("|---|---|---|---|")
    for s in steps:
        res = "skipped" if s.skipped else ("OK" if s.returncode == 0 else f"exit {s.returncode}")
        a(f"| {s.name} | `{s.command}` | {res} | {s.seconds:.0f} s |")
    a("")
    a("## Test suite")
    a("")
    vline = (f"**{_mark(verify.all_green)}** - {verify.passed} passed, {verify.failed} failed, "
             f"{verify.errors} errors, {verify.skipped} skipped; "
             f"`VERIFY: ALL GREEN` {'printed' if verify.all_green else 'NOT printed'}.")
    a(vline)
    for d in verify.details[:20]:
        a(f"- `{d}`")
    a("")
    a("## Scoreboard of this run (`metrics/demo_last_run.json`)")
    a("")
    a("| quantity | value |")
    a("|---|---|")
    a(f"| scenario / forecast backend | {run.get('scenario')} / {run.get('backend')} |")
    a(f"| ignition → Tier-0 chirp | {rv('tier0_sim_s_after_ignition', '{:.0f} s')} |")
    a(f"| ignition → Tier-1 voice + bearing | {rv('tier1_sim_s_after_ignition', '{:.0f} s')} |")
    a(f"| ignition → Tier-2 full cascade | {rv('tier2_sim_s_after_ignition', '{:.0f} s')} |")
    a(f"| ignition estimate error | {rv('ignition_estimate_error_m', '{} m')} |")
    a(f"| cascade max hops / median alert latency | {rv('cascade_max_hops')} / "
      f"{rv('cascade_median_latency_s', '{} s')} |")
    a(f"| node killed / at / self-heal announced at | {rv('killed_node')} / "
      f"{rv('kill_sim_s_after_ignition', 't+{} s')} / {rv('selfheal_sim_s_after_ignition', 't+{} s')} |")
    a(f"| wind shift / at | {rv('windshift_deg', '{:+.0f}°')} / {rv('windshift_sim_s_after_ignition', 't+{} s')} |")
    a(f"| access points moved to another exit by the re-plan | {rv('replan_moved_access_points')} |")
    a(f"| forecast wall-clock per ensemble | {rv('forecast_wall_s', '{} s')} |")
    a(f"| mesh channel utilisation | {rv('channel_utilization', '{:.2%}')} |")
    a(f"| mesh transmissions / collisions | {rv('mesh_tx_total')} / {rv('mesh_collisions')} |")
    a(f"| truth burned area at end | {rv('truth_burned_ha_at_end', '{} ha')} |")
    a("")
    a("## Standing metrics the scoreboard read (`metrics/surrogate.json`, `metrics/detect.json`)")
    a("")
    a("| quantity | value |")
    a("|---|---|")
    if sur:
        v = sur["val"]
        a(f"| surrogate IoU +10/+30/+60, val worlds | {v['iou_10']:.3f} / {v['iou_30']:.3f} / {v['iou_60']:.3f} |")
        a(f"| surrogate IoU@+30, held-out wind regime | {sur['holdout_wind_regime']['iou_30']:.3f} |")
        a(f"| worst-case p5 IoU@+30 | {sur['worst_case_p5_iou_30']:.3f} |")
        a(f"| arrival MAE (val) | {v['arrival_mae_min']:.1f} min |")
        a(f"| ensemble speedup vs physics | {sur['speedup']['speedup_x']:.1f}× |")
    else:
        a("| surrogate metrics | metrics/surrogate.json missing |")
    if det:
        a(f"| detector F1, CNN / GBM | {det['cnn']['f1']:.3f} / {det['gbm']['f1']:.3f} |")
        a(f"| false positives per node-day (CNN) | {det['ambient_day']['fp_per_node_day_cnn']:.2f} |")
        a(f"| detection latency median | {det['latency']['latency_median_s']:.0f} s |")
    else:
        a("| detection metrics | metrics/detect.json missing |")
    a("")
    checks = report_crosscheck(root, info)
    if checks:
        a("Cross-check that REPORT.md still quotes these values: "
          + ", ".join(f"{lbl} {val} {_mark(ok)}" for lbl, val, ok in checks) + ".")
        a("")
    if scoreboard_text:
        a("### Demo scoreboard as printed")
        a("")
        a("```text")
        a(scoreboard_text.rstrip())
        a("```")
        a("")
    a("## Artifacts")
    a("")
    a("| artifact | path | present |")
    a("|---|---|---|")
    a(f"| demo GIF | `{_rel(root, art['gif'])}` | {'yes' if art['gif'].exists() else 'NO'} |")
    a(f"| demo frames | `{_rel(root, art['frames_dir'])}/` ({art['n_frames']} PNGs) | {'yes' if art['n_frames'] else 'NO'} |")
    a(f"| raw run arrays / plans | `{_rel(root, art['npz'])}`, `{_rel(root, art['json'])}` | "
      f"{'yes' if art['npz'].exists() and art['json'].exists() else 'NO'} |")
    a(f"| run metrics | `{_rel(root, art['metrics_run'])}` | yes |")
    for name, p in art["pitch"].items():
        a(f"| pitch asset `{name}` | `{_rel(root, p)}` | {'yes' if p.exists() else 'NO'} |")
    cap = art["cap"]
    a(f"| CAP draft (newest in outbox/) | `{_rel(root, cap)}` | "
      f"{('yes, status ' + str(art['cap_status'])) if cap else 'NO'} |")
    a("")
    a("The CAP draft is a JSON document with `status: \"Exercise\"`; nothing transmits it. "
      "Frames, GIF and the outbox are gitignored; the six pitch PNGs are committed.")
    a("")
    a("---")
    a("")
    a("Synthetic simulation only. No physical node, radio, real sensor data, real terrain "
      "or real fire record was involved in producing any number above. "
      "Index: [README.md](../README.md).")
    return "\n".join(lines) + "\n"


def write_summary(root: pathlib.Path, text: str) -> pathlib.Path:
    path = root / "docs" / "MVP_RUN.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def extract_scoreboard(demo_output: str) -> str:
    text = _strip_ansi(demo_output)
    if "SCOREBOARD" not in text:
        return ""
    block = text.split("SCOREBOARD", 1)[1]
    block = block.split("Synthetic simulation.", 1)[0]
    lines = block.splitlines()[1:]  # drop the rest of the rule line
    return "\n".join(ln.rstrip() for ln in lines if ln.strip())


# ---------------------------------------------------------------- main ----
def run(skip_verify: bool = False, summary_only: bool = False,
        root: pathlib.Path | None = None) -> int:
    root = root or repo_root()
    started = _utcnow()
    steps: list[StepResult] = []
    verify = VerifySummary()
    scoreboard = ""
    demo_started: datetime | None = None
    py = sys.executable

    if summary_only:
        steps.append(StepResult("verify", "(summary-only)", None, 0.0, skipped=True))
        steps.append(StepResult("demo", "(summary-only)", None, 0.0, skipped=True))
        steps.append(StepResult("pitch assets", "(summary-only)", None, 0.0, skipped=True))
    else:
        if skip_verify:
            steps.append(StepResult("verify", "(skipped by --skip-verify)", None, 0.0, skipped=True))
        else:
            s = _run_step("verify", verify_command(), root)
            steps.append(s)
            verify = parse_verify(s.output, s.returncode)
        demo_started = _utcnow()
        s = _run_step("demo", [py, "-m", "emberline.demo", *DEMO_ARGS], root)
        steps.append(s)
        scoreboard = extract_scoreboard(s.output)
        s = _run_step("pitch assets", [py, "-m", "emberline.demo.pitch_assets"], root)
        steps.append(s)

    info = collect(root, demo_started)  # raises MvpError -> nothing is written
    finished = _utcnow()
    text = build_summary(info, steps, verify, root, started, finished, scoreboard)
    path = write_summary(root, text)
    print(f"\n== mvp: wrote {path} ({(finished - started).total_seconds():.0f} s total)")
    failed = [s for s in steps if not s.ok]
    if failed:
        print("== mvp: steps with non-zero exit: " + ", ".join(s.name for s in failed))
        return 1
    if not summary_only and not skip_verify and not verify.all_green:
        print("== mvp: verify did not print ALL GREEN")
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Emberline one-command MVP run")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip make verify (faster iteration; summary records it as skipped)")
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild docs/MVP_RUN.md from existing artifacts without running anything")
    args = ap.parse_args()
    try:
        sys.exit(run(skip_verify=args.skip_verify, summary_only=args.summary_only))
    except MvpError as e:
        print(f"mvp: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
