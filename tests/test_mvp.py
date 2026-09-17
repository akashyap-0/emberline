"""MVP entrypoint tests: the summary is built from real artifacts and is
refused when the demo did not produce ``metrics/demo_last_run.json``."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone

import pytest

from emberline import mvp
from emberline.config import repo_root


def _fake_root(tmp_path, with_run: bool = True, generated: datetime | None = None,
               canonical: bool = True):
    root = tmp_path / "repo"
    (root / "metrics").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "demo" / "out").mkdir(parents=True)
    if with_run:
        gen = generated or datetime.now(timezone.utc)
        run = {"_generated_utc": gen.isoformat(timespec="seconds"),
               "scenario": "ridgeline" if canonical else "valley",
               "windshift_deg": 40.0, "killed_node": "N3", "backend": "physics",
               "tier0_sim_s_after_ignition": 30, "tier1_sim_s_after_ignition": 120,
               "tier2_sim_s_after_ignition": 300, "channel_utilization": 0.0021,
               "mesh_tx_total": 1, "mesh_collisions": 0, "truth_burned_ha_at_end": 1.0}
        (root / "metrics" / "demo_last_run.json").write_text(json.dumps(run))
    return root


def test_summary_is_built_from_real_repo_artifacts():
    """Uses the committed metrics/demo_last_run.json of the canonical run."""
    root = repo_root()
    if not (root / "metrics" / "demo_last_run.json").exists():
        pytest.skip("no demo_last_run.json in this checkout")
    info = mvp.collect(root, demo_started_utc=None)
    run = info["run"]
    text = mvp.build_summary(info, steps=[], verify=mvp.VerifySummary(), root=root,
                             started=datetime.now(timezone.utc),
                             finished=datetime.now(timezone.utc))
    # Numbers in the page are the numbers in the artifact - not typed in.
    assert f"{run['tier2_sim_s_after_ignition']:.0f} s" in text
    assert f"{run['ignition_estimate_error_m']} m" in text
    assert "synthetic simulation" in text.lower()
    sur = info.get("surrogate")
    if sur:
        assert f"{sur['val']['iou_30']:.3f}" in text


def test_refuses_without_demo_metrics(tmp_path):
    root = _fake_root(tmp_path, with_run=False)
    with pytest.raises(mvp.MvpError, match="does not exist"):
        mvp.collect(root, demo_started_utc=datetime.now(timezone.utc))
    assert not (root / "docs" / "MVP_RUN.md").exists()


def test_refuses_stale_demo_metrics(tmp_path):
    """A demo_last_run.json older than the demo start means the demo did not
    produce it this run - the summary must not be written from it."""
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    root = _fake_root(tmp_path, generated=old)
    with pytest.raises(mvp.MvpError, match="stale"):
        mvp.collect(root, demo_started_utc=datetime.now(timezone.utc))
    assert not (root / "docs" / "MVP_RUN.md").exists()


def test_refuses_non_canonical_run(tmp_path):
    root = _fake_root(tmp_path, canonical=False)
    with pytest.raises(mvp.MvpError, match="canonical"):
        mvp.collect(root, demo_started_utc=None)


def test_fresh_metrics_accepted_and_written(tmp_path):
    root = _fake_root(tmp_path)
    started = datetime.now(timezone.utc) - timedelta(seconds=5)
    info = mvp.collect(root, demo_started_utc=started)
    text = mvp.build_summary(info, steps=[], verify=mvp.VerifySummary(passed=1, all_green=True),
                             root=root, started=started, finished=datetime.now(timezone.utc))
    path = mvp.write_summary(root, text)
    assert path.exists() and "PASS" in path.read_text(encoding="utf-8")


def test_parse_verify_counts_glyphs_and_green():
    out = ("== lint ==\n== unit tests ==\n"
           "......................................................                   [100%]\n"
           "== end-to-end smoke ==\n...\nVERIFY: ALL GREEN\n")
    vs = mvp.parse_verify(out, 0)
    assert vs.passed == 54 and vs.failed == 0 and vs.all_green
    vs2 = mvp.parse_verify(out.replace("VERIFY: ALL GREEN", ""), 1)
    assert not vs2.all_green
    vs3 = mvp.parse_verify("== unit tests ==\n..F.\n== end-to-end\n", 1)
    assert vs3.passed == 3 and vs3.failed == 1


def test_verify_command_resolves():
    cmd = mvp.verify_command()
    assert cmd[-1] in ("verify", "verify.sh")
    assert shutil.which(cmd[0]) or __import__("pathlib").Path(cmd[0]).exists()
