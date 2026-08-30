"""REPORT.md section management + metrics JSON store.

Every measured number lives twice: as JSON under ``metrics/`` (machine-read
by the demo scoreboard) and as a human-readable section in ``REPORT.md``
delimited by ``<!-- BEGIN name --> ... <!-- END name -->`` markers so each
phase can regenerate its own section without clobbering others. Nothing
writes to REPORT.md except through this module, which is how we keep the
"never fabricate a metric" rule auditable: grep for update_section callers.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any

from .config import repo_root

_HEADER = """# Emberline — Measured Results

All numbers below were produced by code in this repository running on
**synthetic, procedurally generated worlds**. Nothing here is a claim about
real-fire detection or real-world performance. Regenerate any section with
the command noted inside it.
"""


def metrics_dir() -> pathlib.Path:
    d = repo_root() / "metrics"
    d.mkdir(exist_ok=True)
    return d


def save_metrics(name: str, payload: dict[str, Any]) -> pathlib.Path:
    payload = dict(payload)
    payload["_generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p = metrics_dir() / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p


def load_metrics(name: str) -> dict[str, Any] | None:
    p = metrics_dir() / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def update_section(name: str, body: str) -> None:
    """Insert or replace the named marker-delimited section of REPORT.md."""
    path = repo_root() / "REPORT.md"
    begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    block = f"{begin}\n{body.strip()}\n{end}"
    text = path.read_text(encoding="utf-8") if path.exists() else _HEADER
    if begin in text and end in text:
        pre = text.split(begin)[0]
        post = text.split(end, 1)[1]
        text = pre + block + post
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")
