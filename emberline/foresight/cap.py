"""CAP-style geo-targeted alert drafts -> outbox/ for HUMAN approval.

We emit a JSON document shaped after the OASIS Common Alerting Protocol 1.2
(the schema real IPAWS/EAS pipelines speak), with two deliberate deviations,
both stated in the document itself:

* ``status`` is always ``"Exercise"`` - nothing this repo produces is a real
  alert, and a downstream integrator must flip it consciously;
* the area polygon is in local grid metres (crs field says so), because our
  worlds are synthetic and have no geodetic datum.

``validate_cap`` is a small hand-rolled structural validator (required
fields, types, enums, polygon closure) against ``CAP_SCHEMA`` below - the
acceptance test validates every draft we write.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..config import repo_root

CAP_SCHEMA: dict[str, Any] = {
    "required": {
        "identifier": str, "sender": str, "sent": str, "status": str,
        "msgType": str, "scope": str, "info": dict,
    },
    "status_enum": ["Actual", "Exercise", "Test", "Draft"],
    "msgType_enum": ["Alert", "Update", "Cancel"],
    "info_required": {
        "category": str, "event": str, "urgency": str, "severity": str,
        "certainty": str, "headline": str, "description": str,
        "instruction": str, "area": dict,
    },
    "area_required": {"areaDesc": str, "polygon": list, "crs": str},
}


def validate_cap(doc: dict[str, Any]) -> list[str]:
    """Structural validation; returns a list of problems (empty = valid)."""
    errs: list[str] = []
    for k, t in CAP_SCHEMA["required"].items():
        if k not in doc:
            errs.append(f"missing field {k}")
        elif not isinstance(doc[k], t):
            errs.append(f"field {k} must be {t.__name__}")
    if doc.get("status") not in CAP_SCHEMA["status_enum"]:
        errs.append(f"status must be one of {CAP_SCHEMA['status_enum']}")
    if doc.get("msgType") not in CAP_SCHEMA["msgType_enum"]:
        errs.append(f"msgType must be one of {CAP_SCHEMA['msgType_enum']}")
    info = doc.get("info", {})
    for k, t in CAP_SCHEMA["info_required"].items():
        if k not in info:
            errs.append(f"missing info.{k}")
        elif not isinstance(info[k], t):
            errs.append(f"info.{k} must be {t.__name__}")
    area = info.get("area", {})
    for k, t in CAP_SCHEMA["area_required"].items():
        if k not in area:
            errs.append(f"missing info.area.{k}")
    poly = area.get("polygon", [])
    if len(poly) < 4:
        errs.append("polygon needs >= 4 vertices")
    elif poly[0] != poly[-1]:
        errs.append("polygon must be closed (first == last vertex)")
    return errs


def cone_polygon(cone_mask: np.ndarray, cell_m: float) -> list[list[float]]:
    """Closed convex-hull polygon (metres, local grid) around the cone."""
    from scipy.spatial import ConvexHull

    rr, cc = np.where(cone_mask)
    if len(rr) < 3:
        return []
    pts = np.stack([cc * cell_m, rr * cell_m], axis=1)  # (x_east, y_north)
    hull = ConvexHull(pts)
    poly = [[float(x), float(y)] for x, y in pts[hull.vertices]]
    poly.append(poly[0])
    return poly


def draft_cap_alert(
    cfg: dict[str, Any],
    incident_id: int,
    tier: int,
    cone_mask: np.ndarray,
    cell_m: float,
    routes_note: str,
    out_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Write a CAP-style draft JSON to outbox/; returns the path.

    The draft is exactly that - a draft. It sits in outbox/ until a human
    reviews it; nothing in Emberline transmits it anywhere.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sev = {1: "Severe", 2: "Extreme"}.get(tier, "Moderate")
    doc = {
        "identifier": f"EMBERLINE-SIM-{incident_id}-{tier}",
        "sender": "emberline-foresight@simulation.invalid",
        "sent": now,
        "status": "Exercise",  # ALWAYS: this system is a simulation
        "msgType": "Alert",
        "scope": "Public",
        "note": "SYNTHETIC DRAFT - requires human approval; ADVISORY decision support",
        "info": {
            "category": "Fire",
            "event": "Wildfire (simulated)",
            "urgency": "Immediate" if tier >= 2 else "Expected",
            "severity": sev,
            "certainty": "Likely" if tier >= 2 else "Possible",
            "headline": f"[SIM] Wildfire threat - mesh Tier {tier} corroborated detection",
            "description": ("Sensor-mesh corroborated smoke detection; surrogate ensemble "
                            "projects the attached probability cone over the next 60 minutes."),
            "instruction": ("ADVISORY: residents inside or downwind of the polygon should "
                            "prepare to evacuate via posted routes. " + routes_note),
            "area": {
                "areaDesc": "Fire probability cone (+60 min, dilated safety margin)",
                "polygon": cone_polygon(cone_mask, cell_m),
                "crs": "EMBERLINE-LOCAL-METRES (synthetic world; no geodetic datum)",
            },
        },
    }
    errs = validate_cap(doc)
    assert not errs, f"internal CAP draft failed validation: {errs}"
    out = out_dir or (repo_root() / "outbox")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cap_{incident_id}_tier{tier}_{now.replace(':', '')}.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path
