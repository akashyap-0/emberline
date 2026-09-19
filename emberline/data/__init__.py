"""Real-data loading layer.

Every model trained on real data loads its bytes through this package, and
this package refuses to touch synthetic data. The rule of the round: real
training uses REAL DATA ONLY — no synthetic samples, no synthetic-pretrained
checkpoints. The synthetic simulation stack elsewhere in the repo stays
untouched, but nothing under here may read from it.

The guard is `assert_real_data_path`, called by every loader before opening
a file. It raises `SyntheticDataError` for any path under a ``data/synthetic``
directory, and more broadly for any data path outside ``data/real/``. Tests
in tests/test_real_data_infra.py prove the guard fires.
"""

from __future__ import annotations

from pathlib import Path

# Repo root = three levels up from this file (emberline/data/__init__.py).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_DATA_ROOT = REPO_ROOT / "data" / "real"


class SyntheticDataError(RuntimeError):
    """Raised when real-data code is pointed at synthetic (or non-real) data."""


def assert_real_data_path(path: str | Path) -> Path:
    """Validate that *path* is a legitimate real-data location.

    Returns the resolved path. Raises SyntheticDataError if the path lies
    under any ``data/synthetic`` directory, or under the repo's ``data/``
    tree but outside ``data/real/`` (the synthetic stack keeps its worlds
    and checkpoints in other ``data/`` subdirectories, so those are refused
    too). Paths entirely outside the repo's ``data/`` tree (e.g. a tmp dir
    in a test fixture) are allowed — the guard exists to stop real-data
    training code from reaching the synthetic stack, not to forbid scratch
    space.
    """
    p = Path(path).resolve()
    parts_lower = [s.lower() for s in p.parts]
    for i in range(len(parts_lower) - 1):
        if parts_lower[i] == "data" and parts_lower[i + 1] == "synthetic":
            raise SyntheticDataError(
                f"Refusing synthetic data path: {p}\n"
                "Real-data training uses real data only (see data/real/README.md). "
                "Nothing under a data/synthetic directory may be read here."
            )
    data_root = (REPO_ROOT / "data").resolve()
    try:
        rel = p.relative_to(data_root)
    except ValueError:
        return p  # outside the repo data tree: fine (tests, scratch)
    if rel.parts and rel.parts[0] != "real":
        raise SyntheticDataError(
            f"Refusing non-real repo data path: {p}\n"
            f"data/{rel.parts[0]}/ belongs to the synthetic simulation stack; "
            "real-data loaders only read under data/real/."
        )
    return p
