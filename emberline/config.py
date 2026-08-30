"""Central config loading and seeding.

One ``config.yaml`` at the repo root controls the whole project. All
randomness flows from ``cfg["seed"]`` through :func:`rng_for`, which derives
independent-but-deterministic child streams per subsystem, so re-running any
stage with the same config reproduces its outputs exactly.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.yaml"


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Load the project config YAML (defaults to repo-root ``config.yaml``)."""
    p = pathlib.Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def rng_for(cfg: dict[str, Any], stream: str, extra: int = 0) -> np.random.Generator:
    """Deterministic child RNG for a named subsystem.

    Uses ``SeedSequence.spawn``-style keying: the global seed plus a stable
    hash of the stream name (and an optional integer, e.g. a world index) so
    subsystems never share or perturb each other's random state.
    """
    seed = int(cfg["seed"])
    key = np.frombuffer(stream.encode(), dtype=np.uint8).sum() * 100003
    ss = np.random.SeedSequence([seed, int(key), int(extra)])
    return np.random.default_rng(ss)


def repo_root() -> pathlib.Path:
    """Absolute path of the repository root."""
    return _REPO_ROOT
