"""On-node smoke detection: 60-s window classifier {fire-smoke, non-fire}.

Pipeline: ``python -m emberline.detect.data`` (windows) ->
``python -m emberline.detect.train`` (1D-CNN + GBM baseline) ->
``python -m emberline.detect.eval`` (metrics -> REPORT.md).
"""

from __future__ import annotations

import pathlib
from typing import Any

from ..config import repo_root


def windows_path(cfg: dict[str, Any]) -> pathlib.Path:
    return repo_root() / cfg["detect"]["dataset"]["dir"] / "windows.npz"


def cnn_ckpt_path(cfg: dict[str, Any]) -> pathlib.Path:
    return repo_root() / cfg["surrogate"]["train"]["dir"] / "detect_cnn.pt"


def gbm_ckpt_path(cfg: dict[str, Any]) -> pathlib.Path:
    return repo_root() / cfg["surrogate"]["train"]["dir"] / "detect_gbm.joblib"
