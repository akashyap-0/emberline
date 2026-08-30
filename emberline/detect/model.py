"""Smoke classifiers: small 1D-CNN and a gradient-boosted baseline.

The CNN sees the raw (normalised) 4-channel 60-s window; convolutions can
pick up temporal shape (sustained growth vs spike-decay) that fixed features
summarise lossily. The GBM baseline gets exactly the summary statistics a
sensible engineer would handcraft; if the CNN cannot beat it, the honest
conclusion is that 60 s of context is feature-summarisable and we ship the
cheaper model (the report states whichever way it lands).

Normalisation is fixed (not fit on data) so a deployed node needs no
calibration pass: channels are scaled to O(1) by physical ranges.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# (offset, scale) per channel: pm25, voc, temp, rh
NORM = np.array([[10.0, 25.0], [0.3, 1.0], [15.0, 10.0], [60.0, 25.0]], dtype=np.float32)


def normalize(x: np.ndarray) -> np.ndarray:
    """(N, 4, 60) raw units -> O(1) floats."""
    return (x - NORM[:, 0][None, :, None]) / NORM[:, 1][None, :, None]


class SmokeCNN(nn.Module):
    """~30k-param 1D-CNN: 3 conv stages, avg+max global pooling, linear head.

    Dual pooling matters here: average pooling summarises sustained level
    (fire smoke grows and stays), max pooling keeps transient spike shape
    (vehicle/aerosol confounders) - together they cover both failure modes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 24, 5, padding=2), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(24, 32, 5, padding=2), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 48, 3, padding=1), nn.SiLU(),
        )
        self.head = nn.Linear(96, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def confidence(self, window_raw: np.ndarray) -> float:
        """P(fire smoke) for one raw (4, 60) window - the mesh's input."""
        x = torch.from_numpy(normalize(window_raw[None].astype(np.float32)))
        was_training = self.training
        self.eval()
        p = float(torch.sigmoid(self.forward(x)))
        self.train(was_training)
        return p


def gbm_features(x: np.ndarray) -> np.ndarray:
    """(N, 4, 60) -> (N, F) handcrafted features for the GBM baseline.

    Per channel: mean, std, max, linear slope, last-minus-first. Plus
    cross-channel cues that encode the confounder signatures: VOC/PM ratio
    (bbq/aerosol high, dust zero), RH level (fog), PM x VOC co-trend (fire).
    """
    t = np.arange(x.shape[2], dtype=np.float32)
    t = (t - t.mean()) / t.std()
    feats = []
    for c in range(4):
        ch = x[:, c, :]
        slope = (ch * t).mean(axis=1)
        feats += [ch.mean(1), ch.std(1), ch.max(1), slope, ch[:, -10:].mean(1) - ch[:, :10].mean(1)]
    pm, voc, rh = x[:, 0], x[:, 1], x[:, 3]
    feats.append(voc.mean(1) / (pm.mean(1) + 1.0))
    feats.append(rh.mean(1))
    pm_s = (pm * t).mean(1)
    voc_s = (voc * t).mean(1)
    feats.append(pm_s * voc_s)
    return np.stack(feats, axis=1).astype(np.float32)
