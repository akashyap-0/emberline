"""Compact fully-convolutional UNet fire-spread surrogate (<= 5M params).

Why a UNet: one surrogate step maps (fire state, terrain, fuel, wind) at t to
the fire state at t+10 min. Spread physics is local (a front moves at most a
couple hundred metres in 10 min), so a convolutional encoder-decoder with a
receptive field of a few dozen cells captures it; skip connections preserve
the sharp front geometry that a plain encoder would blur. Because the network
is fully convolutional we can *train on 64x64 crops around the front* (cheap
CPU steps) and *evaluate on the full 256x256 grid* unchanged.

Outputs are logits for two planes: ``touched`` (burned|burning) and
``burning`` at t+10. During autoregressive rollout we clamp
``touched := max(touched, touched_prev)`` so the surrogate provably respects
the physics invariant that burned cells never unburn.

GroupNorm instead of BatchNorm: batch statistics are unstable with small
CPU batches and crop training / full-grid eval have very different
activation statistics; GroupNorm is resolution- and batch-agnostic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

IN_CHANNELS = 10  # touched, burning, elev, fuel one-hot x5, wind u, v
OUT_CHANNELS = 2  # touched_{t+1}, burning_{t+1} logits
MAX_PARAMS = 5_000_000

ELEV_SCALE = 150.0  # normalization constants shared by training and rollout
WIND_SCALE = 15.0


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(),
    )


class FireUNet(nn.Module):
    """3-level UNet: base -> 2x -> 4x channels, stride-2 down, bilinear up."""

    def __init__(self, base: int = 24) -> None:
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.enc1 = _block(IN_CHANNELS, c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = _block(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid = _block(c3, c3)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _block(c3 + c2, c2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _block(c2 + c1, c1)
        self.head = nn.Conv2d(c1, OUT_CHANNELS, 1)
        n = sum(p.numel() for p in self.parameters())
        assert n <= MAX_PARAMS, f"surrogate has {n} params, cap is {MAX_PARAMS}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

    @torch.no_grad()
    def rollout_step(self, touched: torch.Tensor, burning: torch.Tensor,
                     static: torch.Tensor, wind_uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One +10 min autoregressive step on full grids.

        Parameters: ``touched``/``burning`` (B,1,H,W) in {0,1};
        ``static`` (B,6,H,W) = [elev_norm, fuel one-hot]; ``wind_uv`` (B,2)
        m/s. Returns hard-thresholded next (touched, burning) with the
        monotonicity clamp applied.
        """
        b, _, h, w = touched.shape
        wind = (wind_uv / WIND_SCALE).view(b, 2, 1, 1).expand(b, 2, h, w)
        x = torch.cat([touched, burning, static, wind], dim=1)
        logits = self.forward(x)
        prob = torch.sigmoid(logits)
        nxt_touched = torch.maximum((prob[:, :1] > 0.5).float(), touched)
        nxt_burning = (prob[:, 1:2] > 0.5).float() * nxt_touched
        return nxt_touched, nxt_burning


def build_static_planes(elevation, fuel) -> torch.Tensor:
    """(6,H,W) float32: normalized elevation + fuel one-hot. numpy in, torch out."""
    import numpy as np

    elev = (elevation - elevation.mean()) / ELEV_SCALE
    onehot = np.eye(5, dtype=np.float32)[fuel.astype(np.int64)]  # (H,W,5)
    static = np.concatenate([elev[None].astype(np.float32),
                             np.moveaxis(onehot, -1, 0)], axis=0)
    return torch.from_numpy(static)
