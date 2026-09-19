"""Small UNet for next-day fire-mask prediction on 64x64 NDWS tiles.

~1.9M parameters (cap: 5M, asserted in tests) — three encoder stages, a
256-channel bottleneck, skip connections, logit output per pixel.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.SiLU(),
    )


class SpreadUNet(nn.Module):
    def __init__(self, in_channels: int = 12) -> None:
        super().__init__()
        self.enc1 = _block(in_channels, 32)
        self.enc2 = _block(32, 64)
        self.enc3 = _block(64, 128)
        self.bott = _block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                      # 64
        e2 = self.enc2(self.pool(e1))          # 32
        e3 = self.enc3(self.pool(e2))          # 16
        b = self.bott(self.pool(e3))           # 8
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1).squeeze(1)        # (N, 64, 64) logits


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
