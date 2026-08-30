"""Autoregressive surrogate rollout + batched surrogate ensembles.

The surrogate advances fire state in +10 min strides; +30/+60 forecasts are
3/6 autoregressive steps. An ensemble runs ALL members as one batch through
the network (members only differ in their fire-state planes and wind
scalars), which is where the surrogate's wall-clock advantage over running
N independent physics simulations comes from.
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable

import numpy as np
import torch

from ..config import repo_root
from ..firesim.ensemble import EnsembleResult
from ..worldgen import World
from .data import SNAP_MIN
from .model import FireUNet, build_static_planes


def default_ckpt(cfg: dict[str, Any]) -> pathlib.Path:
    return repo_root() / cfg["surrogate"]["train"]["dir"] / "best.pt"


class SurrogateEngine:
    """Loads the trained checkpoint and serves rollouts/ensembles."""

    def __init__(self, cfg: dict[str, Any], ckpt: pathlib.Path | None = None) -> None:
        torch.set_num_threads(4)
        self.cfg = cfg
        self.model = FireUNet(int(cfg["surrogate"]["model"]["base_channels"]))
        path = ckpt or default_ckpt(cfg)
        state = torch.load(path, weights_only=False, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.trained_steps = int(state.get("step", -1))
        self.val_iou = float(state.get("best_iou", float("nan")))

    @staticmethod
    def available(cfg: dict[str, Any]) -> bool:
        return default_ckpt(cfg).exists()

    @torch.no_grad()
    def rollout(
        self,
        static: torch.Tensor,
        touched0: np.ndarray,
        burning0: np.ndarray,
        n_steps: int,
        wind_fn: Callable[[int], tuple[float, float]],
    ) -> list[np.ndarray]:
        """Single-fire rollout. ``wind_fn(k)`` -> (u, v) m/s at step k.

        Returns the ``touched`` mask after each step (list of bool (H,W)).
        """
        t = torch.from_numpy(touched0.astype(np.float32))[None, None]
        b = torch.from_numpy(burning0.astype(np.float32))[None, None]
        s = static[None]
        out = []
        for k in range(n_steps):
            wind = torch.tensor([wind_fn(k)], dtype=torch.float32)
            t, b = self.model.rollout_step(t, b, s, wind)
            out.append(t[0, 0].numpy() > 0.5)
        return out

    @torch.no_grad()
    def ensemble(
        self,
        world: World,
        ignition: tuple[int, int],
        horizons_min: list[float],
        n_members: int,
        rng: np.random.Generator,
        ens_cfg: dict[str, Any] | None = None,
        t0_s: float = 0.0,
    ) -> EnsembleResult:
        """Batched Monte Carlo ensemble; perturbations mirror the physics one."""
        e = ens_cfg or self.cfg["firesim"]["ensemble"]
        horizons = sorted(float(h) for h in horizons_min)
        n = world.n
        B = n_members

        touched = torch.zeros((B, 1, n, n))
        for m in range(B):
            r = int(np.clip(round(ignition[0] + rng.normal(0, e["ignition_jitter_cells"])), 1, n - 2))
            c = int(np.clip(round(ignition[1] + rng.normal(0, e["ignition_jitter_cells"])), 1, n - 2))
            touched[m, 0, r : r + 2, c : c + 2] = 1.0
        burning = touched.clone()
        static = build_static_planes(world.elevation, world.fuel).expand(B, -1, -1, -1)

        dir_delta = rng.normal(0, e["wind_dir_jitter_deg"], size=B)
        speed_scale = np.clip(rng.normal(1.0, e["wind_speed_jitter_frac"], size=B), 0.3, 2.0)

        n_steps = int(round(horizons[-1] / SNAP_MIN))
        prob = np.zeros((len(horizons), n, n))
        for k in range(n_steps):
            t_s = t0_s + k * SNAP_MIN * 60.0
            speed, direction = world.wind.at(t_s)
            s = speed * speed_scale
            d = direction + np.deg2rad(dir_delta)
            wind = torch.tensor(np.stack([s * np.cos(d), s * np.sin(d)], axis=1),
                                dtype=torch.float32)
            touched, burning = self.model.rollout_step(touched, burning, static, wind)
            t_min = (k + 1) * SNAP_MIN
            for h, horizon in enumerate(horizons):
                if abs(t_min - horizon) < SNAP_MIN / 2:
                    prob[h] = touched[:, 0].mean(dim=0).numpy()
        # Horizons beyond what we stepped (shouldn't happen) keep zeros.
        return EnsembleResult(horizons_min=horizons, prob=prob, n_members=B)
