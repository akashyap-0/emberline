"""Torch Dataset over per-world shards, split BY WORLD (no frame leakage).

Splits
------
* ``holdout``: worlds whose base wind direction is in the held-out regime
  band - never trained or validated on; eval-only generalization set.
* remaining worlds: last ``val_world_frac`` of ids -> ``val``, rest ``train``.

Splitting by world (not frame) matters because consecutive frames of one
fire are nearly identical; splitting frames would leak the answer into
validation and inflate every metric we later put in front of judges.

Training samples are 64x64 crops centred near the fire front with random
jitter: the front is where the learning signal lives (empty wilderness and
fully-burned interior are trivially predictable), and small crops are what
keep CPU training steps fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset

from .data import is_holdout_regime, shard_path
from .model import ELEV_SCALE, WIND_SCALE


def rotate_wind(u: float, v: float, theta: float) -> tuple[float, float]:
    """Rotate the wind vector by ``theta`` radians in array (col=x, row=y) space.

    Must use the SAME rotation matrix as :func:`rotated_crop_coords` so the
    world and its wind stay physically consistent under augmentation.
    """
    c, s = np.cos(theta), np.sin(theta)
    return float(u * c - v * s), float(u * s + v * c)


def rotated_crop_coords(center_rc: tuple[float, float], crop: int,
                        theta: float) -> np.ndarray:
    """(2, crop, crop) source [row, col] coordinates for a crop rotated by theta.

    Output pixel at offset (dc', dr') from the crop centre samples the world at
    offset R(-theta)·(dc', dr') from ``center_rc`` — i.e. the crop shows the
    world rotated by +theta about the centre, matching :func:`rotate_wind`.
    """
    off = np.arange(crop, dtype=np.float64) - (crop - 1) / 2.0
    dr, dc = np.meshgrid(off, off, indexing="ij")
    c, s = np.cos(theta), np.sin(theta)
    src_c = center_rc[1] + c * dc + s * dr
    src_r = center_rc[0] - s * dc + c * dr
    return np.stack([src_r, src_c])


@dataclass
class FireRecord:
    """All snapshots of one simulated fire plus its world's static planes."""

    states: np.ndarray  # (T, H, W) uint8
    winds: np.ndarray  # (T, 2) float32
    static: np.ndarray  # (6, H, W) float32 (elev_norm + fuel one-hot)
    world_id: int


def load_records(cfg: dict[str, Any], world_ids: list[int]) -> list[FireRecord]:
    records: list[FireRecord] = []
    for wid in world_ids:
        p = shard_path(cfg, wid)
        if not p.exists():
            continue
        with np.load(p) as z:
            elev = z["elevation"]
            fuel = z["fuel"]
            static = _static_planes(elev, fuel)
            for i in range(int(z["n_fires"])):
                states = z[f"fire{i}_states"]
                if states.shape[0] < 2:
                    continue
                records.append(FireRecord(states, z[f"fire{i}_winds"], static, wid))
    return records


def _static_planes(elevation: np.ndarray, fuel: np.ndarray) -> np.ndarray:
    elev = ((elevation - elevation.mean()) / ELEV_SCALE).astype(np.float32)
    onehot = np.eye(5, dtype=np.float32)[fuel.astype(np.int64)]
    return np.concatenate([elev[None], np.moveaxis(onehot, -1, 0)], axis=0)


def split_world_ids(cfg: dict[str, Any]) -> dict[str, list[int]]:
    """{'train': [...], 'val': [...], 'holdout': [...]} world ids."""
    ds = cfg["surrogate"]["dataset"]
    train_ids, val_ids, hold_ids = [], [], []
    usable = []
    for wid in range(1, int(ds["n_worlds"]) + 1):
        p = shard_path(cfg, wid)
        if not p.exists():
            continue
        with np.load(p) as z:
            base_dir = float(z["wind_base_dir_deg"])
        (hold_ids if is_holdout_regime(cfg, base_dir) else usable).append(wid)
    n_val = max(1, int(len(usable) * float(cfg["surrogate"]["train"]["val_world_frac"])))
    val_ids = usable[-n_val:]
    train_ids = usable[:-n_val]
    return {"train": train_ids, "val": val_ids, "holdout": hold_ids}


class FirePairDataset(Dataset):
    """(t, t+10min) pairs as crop tensors: x (10,c,c), y (2,c,c)."""

    def __init__(self, records: list[FireRecord], crop: int, rng_seed: int,
                 full_frame: bool = False, oversample_early: int = 1,
                 rotate_augment: bool = False, rotate_keep_frac: float = 0.25) -> None:
        """``oversample_early``: repeat pairs with t <= 2 this many times.

        Detection-time forecasting always starts from a tiny fire, but only
        ~1 in 8 harvested pairs shows one; oversampling the early frames
        rebalances training toward the operationally critical regime.

        ``rotate_augment``: rotate world (terrain+fuel+fire) AND wind jointly
        by a uniform random angle per sample (Phase 9). This makes the wind-
        direction distribution seen in training isotropic, which is the fix
        for the held-out-wind-regime generalization gap. ``rotate_keep_frac``
        of samples are left unrotated so native (un-resampled) grid statistics
        stay represented — rollout evaluation always runs on unrotated grids.
        """
        self.records = records
        self.crop = crop
        self.full = full_frame
        self.rotate = rotate_augment
        self.keep_frac = float(rotate_keep_frac)
        self.rng = np.random.default_rng(rng_seed)
        self.index: list[tuple[int, int]] = []  # (record, t)
        for ri, rec in enumerate(records):
            for t in range(rec.states.shape[0] - 1):
                if (rec.states[t] > 0).any():
                    reps = oversample_early if t <= 2 else 1
                    self.index.extend([(ri, t)] * reps)

    def __len__(self) -> int:
        return len(self.index)

    def _crop_origin(self, touched: np.ndarray) -> tuple[int, int]:
        h, w = touched.shape
        rr, cc = np.where(touched)
        # Centre on a random front cell + jitter so the front position inside
        # the crop varies (the model must not learn "fire is always centred").
        k = self.rng.integers(len(rr))
        r = int(rr[k] + self.rng.integers(-10, 11))
        c = int(cc[k] + self.rng.integers(-10, 11))
        r0 = int(np.clip(r - self.crop // 2, 0, h - self.crop))
        c0 = int(np.clip(c - self.crop // 2, 0, w - self.crop))
        return r0, c0

    def _rotated_item(self, rec: FireRecord, t: int):
        """Crop sampled through a rotated grid; wind rotated by the same angle."""
        s_t, s_n = rec.states[t], rec.states[t + 1]
        h, w = s_t.shape
        theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
        # Centre on a jittered front cell, clamped so every rotated-crop corner
        # stays inside the world (half-diagonal margin) — no fill artefacts.
        rr, cc = np.where(s_t > 0)
        k = self.rng.integers(len(rr))
        margin = int(np.ceil(self.crop / 2 * np.sqrt(2))) + 1
        cr = float(np.clip(rr[k] + self.rng.integers(-10, 11), margin, h - 1 - margin))
        cc_ = float(np.clip(cc[k] + self.rng.integers(-10, 11), margin, w - 1 - margin))
        coords = rotated_crop_coords((cr, cc_), self.crop, theta)

        def nearest(plane: np.ndarray) -> np.ndarray:
            return map_coordinates(plane.astype(np.float32), coords, order=0,
                                   mode="nearest")

        elev = map_coordinates(rec.static[0], coords, order=1, mode="nearest")
        fuel_planes = [nearest(rec.static[j]) for j in range(1, rec.static.shape[0])]
        u, v = rotate_wind(*(rec.winds[t] / WIND_SCALE), theta)
        shape = (1, self.crop, self.crop)
        x = np.concatenate([
            nearest(s_t > 0)[None],
            nearest(s_t == 1)[None],
            elev[None].astype(np.float32),
            np.stack(fuel_planes),
            np.full(shape, u, dtype=np.float32),
            np.full(shape, v, dtype=np.float32),
        ])
        y = np.stack([nearest(s_n > 0), nearest(s_n == 1)])
        return torch.from_numpy(x), torch.from_numpy(y)

    def __getitem__(self, i: int):
        ri, t = self.index[i]
        rec = self.records[ri]
        if self.rotate and not self.full and self.rng.random() >= self.keep_frac:
            return self._rotated_item(rec, t)
        s_t, s_n = rec.states[t], rec.states[t + 1]
        touched_t = (s_t > 0).astype(np.float32)
        if self.full:
            sl = np.s_[:, :]
        else:
            r0, c0 = self._crop_origin(s_t > 0)
            sl = np.s_[r0 : r0 + self.crop, c0 : c0 + self.crop]
        u, v = rec.winds[t] / WIND_SCALE
        c = touched_t[sl].shape
        x = np.concatenate([
            touched_t[sl][None],
            (s_t[sl] == 1).astype(np.float32)[None],
            rec.static[:, sl[0], sl[1]],
            np.full((1, *c), u, dtype=np.float32),
            np.full((1, *c), v, dtype=np.float32),
        ])
        y = np.stack([(s_n[sl] > 0).astype(np.float32), (s_n[sl] == 1).astype(np.float32)])
        return torch.from_numpy(x), torch.from_numpy(y)
