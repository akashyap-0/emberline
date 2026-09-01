"""Phase 3 tests: shapes, param cap, world-split hygiene, rollout invariants.

These run on a tiny throwaway dataset (2 worlds, 1 fire each) so the suite
stays fast and never depends on the full training artifacts.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from emberline.config import load_config
from emberline.surrogate.data import generate_world_shard, shard_path
from emberline.surrogate.datasets import (FirePairDataset, FireRecord, load_records,
                                          rotate_wind, rotated_crop_coords,
                                          split_world_ids)
from emberline.surrogate.model import (IN_CHANNELS, MAX_PARAMS, WIND_SCALE, FireUNet,
                                       build_static_planes)


@pytest.fixture(scope="module")
def tiny_cfg(tmp_path_factory):
    cfg = copy.deepcopy(load_config())
    cfg["world"]["grid_size"] = 96
    ds = cfg["surrogate"]["dataset"]
    ds["n_worlds"] = 2
    ds["fires_per_world"] = 1
    ds["snapshots_per_fire"] = 4
    ds["dir"] = str(tmp_path_factory.mktemp("shards"))
    return cfg


@pytest.fixture(scope="module")
def shards(tiny_cfg):
    # dataset dir is absolute in tiny_cfg, so shard_path must handle it
    return [generate_world_shard(tiny_cfg, i) for i in (1, 2)]


def test_model_param_cap_and_shapes():
    model = FireUNet(24)
    n = sum(p.numel() for p in model.parameters())
    assert n <= MAX_PARAMS
    x = torch.randn(2, IN_CHANNELS, 64, 64)
    y = model(x)
    assert y.shape == (2, 2, 64, 64)
    # Fully convolutional: same weights run at full grid size.
    assert model(torch.randn(1, IN_CHANNELS, 128, 128)).shape == (1, 2, 128, 128)


def test_shard_roundtrip_and_dataset(tiny_cfg, shards):
    recs = load_records(tiny_cfg, [1, 2])
    assert len(recs) >= 1
    ds = FirePairDataset(recs, crop=32, rng_seed=0)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (IN_CHANNELS, 32, 32) and y.shape == (2, 32, 32)
    assert x.dtype == torch.float32
    # touched plane is binary and consistent with burning plane
    assert set(np.unique(x[0].numpy())) <= {0.0, 1.0}
    assert (x[1] <= x[0] + 1e-6).all(), "burning implies touched"


def test_split_by_world_no_overlap(tiny_cfg, shards):
    splits = split_world_ids(tiny_cfg)
    all_ids = splits["train"] + splits["val"] + splits["holdout"]
    assert len(all_ids) == len(set(all_ids)), "a world appears in two splits"
    assert set(all_ids) == {1, 2}


def test_rollout_monotone_touched():
    """The rollout clamp guarantees burned cells never unburn, even untrained."""
    model = FireUNet(8)
    n = 64
    touched = torch.zeros(1, 1, n, n)
    touched[0, 0, 30:34, 30:34] = 1
    burning = touched.clone()
    elev = np.zeros((n, n))
    fuel = np.ones((n, n), dtype=np.int8)
    static = build_static_planes(elev, fuel)[None]
    prev = touched
    for _ in range(3):
        nxt, burning = model.rollout_step(prev, burning, static, torch.tensor([[5.0, 0.0]]))
        assert (nxt >= prev).all(), "touched mask must be monotone"
        assert (burning <= nxt).all()
        prev = nxt


def _bar_record(n: int = 96) -> FireRecord:
    """Synthetic fire: a downwind bar with wind blowing toward +col (east)."""
    s0 = np.zeros((n, n), dtype=np.uint8)
    s0[47:49, 47:49] = 1
    s1 = s0.copy()
    s1[47:49, 47:60] = 1  # spread strictly downwind (+col)
    fuel = np.ones((n, n), dtype=np.int8)
    static = build_static_planes(np.zeros((n, n)), fuel).numpy()
    winds = np.array([[10.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    return FireRecord(np.stack([s0, s1]), winds, static, world_id=0)


def test_rotation_world_and_wind_stay_consistent():
    """Phase 9 core invariant: rotating world and wind by the same angle keeps
    spread aligned with wind. Exact check at 90 deg (no interpolation error)."""
    rec = _bar_record()
    theta = np.pi / 2
    coords = rotated_crop_coords((48.0, 48.0), 32, theta)
    from scipy.ndimage import map_coordinates

    r0 = map_coordinates((rec.states[0] > 0).astype(np.float32), coords, order=0)
    r1 = map_coordinates((rec.states[1] > 0).astype(np.float32), coords, order=0)
    growth = (r1 > 0) & ~(r0 > 0)
    assert growth.any()
    rr, cc = np.where(growth)
    disp = np.array([cc.mean() - 15.5, rr.mean() - 15.5])  # (dcol, drow) from centre
    u2, v2 = rotate_wind(10.0, 0.0, theta)
    wind = np.array([u2, v2])
    cos_sim = disp @ wind / (np.linalg.norm(disp) * np.linalg.norm(wind))
    assert cos_sim > 0.95, f"rotated spread not aligned with rotated wind ({cos_sim:.2f})"
    assert abs(np.hypot(u2, v2) - 10.0) < 1e-6, "rotation must preserve wind speed"


def test_rotated_dataset_items_valid():
    rec = _bar_record()
    ds = FirePairDataset([rec], crop=32, rng_seed=7, rotate_augment=True,
                         rotate_keep_frac=0.0)
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (IN_CHANNELS, 32, 32) and y.shape == (2, 32, 32)
        assert set(np.unique(x[0].numpy())) <= {0.0, 1.0}, "touched must stay binary"
        assert (x[1] <= x[0] + 1e-6).all(), "burning implies touched"
        assert (y[0] >= x[0] - 1e-6).all(), "rotation must preserve monotone growth"
        onehot_sum = x[3:8].sum(dim=0).numpy()
        assert np.allclose(onehot_sum, 1.0), "fuel planes must stay one-hot"
        for p in (8, 9):
            assert float(x[p].max() - x[p].min()) < 1e-6, "wind planes are constant"
        speed = float(np.hypot(float(x[8, 0, 0]), float(x[9, 0, 0]))) * WIND_SCALE
        assert abs(speed - 10.0) < 1e-4, "wind speed preserved under rotation"


def test_shard_resume_skips_existing(tiny_cfg, shards):
    import os

    p = shard_path(tiny_cfg, 1)
    mtime = os.path.getmtime(p)
    generate_world_shard(tiny_cfg, 1)  # must be a no-op
    assert os.path.getmtime(p) == mtime
