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
from emberline.surrogate.datasets import FirePairDataset, load_records, split_world_ids
from emberline.surrogate.model import IN_CHANNELS, MAX_PARAMS, FireUNet, build_static_planes


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


def test_shard_resume_skips_existing(tiny_cfg, shards):
    import os

    p = shard_path(tiny_cfg, 1)
    mtime = os.path.getmtime(p)
    generate_world_shard(tiny_cfg, 1)  # must be a no-op
    assert os.path.getmtime(p) == mtime
