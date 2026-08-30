"""Phase 4 tests: plume physics, signatures, event-split hygiene, model io."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emberline.config import load_config, rng_for
from emberline.detect.model import SmokeCNN, gbm_features, normalize
from emberline.detect.train import event_split
from emberline.sensors import baseline, place_nodes, sample_series
from emberline.sensors.events import CONFOUNDERS, make_confounder
from emberline.sensors.plume import plume_concentration
from emberline.worldgen import generate_world


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_plume_downwind_only():
    src = np.array([[100, 100]])
    sensors = np.array([[100, 120], [100, 80], [120, 100]])  # E, W, N of source
    c = plume_concentration(src, sensors, (5.0, 0.0), 10.0, 60.0, 0.1, 0.06)
    assert c[0] > 0, "downwind sensor must receive smoke"
    assert c[1] == 0, "upwind sensor must receive nothing"
    assert c[2] < c[0] * 0.05, "crosswind sensor sees ~nothing at 200 m"


def test_plume_decays_with_distance():
    src = np.array([[100, 100]])
    sensors = np.array([[100, 110], [100, 140], [100, 190]])
    c = plume_concentration(src, sensors, (5.0, 0.0), 10.0, 60.0, 0.1, 0.06)
    assert c[0] > c[1] > c[2] > 0


def test_baseline_diurnal_shapes():
    t = np.arange(0, 86400, 60).astype(float)
    b = baseline(t)
    temp, rh = b[2], b[3]
    assert 55000 < t[np.argmax(temp)] < 62000, "temp peaks mid-afternoon"
    assert np.corrcoef(temp, rh)[0, 1] < -0.9, "RH anti-correlates with temp"
    assert (b[0] > 0).all()


def test_confounder_signatures_distinct(cfg):
    rng = rng_for(cfg, "t-conf")
    n_nodes = 8
    for kind in CONFOUNDERS:
        spec = make_confounder(kind, n_nodes, rng)
        t = np.arange(min(int(spec.duration_s), 1800)).astype(float)
        env = spec.envelope(spec.scope[0], t)
        assert env.shape == (4, len(t))
        assert env[0].max() > 5, f"{kind} must move PM"
    fog = make_confounder("fog", n_nodes, rng)
    assert len(fog.scope) == n_nodes, "fog blankets every node"
    t = np.arange(int(fog.duration_s // 2)).astype(float)
    assert fog.envelope(0, t)[3].max() > 20, "fog pins RH high"
    dust = make_confounder("dust", n_nodes, rng)
    t = np.arange(int(dust.duration_s)).astype(float)
    assert dust.envelope(dust.scope[0], t)[1].max() == 0, "dust has no VOC"


def test_node_placement(cfg):
    world = generate_world(cfg, 0)
    nodes = place_nodes(world, 12, rng_for(cfg, "t-nodes"))
    assert len(nodes) == 12
    assert len({n.node_id for n in nodes}) == 12
    r0, c0, r1, c1 = world.town.district
    cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
    dists = [np.hypot(n.r - cr, n.c - cc) for n in nodes]
    assert min(dists) > (r1 - r0) / 2 - 2, "nodes sit outside/at the district edge"


def test_event_split_never_splits_an_event(cfg):
    rng = rng_for(cfg, "t-split")
    event_id = np.repeat(np.arange(50), 8)
    val = event_split(event_id, 0.25, rng)
    for e in range(50):
        vals = val[event_id == e]
        assert vals.all() or (~vals).all(), "an event straddles train/val"
    assert 0.05 < val.mean() < 0.6


def test_cnn_and_features_shapes():
    X = np.abs(np.random.default_rng(0).normal(10, 3, (5, 4, 60))).astype(np.float32)
    model = SmokeCNN()
    out = model(torch.from_numpy(normalize(X)))
    assert out.shape == (5,)
    f = gbm_features(X)
    assert f.shape[0] == 5 and np.isfinite(f).all()
    p = model.confidence(X[0])
    assert 0.0 <= p <= 1.0


def test_sample_series_noise_and_drift(cfg):
    world = generate_world(cfg, 0)
    node = place_nodes(world, 4, rng_for(cfg, "t-nodes2"))[0]
    t = np.arange(0, 3600).astype(float)
    s = sample_series(node, t, np.zeros((4, len(t))), cfg["sensors"]["noise"],
                      rng_for(cfg, "t-noise"))
    assert s.shape == (4, 3600)
    assert s[0].std() > 0.5, "PM noise present"
    assert (s[3] <= 100).all() and (s[0] >= 0).all()
