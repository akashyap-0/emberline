"""Confounder event signatures + fire-plume event synthesis.

Each confounder has a distinct multichannel signature; the assumptions are
what make the classification problem honest rather than trivially separable:

=============  =========================================================
bbq            20-40 min; moderate PM with flicker; VOC/PM ratio ~10x a
               fire's (fatty pyrolysis); single node; no RH/temp change.
wood_stove     2-4 h evening burn; slow steady PM + moderate VOC. The
               hardest confounder: it IS wood smoke - separated mainly by
               its flat (non-growing) profile and single-node footprint.
vehicle        2-5 min exhaust pass/idle; sharp PM spike, big VOC spike,
               fast exponential decay; single node near roads.
fog            2-5 h; optical PM sensors misread droplets as particles ->
               apparent PM rise; RH pinned near 100, slight cooling; hits
               ALL nodes simultaneously (spatial footprint is a giveaway
               the mesh exploits, but a single node must rely on RH).
dust           10-30 min gust event; large ragged PM, no VOC, RH drops;
               one or two nodes.
aerosol        1-3 min point release (spray near a node); extreme brief
               PM + VOC; single node.
fire (plume)   driven by the physics sim through the Gaussian plume:
               PM grows as the fire grows, VOC rises with PM at a low
               ratio (smouldering fraction), other channels ~flat at
               ignition-relevant distances.
=============  =========================================================

An event is a callable envelope: t (s, relative) -> (4, len(t)) additive
contributions, plus its node scope. Windows are later labelled by EVENT so
train/val never share an event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

CONFOUNDERS = ["bbq", "wood_stove", "vehicle", "fog", "dust", "aerosol"]


@dataclass
class EventSpec:
    """One synthetic event: kind, duration, per-node envelope."""

    kind: str  # 'fire' or a confounder name
    duration_s: float
    # envelope(node_index, t_rel) -> (4, len(t)) additive signal
    envelope: Callable[[int, np.ndarray], np.ndarray]
    scope: list[int] = field(default_factory=list)  # node indices affected


def _smooth_noise(t: np.ndarray, rng: np.random.Generator, tau: float = 60.0) -> np.ndarray:
    """Positive low-pass flicker ~1.0 mean, for organic-looking envelopes."""
    n = len(t)
    x = rng.standard_normal(n // max(1, int(tau)) + 2)
    xp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)
    return np.clip(1.0 + 0.35 * xp, 0.1, None)


def make_confounder(kind: str, n_nodes: int, rng: np.random.Generator) -> EventSpec:
    """Sample one confounder event with randomized amplitude/duration."""
    if kind == "bbq":
        dur = float(rng.uniform(20, 40) * 60)
        amp_pm = float(rng.uniform(15, 45))
        amp_voc = float(rng.uniform(0.8, 2.0))
        node = int(rng.integers(n_nodes))

        def env(ni: int, t: np.ndarray) -> np.ndarray:
            out = np.zeros((4, len(t)))
            ramp = np.clip(t / 180, 0, 1) * np.clip((dur - t) / 300, 0, 1)
            flick = _smooth_noise(t, rng)
            out[0] = amp_pm * ramp * flick
            out[1] = amp_voc * ramp * flick
            return out

        return EventSpec(kind, dur, env, [node])

    if kind == "wood_stove":
        dur = float(rng.uniform(2, 4) * 3600)
        amp_pm = float(rng.uniform(12, 30))
        amp_voc = float(rng.uniform(0.25, 0.6))
        node = int(rng.integers(n_nodes))

        def env(ni: int, t: np.ndarray) -> np.ndarray:
            out = np.zeros((4, len(t)))
            ramp = np.clip(t / 900, 0, 1) * np.clip((dur - t) / 1800, 0, 1)
            flick = _smooth_noise(t, rng, tau=300)
            out[0] = amp_pm * ramp * flick
            out[1] = amp_voc * ramp * flick
            return out

        return EventSpec(kind, dur, env, [node])

    if kind == "vehicle":
        dur = float(rng.uniform(2, 5) * 60)
        amp_pm = float(rng.uniform(12, 40))
        amp_voc = float(rng.uniform(1.0, 3.0))
        node = int(rng.integers(n_nodes))

        def env(ni: int, t: np.ndarray) -> np.ndarray:
            out = np.zeros((4, len(t)))
            spike = np.exp(-np.clip(t - 30, 0, None) / (dur / 3)) * np.clip(t / 30, 0, 1)
            out[0] = amp_pm * spike
            out[1] = amp_voc * spike
            return out

        return EventSpec(kind, dur, env, [node])

    if kind == "fog":
        dur = float(rng.uniform(2, 5) * 3600)
        amp_pm = float(rng.uniform(15, 45))

        def env(ni: int, t: np.ndarray) -> np.ndarray:
            out = np.zeros((4, len(t)))
            ramp = np.clip(t / 1200, 0, 1) * np.clip((dur - t) / 2400, 0, 1)
            out[0] = amp_pm * ramp
            out[2] = -2.0 * ramp
            out[3] = 35.0 * ramp  # RH pinned high (clipped to 100 downstream)
            return out

        return EventSpec(kind, dur, env, list(range(n_nodes)))

    if kind == "dust":
        dur = float(rng.uniform(10, 30) * 60)
        amp_pm = float(rng.uniform(25, 70))
        nodes = list(rng.choice(n_nodes, size=int(rng.integers(1, 3)), replace=False))

        def env(ni: int, t: np.ndarray) -> np.ndarray:
            out = np.zeros((4, len(t)))
            ramp = np.clip(t / 120, 0, 1) * np.clip((dur - t) / 240, 0, 1)
            gusty = _smooth_noise(t, rng, tau=20) ** 2
            out[0] = amp_pm * ramp * gusty
            out[3] = -12.0 * ramp
            return out

        return EventSpec(kind, dur, env, [int(x) for x in nodes])

    if kind == "aerosol":
        dur = float(rng.uniform(1, 3) * 60)
        amp_pm = float(rng.uniform(40, 140))
        amp_voc = float(rng.uniform(2.0, 5.0))
        node = int(rng.integers(n_nodes))

        def env(ni: int, t: np.ndarray) -> np.ndarray:
            out = np.zeros((4, len(t)))
            spike = np.exp(-np.clip(t - 10, 0, None) / (dur / 2)) * np.clip(t / 10, 0, 1)
            out[0] = amp_pm * spike
            out[1] = amp_voc * spike
            return out

        return EventSpec(kind, dur, env, [node])

    raise ValueError(f"unknown confounder {kind}")


def fire_added_series(conc_1hz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(4, T) additive contribution from a fire plume PM2.5 series.

    VOC tracks PM at a low smouldering ratio; temp/RH untouched (nodes are
    hundreds of metres from the flame front at detection time).
    """
    out = np.zeros((4, len(conc_1hz)))
    out[0] = conc_1hz
    out[1] = conc_1hz * float(rng.uniform(0.015, 0.035))
    return out
