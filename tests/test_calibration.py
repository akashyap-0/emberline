"""Phase 10 tests: temperature-scaling math (no model or dataset needed)."""

from __future__ import annotations

import numpy as np

from emberline.surrogate.calibration import (apply_temperature, fit_temperature,
                                             reliability)


def _overconfident_sample(n: int = 20000, t_true: float = 2.5, members: int = 20):
    """Synthetic ensemble output that is over-confident by construction:
    the true P(y=1) is sigmoid(z/t_true) but the ensemble reports sigmoid(z)."""
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 2.0, n)
    p_reported = 1 / (1 + np.exp(-z))
    # Quantise to member frequencies like a real M-member ensemble.
    p_reported = np.round(p_reported * members) / members
    y = (rng.random(n) < 1 / (1 + np.exp(-z / t_true))).astype(float)
    return p_reported, y


def test_fit_recovers_softening_temperature():
    p, y = _overconfident_sample()
    t = fit_temperature(p, y, members=20)
    assert 1.5 < t < 4.0, f"expected T near 2.5 for over-confident data, got {t:.2f}"


def test_temperature_improves_ece_and_preserves_order():
    p, y = _overconfident_sample()
    t = fit_temperature(p, y, members=20)
    p_cal = apply_temperature(p, t, members=20)
    *_, ece_raw = reliability(p, y)
    *_, ece_cal = reliability(p_cal, y)
    assert ece_cal < ece_raw, "temperature scaling must reduce ECE on its own regime"
    # Monotone: ranking of cells by risk is unchanged.
    order = np.argsort(p[:200])
    assert (np.diff(p_cal[:200][order]) >= -1e-12).all()


def test_identity_temperature_is_noop_inside_clip_range():
    p = np.linspace(0.05, 0.95, 19)
    out = apply_temperature(p, 1.0, members=20)
    assert np.allclose(out, p, atol=1e-9)


def test_saturated_frequencies_pass_through_unchanged():
    p = np.array([0.0, 0.05, 0.5, 0.95, 1.0])
    out = apply_temperature(p, 3.0, members=20)
    assert out[0] == 0.0 and out[-1] == 1.0, "unanimity must not be softened"
    assert 0.05 < out[1] < 0.5 < out[3] < 0.95, "interior must soften toward 0.5"
