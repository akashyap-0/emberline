"""Fractal terrain via spectral synthesis.

We synthesise elevation as approximate fractional Brownian motion: draw white
Gaussian noise, go to Fourier space, and shape the amplitude spectrum as
``|k|^(-beta/2)`` so the *power* spectrum falls off as ``|k|^-beta``. Real
terrain power spectra are empirically close to power laws with beta ~ 2-4
(rougher to smoother); this gives visually plausible ridges/valleys with a
single tunable roughness knob and is fully deterministic given the RNG.
"""

from __future__ import annotations

import numpy as np


def fractal_field(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-std fractal noise field of shape (n, n).

    Parameters
    ----------
    n : grid side length.
    beta : power-spectrum exponent. Higher = smoother (more energy at
        long wavelengths).
    rng : seeded generator; identical inputs reproduce the field exactly.
    """
    white = rng.standard_normal((n, n))
    spec = np.fft.fft2(white)
    kx = np.fft.fftfreq(n)
    ky = np.fft.fftfreq(n)
    k = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    k[0, 0] = 1.0  # avoid div-by-zero; DC amplitude zeroed below
    amp = k ** (-beta / 2.0)
    amp[0, 0] = 0.0
    field = np.real(np.fft.ifft2(spec * amp))
    field -= field.mean()
    field /= field.std() + 1e-12
    return field


def make_elevation(n: int, relief_m: float, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Elevation map (metres), min 0, peak-to-valley ~ ``relief_m``."""
    f = fractal_field(n, beta, rng)
    f = f - f.min()
    f = f / (f.max() + 1e-12)
    return (f * relief_m).astype(np.float64)


def slope_components(elevation: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Terrain gradient (dz/dx, dz/dy) in m/m via central differences.

    x = column direction (east), y = row direction (north). Used by the fire
    model's slope factor and by mesh line-of-sight occlusion.
    """
    dz_dy, dz_dx = np.gradient(elevation, cell_m)
    return dz_dx, dz_dy
