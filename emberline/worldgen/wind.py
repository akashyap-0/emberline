"""Time-varying wind: mean regime + Ornstein-Uhlenbeck gusts + shifts.

Wind is modelled as a spatially uniform vector (fine at 2.5 km scale for a
surface-spread model; real fire-atmosphere coupling is out of scope) whose
speed and direction each follow

    x_{k+1} = x_k + (0 - x_k) * dt/tau + sigma * sqrt(2 dt / tau) * N(0,1)

i.e. a zero-mean OU *deviation* path with stationary std ``sigma`` and
relaxation time ``tau``, added to a piecewise-constant mean. Gusts are thus
temporally correlated (unlike white noise) which is what makes fire runs and
lulls look right. The mean is a step function so that

* configured ``regime_shifts`` (e.g. a frontal passage), and
* runtime shifts (the demo's ``--wind-shift``)

are both just edits to the mean after time t - the gust path is untouched,
keeping the run deterministic under a given seed.

Convention: ``dir`` is the direction the wind blows TOWARD, radians CCW from
+x (east); u = speed*cos(dir), v = speed*sin(dir) with y = north (row index
increases northward when plotted with origin='lower').
"""

from __future__ import annotations

import numpy as np


class WindModel:
    """Deterministic seeded wind time series with runtime-editable mean."""

    def __init__(
        self,
        base_speed_ms: float,
        base_dir_deg: float,
        gust_sigma: float,
        dir_sigma_deg: float,
        ou_tau_s: float,
        rng: np.random.Generator,
        regime_shifts: list[dict] | None = None,
        horizon_s: float = 8 * 3600.0,
        dt_s: float = 5.0,
    ) -> None:
        self.dt_s = dt_s
        n = int(horizon_s / dt_s) + 2
        self._mean_speed = np.full(n, float(base_speed_ms))
        self._mean_dir = np.full(n, np.deg2rad(float(base_dir_deg)))
        for shift in regime_shifts or []:
            k = int(float(shift["t_min"]) * 60.0 / dt_s)
            if "speed_ms" in shift:
                self._mean_speed[k:] = float(shift["speed_ms"])
            if "dir_deg" in shift:
                self._mean_dir[k:] = np.deg2rad(float(shift["dir_deg"]))

        def ou_path(sigma: float) -> np.ndarray:
            x = np.zeros(n)
            a = dt_s / ou_tau_s
            noise = rng.standard_normal(n) * sigma * np.sqrt(2 * a)
            for k in range(1, n):
                x[k] = x[k - 1] * (1 - a) + noise[k]
            return x

        self._dev_speed = ou_path(gust_sigma)
        self._dev_dir = ou_path(np.deg2rad(dir_sigma_deg))

    def _idx(self, t_s: float) -> int:
        return min(int(t_s / self.dt_s), len(self._mean_speed) - 1)

    def at(self, t_s: float) -> tuple[float, float]:
        """(speed m/s, direction rad) at time t. Speed clipped >= 0."""
        k = self._idx(t_s)
        speed = max(0.0, self._mean_speed[k] + self._dev_speed[k])
        direction = self._mean_dir[k] + self._dev_dir[k]
        return float(speed), float(direction)

    def uv(self, t_s: float) -> tuple[float, float]:
        """Cartesian wind vector (u east, v north) in m/s at time t."""
        s, d = self.at(t_s)
        return s * np.cos(d), s * np.sin(d)

    def field_at(self, t_s: float, shape: tuple[int, int]) -> np.ndarray:
        """(2, H, W) uniform u/v fields - the surrogate's wind input planes."""
        u, v = self.uv(t_s)
        out = np.empty((2, *shape))
        out[0], out[1] = u, v
        return out

    def apply_shift(self, t_s: float, dir_delta_deg: float = 0.0,
                    speed_delta_ms: float = 0.0) -> None:
        """Shift the mean wind from time t onward (demo ``--wind-shift``)."""
        k = self._idx(t_s)
        self._mean_dir[k:] += np.deg2rad(dir_delta_deg)
        self._mean_speed[k:] += speed_delta_ms
