"""Synthetic-data generation utilities."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


def simulate(
    rhs,
    x0,
    t,
    params: Mapping[str, float],
    *,
    max_step: float | None = None,
) -> np.ndarray:
    """Solve an ODE using fixed-step RK4 and return values at ``t`.

    Internal substeps make this suitable for generating the paper's stiff-ish
    glycolysis benchmark without introducing SciPy as a runtime requirement.
    """
    times = np.asarray(t, dtype=float)
    state = np.asarray(x0, dtype=float)
    values = [state.copy()]
    for start, stop in zip(times[:-1], times[1:]):
        interval = stop - start
        steps = 1 if max_step is None else max(1, math.ceil(interval / max_step))
        dt = interval / steps
        current_t = start
        for _ in range(steps):
            k1 = np.asarray(rhs(current_t, state, params))
            k2 = np.asarray(rhs(current_t + dt / 2, state + dt * k1 / 2, params))
            k3 = np.asarray(rhs(current_t + dt / 2, state + dt * k2 / 2, params))
            k4 = np.asarray(rhs(current_t + dt, state + dt * k3, params))
            state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            current_t += dt
        values.append(state.copy())
    return np.stack(values)


def multiplicative_noise(values, standard_deviation: float = 0.05, seed: int = 42):
    """Multiply observations by Gaussian noise with mean one."""
    if standard_deviation < 0:
        raise ValueError("standard_deviation must be non-negative")
    generator = np.random.default_rng(seed)
    return np.asarray(values) * generator.normal(1.0, standard_deviation, np.shape(values))
