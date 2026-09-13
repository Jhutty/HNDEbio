"""Small differentiable ODE solvers with no dependency beyond PyTorch."""

from __future__ import annotations

import torch


def rk4_step(model, t: torch.Tensor, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """One classical fourth-order Runge--Kutta step."""
    state_dt = dt
    while state_dt.ndim < x.ndim:
        state_dt = state_dt.unsqueeze(-1)
    half = state_dt / 2
    k1 = model(t, x)
    k2 = model(t + dt / 2, x + half * k1)
    k3 = model(t + dt / 2, x + half * k2)
    k4 = model(t + dt, x + state_dt * k3)
    return x + state_dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def integrate(
    model,
    x0: torch.Tensor,
    t: torch.Tensor,
    method: str = "rk4",
    max_step: float | None = None,
) -> torch.Tensor:
    """Integrate at supplied observation times.

    The leading dimensions of ``x0`` are treated as independent batches and the
    returned shape is ``(len(t), *x0.shape)``.
    """
    if t.ndim not in {1, 2}:
        raise ValueError("t must be (steps,) or (steps, batch)")
    if method not in {"rk4", "euler"}:
        raise ValueError("method must be 'rk4' or 'euler'")
    states = [x0]
    for index in range(len(t) - 1):
        interval = t[index + 1] - t[index]
        substeps = 1
        if max_step is not None:
            substeps = max(1, int(torch.ceil(interval.detach().abs().max() / max_step)))
        dt = interval / substeps
        next_state = states[-1]
        current_t = t[index]
        for _ in range(substeps):
            if method == "rk4":
                next_state = rk4_step(model, current_t, next_state, dt)
            else:
                state_dt = dt
                while state_dt.ndim < next_state.ndim:
                    state_dt = state_dt.unsqueeze(-1)
                next_state = next_state + state_dt * model(current_t, next_state)
            current_t = current_t + dt
        states.append(next_state)
    return torch.stack(states)


def euler_sim(model, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compatibility wrapper for the prototype's Euler integrator."""
    return integrate(model, x0, t, method="euler")
