"""Reproducible benchmark problems from the accompanying paper."""

from __future__ import annotations

import numpy as np

from .core import Problem
from .simulation import multiplicative_noise, simulate


GLYCOLYSIS_PARAMETERS = {
    "J0": 2.5,
    "k1": 100.0,
    "k2": 6.0,
    "k3": 16.0,
    "k4": 100.0,
    "k5": 1.28,
    "k6": 12.0,
    "k": 1.8,
    "kappa": 13.0,
    "q": 4.0,
    "K1": 0.52,
    "psi": 0.1,
    "N": 1.0,
    "A": 4.0,
}

GLYCOLYSIS_LEARNABLE = (
    "J0",
    "k2",
    "k3",
    "k4",
    "k5",
    "k6",
    "k",
    "kappa",
    "psi",
    "N",
    "A",
)

GLYCOLYSIS_INITIAL_GUESS = {
    "J0": 1.0,
    "k2": 1.0,
    "k3": 10.0,
    "k4": 100.0,
    "k5": 1.0,
    "k6": 10.0,
    "k": 1.0,
    "kappa": 10.0,
    "psi": 0.1,
    "N": 1.0,
    "A": 1.0,
}


def _stack(values, like):
    if type(like).__module__.startswith("torch"):
        import torch

        return torch.stack(values, dim=-1)
    return np.stack(values, axis=-1)


def glycolysis_rhs(t, x, p):
    """Complete seven-state yeast glycolysis model from Ruoff et al."""
    S1, S2, S3, S4, S5, S6, S7 = (x[..., index] for index in range(7))
    flux = p["k1"] * S1 * S6 / (1 + (S6 / p["K1"]) ** p["q"])
    return _stack(
        (
            p["J0"] - flux,
            2 * flux - p["k2"] * S2 * (p["N"] - S5) - p["k6"] * S2 * S5,
            p["k2"] * S2 * (p["N"] - S5) - p["k3"] * S3 * (p["A"] - S6),
            p["k3"] * S3 * (p["A"] - S6) - p["k4"] * S4 * S5 - p["kappa"] * (S4 - S7),
            p["k2"] * S2 * (p["N"] - S5) - p["k4"] * S4 * S5 - p["k6"] * S2 * S5,
            -2 * flux + 2 * p["k3"] * S3 * (p["A"] - S6) - p["k5"] * S6,
            p["psi"] * p["kappa"] * (S4 - S7) - p["k"] * S7,
        ),
        x,
    )


def glycolysis_known_rhs(t, x, p):
    """Glycolysis dynamics with the unknown S1-to-S2 flux removed."""
    S1, S2, S3, S4, S5, S6, S7 = (x[..., index] for index in range(7))
    zero = 0 * S1
    return _stack(
        (
            p["J0"] + zero,
            -p["k2"] * S2 * (p["N"] - S5) - p["k6"] * S2 * S5,
            p["k2"] * S2 * (p["N"] - S5) - p["k3"] * S3 * (p["A"] - S6),
            p["k3"] * S3 * (p["A"] - S6) - p["k4"] * S4 * S5 - p["kappa"] * (S4 - S7),
            p["k2"] * S2 * (p["N"] - S5) - p["k4"] * S4 * S5 - p["k6"] * S2 * S5,
            2 * p["k3"] * S3 * (p["A"] - S6) - p["k5"] * S6,
            p["psi"] * p["kappa"] * (S4 - S7) - p["k"] * S7,
        ),
        x,
    )


def glycolysis(
    *,
    n: int = 225,
    t_end: float = 6.0,
    train_size: int = 150,
    noise: float = 0.05,
    seed: int = 42,
) -> Problem:
    """Paper's missing-mechanism glycolysis benchmark."""
    t = np.linspace(0.0, t_end, n)
    true_x0 = np.array([1.6, 1.5, 0.2, 0.35, 0.3, 2.67, 0.1])
    truth = simulate(glycolysis_rhs, true_x0, t, GLYCOLYSIS_PARAMETERS, max_step=0.002)
    data = multiplicative_noise(truth, noise, seed)
    return Problem(
        name="glycolysis",
        rhs=glycolysis_rhs,
        known_rhs=glycolysis_known_rhs,
        t=t,
        x0=data[0],
        data=data,
        states=("S1", "S2", "S3", "S4", "S5", "S6", "S7"),
        params=GLYCOLYSIS_PARAMETERS,
        learnable_params=GLYCOLYSIS_LEARNABLE,
        param_init=GLYCOLYSIS_INITIAL_GUESS,
        train_size=train_size,
        truth=truth,
        metadata={
            "uncertainty": "mechanism",
            "missing_mechanism": "k1*S1*S6 / (1 + (S6/K1)**q)",
            "paper_defaults": True,
            "mechanistic_x0": data[0],
            "solver_max_step": 0.01,
        },
    )


LOTKA_VOLTERRA_PARAMETERS = {
    "alpha": 0.5,
    "beta": 1.5,
    "gamma": 3.0,
    "delta": 1.0,
}


def lotka_volterra_rhs(t, x, p):
    """Complete Lotka--Volterra dynamics in conventional S1, S2, S3 order."""
    S1, S2, S3 = (x[..., index] for index in range(3))
    return _stack(
        (
            p["alpha"] * S1 - p["beta"] * S1 * S2,
            p["beta"] * S1 * S2 - p["gamma"] * S2 * S3,
            p["gamma"] * S2 * S3 - p["delta"] * S3,
        ),
        x,
    )


def lotka_volterra_model_rhs(t, x, p):
    """Complete dynamics in model order: observed S2, S3, then latent S1."""
    S2, S3, S1 = (x[..., index] for index in range(3))
    return _stack(
        (
            p["beta"] * S1 * S2 - p["gamma"] * S2 * S3,
            p["gamma"] * S2 * S3 - p["delta"] * S3,
            p["alpha"] * S1 - p["beta"] * S1 * S2,
        ),
        x,
    )


def lotka_volterra_known_rhs(t, x, p):
    """Known measured-state dynamics; the latent-state equation is learned."""
    S2, S3, S1 = (x[..., index] for index in range(3))
    return _stack(
        (
            p["beta"] * S1 * S2 - p["gamma"] * S2 * S3,
            p["gamma"] * S2 * S3 - p["delta"] * S3,
            0 * S1,
        ),
        x,
    )


def lotka_volterra(
    *,
    n: int = 150,
    t_end: float = 15.0,
    train_size: int = 90,
    noise: float = 0.05,
    seed: int = 42,
) -> Problem:
    """Paper's three-species benchmark with unmeasured S1."""
    t = np.linspace(0.0, t_end, n)
    conventional_truth = simulate(
        lotka_volterra_rhs,
        np.ones(3),
        t,
        LOTKA_VOLTERRA_PARAMETERS,
        max_step=0.01,
    )
    model_truth = conventional_truth[:, [1, 2, 0]]
    noisy = multiplicative_noise(conventional_truth, noise, seed)
    observed_data = noisy[:, [1, 2]]
    return Problem(
        name="lotka_volterra",
        rhs=lotka_volterra_model_rhs,
        known_rhs=lotka_volterra_known_rhs,
        t=t,
        x0=np.array([observed_data[0, 0], observed_data[0, 1], 0.0]),
        data=observed_data,
        states=("S2", "S3", "S1_latent"),
        observed=(0, 1),
        correction=(2,),
        params=LOTKA_VOLTERRA_PARAMETERS,
        learnable_params=("beta", "gamma", "delta"),
        param_init={"beta": 1.0, "gamma": 2.0, "delta": 1.0},
        train_size=train_size,
        truth=model_truth,
        metadata={
            "uncertainty": "state",
            "unobserved_state": "S1",
            "paper_defaults": True,
            "mechanistic_x0": model_truth[0],
            "solver_max_step": 0.02,
        },
    )


def lv3(**kwargs) -> Problem:
    """Compatibility alias for :func:`lotka_volterra`."""
    return lotka_volterra(**kwargs)


# Compatibility alias retained for code that imported examples.rhs.
rhs = lotka_volterra_rhs
