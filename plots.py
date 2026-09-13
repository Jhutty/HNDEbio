"""Optional plotting helpers (matplotlib is imported only when used)."""

from __future__ import annotations

import numpy as np


def plot_fit(problem, fit, *, show_truth: bool = False):
    """Plot observations and the fitted trajectory for every observed state."""
    import matplotlib.pyplot as plt

    prediction = fit.prediction.detach().cpu().numpy()
    fig, axes = plt.subplots(
        problem.n_observed,
        1,
        figsize=(8, max(3, 2.4 * problem.n_observed)),
        sharex=True,
        squeeze=False,
    )
    for column, (state_index, axis) in enumerate(zip(problem.observed, axes[:, 0])):
        name = problem.states[state_index]
        axis.scatter(problem.t, problem.data[:, column], s=12, color="0.55", label="data")
        axis.plot(problem.t, prediction[:, state_index], label=fit.kind)
        if show_truth and problem.truth is not None:
            axis.plot(problem.t, problem.truth[:, state_index], "k--", alpha=0.7, label="truth")
        if problem.train_size < len(problem.t):
            axis.axvline(problem.t[problem.train_size - 1], color="tab:red", linestyle=":")
        axis.set_ylabel(name)
        axis.legend()
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("time")
    fig.tight_layout()
    return fig


def plot_losses(fit, *, log_scale: bool = True):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    axis.plot(np.arange(1, len(fit.losses) + 1), fit.losses)
    if log_scale and fit.losses and min(fit.losses) > 0:
        axis.set_yscale("log")
    axis.set(xlabel="iteration", ylabel=fit.config.loss, title=f"{fit.problem.name}: {fit.kind}")
    axis.grid(alpha=0.2)
    return fig


def plot_profile(profile, *, metric: str = "train_rmse"):
    """Plot replicate points and their median loss profile."""
    import matplotlib.pyplot as plt

    names = profile.parameter_names
    fig, axes = plt.subplots(len(names), 1, figsize=(7, max(3, 3 * len(names))), squeeze=False)
    for name, axis in zip(names, axes[:, 0]):
        rows = profile.for_parameter(name)
        values = sorted({row["value"] for row in rows})
        medians = [np.median([row[metric] for row in rows if row["value"] == value]) for value in values]
        axis.scatter([row["value"] for row in rows], [row[metric] for row in rows], s=12, alpha=0.35)
        axis.plot(values, medians)
        axis.axvline(rows[0]["true_value"], color="tab:red", linestyle=":")
        axis.set(xscale="log", xlabel=name, ylabel=metric)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_profile_comparison(hybrid, mechanistic, *, metric: str = "train_rmse"):
    """Overlay HNDE and complete-mechanistic loss profiles as in the paper."""
    import matplotlib.pyplot as plt

    names = tuple(name for name in hybrid.parameter_names if name in mechanistic.parameter_names)
    if not names:
        raise ValueError("the profiles have no parameters in common")
    fig, axes = plt.subplots(len(names), 1, figsize=(7, max(3, 3 * len(names))), squeeze=False)
    for name, axis in zip(names, axes[:, 0]):
        for profile, label, style in (
            (hybrid, "unknown-parameter HNDE", "-"),
            (mechanistic, "complete mechanistic", "--"),
        ):
            rows = profile.for_parameter(name)
            values = sorted({row["value"] for row in rows})
            medians = [
                np.median([row[metric] for row in rows if row["value"] == value])
                for value in values
            ]
            axis.plot(values, medians, style, marker="o", label=label)
        true_value = hybrid.for_parameter(name)[0]["true_value"]
        axis.axvline(true_value, color="tab:red", linestyle=":", label="true value")
        axis.set(xscale="log", xlabel=name, ylabel=metric)
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig
