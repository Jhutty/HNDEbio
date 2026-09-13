"""Core data structures used throughout :mod:`udebio`.

The package deliberately keeps a biological model separate from the optimisation
machinery. A :class:`Problem` contains only the scientifically meaningful inputs:
dynamics, observations, initial state, and the known part of a hybrid model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import json
import random

import numpy as np


RHS = Callable[[Any, Any, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for one UDE calibration.

    ``batch_time`` is the number of consecutive observation times integrated
    from every sampled initial state. ``batch_size`` is the number of such
    trajectory fragments per optimiser update.
    """

    lr: float = 1e-2
    batch_size: int = 128
    batch_time: int = 32
    iterations: int = 2_000
    width: int = 15
    hidden_layers: int = 1
    lr_step: int = 1_000
    lr_gamma: float = 0.1
    loss: str = "mae"
    l1: float = 0.0
    seed: int = 42
    device: str = "cpu"
    max_attempts: int = 3
    grad_clip: float | None = None
    solver_max_step: float | None = None

    def __post_init__(self) -> None:
        if self.lr <= 0 or self.batch_size <= 0 or self.batch_time < 2:
            raise ValueError("lr and batch_size must be positive; batch_time must be >= 2")
        if self.iterations <= 0 or self.width <= 0 or self.hidden_layers <= 0:
            raise ValueError("iterations, width, and hidden_layers must be positive")
        if self.solver_max_step is not None and self.solver_max_step <= 0:
            raise ValueError("solver_max_step must be positive when supplied")
        if self.loss not in {"mae", "mse", "rmse"}:
            raise ValueError("loss must be one of: mae, mse, rmse")


# Backwards-compatible name used by the initial prototype.
TrainCfg = TrainConfig


@dataclass
class Problem:
    """A fully specified UDE analysis problem.

    ``rhs(t, x, params)`` is the complete mechanistic dynamics. ``known_rhs``
    is the part retained in a hybrid model. ``data`` has shape
    ``(n_times, n_observed)``; ``observed`` maps those columns to model-state
    indices. ``correction`` identifies states receiving a neural correction,
    and ``learnable_params`` identifies parameters fitted with the network.
    Both right-hand sides must support NumPy arrays and torch tensors.
    """

    name: str
    rhs: RHS
    t: Any
    x0: Any
    data: Any
    states: Sequence[str]
    params: Mapping[str, float] = field(default_factory=dict)
    known_rhs: RHS | None = None
    observed: Sequence[int] | None = None
    correction: Sequence[int] | None = None
    learnable_params: Sequence[str] = field(default_factory=tuple)
    param_init: Mapping[str, float] = field(default_factory=dict)
    train_size: int | None = None
    truth: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.x0 = np.asarray(self.x0, dtype=float)
        self.data = np.asarray(self.data, dtype=float)
        self.states = tuple(self.states)
        self.params = dict(self.params)
        self.param_init = dict(self.param_init)
        self.learnable_params = tuple(self.learnable_params)
        self.observed = tuple(range(len(self.states))) if self.observed is None else tuple(self.observed)
        self.correction = tuple(range(len(self.states))) if self.correction is None else tuple(self.correction)
        self.train_size = len(self.t) if self.train_size is None else int(self.train_size)
        if self.truth is not None:
            self.truth = np.asarray(self.truth, dtype=float)
        self.validate()

    def validate(self) -> None:
        if self.t.ndim != 1 or len(self.t) < 2 or np.any(np.diff(self.t) <= 0):
            raise ValueError("t must be a strictly increasing one-dimensional array")
        if self.x0.shape != (len(self.states),):
            raise ValueError("x0 must contain one value for each model state")
        if self.data.ndim != 2 or self.data.shape[0] != len(self.t):
            raise ValueError("data must have shape (len(t), n_observed)")
        if self.data.shape[1] != len(self.observed):
            raise ValueError("data columns must match the number of observed state indices")
        valid = set(range(len(self.states)))
        if not set(self.observed) <= valid or not set(self.correction) <= valid:
            raise ValueError("observed and correction indices must refer to model states")
        if len(set(self.observed)) != len(self.observed):
            raise ValueError("observed indices must be unique")
        if not 2 <= self.train_size <= len(self.t):
            raise ValueError("train_size must be between 2 and len(t)")
        missing = set(self.learnable_params) - set(self.params)
        if missing:
            raise ValueError(f"learnable parameters absent from params: {sorted(missing)}")

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_obs(self) -> int:
        return len(self.t)

    @property
    def n_observed(self) -> int:
        return len(self.observed)

    @property
    def known_dynamics(self) -> RHS:
        return self.known_rhs or self.rhs

    def initial_parameters(self) -> dict[str, float]:
        values = dict(self.params)
        values.update(self.param_init)
        return values


@dataclass
class FitResult:
    """Model, predictions, parameters, and metrics from one calibration."""

    problem: Problem
    config: TrainConfig
    kind: str
    model: Any
    prediction: Any
    losses: list[float]
    metrics: dict[str, float]
    parameters: dict[str, float] = field(default_factory=dict)
    attempt: int = 1

    @property
    def prob(self) -> Problem:
        return self.problem

    @property
    def cfg(self) -> TrainConfig:
        return self.config

    @property
    def artifacts(self) -> dict[str, Any]:
        return {"model": self.model, "pred": self.prediction, "losses": self.losses}

    def save(self, path: str | Path) -> Path:
        from .io import save_fit

        return save_fit(path, self)


@dataclass
class SweepResult:
    """Rows from a hyperparameter experiment and its selected configuration."""

    rows: list[dict[str, Any]]
    best_config: TrainConfig
    objective: str = "train_rmse"
    best_by_width: dict[int, TrainConfig] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Profile-likelihood (loss-profile) samples for mechanistic parameters."""

    rows: list[dict[str, Any]]
    parameter_names: tuple[str, ...]
    kind: str

    def for_parameter(self, name: str) -> list[dict[str, Any]]:
        return [row for row in self.rows if row["parameter"] == name]


@dataclass
class AnalysisResult:
    """Output of the high-level :func:`udebio.analyze` workflow."""

    problem: Problem
    fits: dict[str, list[FitResult]]
    sweeps: dict[str, SweepResult] = field(default_factory=dict)
    profiles: dict[str, ProfileResult] = field(default_factory=dict)
    insights: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None


@dataclass
class RunStore:
    root: str | Path = "results"
    name: str = "run"

    @property
    def path(self) -> Path:
        return Path(self.root) / self.name

    def mkdir(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def save_json(self, value: Any, filename: str) -> Path:
        path = self.mkdir() / filename
        write_json(value, path)
        return path

    def save_array(self, value: Any, filename: str) -> Path:
        path = self.mkdir() / filename
        np.save(path, value)
        return path

    def save_fig(self, fig: Any, filename: str) -> Path:
        path = self.mkdir() / filename
        fig.savefig(path, bbox_inches="tight")
        return path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value)


def write_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def run_dir(root: str | Path = "results", name: str = "run") -> Path:
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_idx(n: int, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(n_train), np.arange(n_train, n)


def rmse(y: Any, yhat: Any) -> Any:
    """Backend-neutral root mean square error (NumPy or torch)."""
    if type(y).__module__.startswith("torch"):
        import torch

        return torch.sqrt(torch.mean((y - yhat) ** 2))
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))
