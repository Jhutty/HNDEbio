"""Neural and hybrid differential-equation model construction."""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn

from .core import Problem, TrainConfig


MODEL_KINDS = ("nde", "known_hybrid", "unknown_hybrid", "mechanistic")


class MLP(nn.Module):
    """Single- or multi-hidden-layer tanh network used by the paper."""

    def __init__(self, n_in: int, n_out: int, width: int = 15, hidden_layers: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        previous = n_in
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        layers.append(nn.Linear(previous, n_out))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParameterMap(Mapping[str, torch.Tensor]):
    """Read-only merged view of fixed values and positive fitted parameters."""

    def __init__(
        self,
        fixed: Mapping[str, torch.Tensor],
        log_parameters: nn.ParameterDict,
    ):
        self.fixed = fixed
        self.log_parameters = log_parameters

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self.log_parameters:
            return self.log_parameters[key].exp()
        return self.fixed[key]

    def __iter__(self):
        return iter(set(self.fixed) | set(self.log_parameters))

    def __len__(self) -> int:
        return len(set(self.fixed) | set(self.log_parameters))


class DifferentialEquation(nn.Module):
    """Base module exposing mechanistic parameter estimates uniformly."""

    def mechanistic_parameters(self) -> dict[str, float]:
        return {}


class NeuralODE(DifferentialEquation):
    def __init__(self, problem: Problem, config: TrainConfig):
        super().__init__()
        self.net = MLP(
            problem.n_states,
            problem.n_states,
            config.width,
            config.hidden_layers,
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Compatibility alias from the early prototype.
NDE = NeuralODE


class HybridODE(DifferentialEquation):
    def __init__(
        self,
        problem: Problem,
        config: TrainConfig,
        *,
        learn_parameters: bool,
        fixed_params: Mapping[str, float] | None = None,
    ):
        super().__init__()
        self.problem = problem
        self.net = MLP(
            problem.n_states,
            len(problem.correction),
            config.width,
            config.hidden_layers,
        )
        initial = problem.initial_parameters()
        overrides = dict(fixed_params or {})
        learnable = set(problem.learnable_params) if learn_parameters else set()
        learnable -= set(overrides)

        self.log_mechanistic = nn.ParameterDict()
        fixed: dict[str, torch.Tensor] = {}
        for name, value in initial.items():
            value = overrides.get(name, value)
            if name in learnable:
                if value <= 0:
                    raise ValueError(
                        f"initial value for positive mechanistic parameter {name!r} must be > 0"
                    )
                self.log_mechanistic[name] = nn.Parameter(
                    torch.tensor(math.log(float(value)), dtype=torch.get_default_dtype())
                )
            else:
                fixed[name] = torch.tensor(float(value), dtype=torch.get_default_dtype())
        self._fixed_parameters = fixed

        projection = torch.zeros(len(problem.correction), problem.n_states)
        for row, state_index in enumerate(problem.correction):
            projection[row, state_index] = 1.0
        self.register_buffer("correction_projection", projection)

    def _parameters_for(self, x: torch.Tensor) -> ParameterMap:
        fixed = {name: value.to(device=x.device, dtype=x.dtype) for name, value in self._fixed_parameters.items()}
        return ParameterMap(fixed, self.log_mechanistic)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        known = self.problem.known_dynamics(t, x, self._parameters_for(x))
        correction = self.net(x) @ self.correction_projection.to(dtype=x.dtype)
        return known + correction

    def mechanistic_parameters(self) -> dict[str, float]:
        values = {name: float(value.detach()) for name, value in self._fixed_parameters.items()}
        values.update(
            {name: float(value.detach().exp()) for name, value in self.log_mechanistic.items()}
        )
        return values


class MechanisticODE(DifferentialEquation):
    """Complete mechanistic model, useful for baseline likelihood profiles."""

    def __init__(
        self,
        problem: Problem,
        *,
        fixed_params: Mapping[str, float] | None = None,
    ):
        super().__init__()
        self.problem = problem
        initial = problem.initial_parameters()
        overrides = dict(fixed_params or {})
        learnable = set(problem.learnable_params) - set(overrides)
        self.log_mechanistic = nn.ParameterDict()
        fixed: dict[str, torch.Tensor] = {}
        for name, value in initial.items():
            value = overrides.get(name, value)
            if name in learnable:
                if value <= 0:
                    raise ValueError(f"initial value for {name!r} must be > 0")
                self.log_mechanistic[name] = nn.Parameter(
                    torch.tensor(math.log(float(value)), dtype=torch.get_default_dtype())
                )
            else:
                fixed[name] = torch.tensor(float(value), dtype=torch.get_default_dtype())
        self._fixed_parameters = fixed

    def _parameters_for(self, x: torch.Tensor) -> ParameterMap:
        fixed = {name: value.to(device=x.device, dtype=x.dtype) for name, value in self._fixed_parameters.items()}
        return ParameterMap(fixed, self.log_mechanistic)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.problem.rhs(t, x, self._parameters_for(x))

    def mechanistic_parameters(self) -> dict[str, float]:
        values = {name: float(value.detach()) for name, value in self._fixed_parameters.items()}
        values.update(
            {name: float(value.detach().exp()) for name, value in self.log_mechanistic.items()}
        )
        return values


def build_model(
    problem: Problem,
    kind: str,
    config: TrainConfig | None = None,
    *,
    fixed_params: Mapping[str, float] | None = None,
) -> DifferentialEquation:
    """Construct one of the four model structures used by the analysis."""
    config = config or TrainConfig()
    if kind == "nde":
        if fixed_params:
            raise ValueError("fixed_params are not meaningful for a pure neural ODE")
        return NeuralODE(problem, config)
    if kind == "known_hybrid":
        return HybridODE(problem, config, learn_parameters=False, fixed_params=fixed_params)
    if kind == "unknown_hybrid":
        return HybridODE(problem, config, learn_parameters=True, fixed_params=fixed_params)
    if kind == "mechanistic":
        return MechanisticODE(problem, fixed_params=fixed_params)
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


def make_nn(problem: Problem, config: TrainConfig | None = None) -> NeuralODE:
    """Compatibility wrapper returning a pure neural ODE."""
    return NeuralODE(problem, config or TrainConfig())
