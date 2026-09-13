"""Calibration, prediction, and scoring for UDE models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import math

import numpy as np
import torch

from .core import FitResult, Problem, TrainConfig, set_seed
from .models import build_model
from .solvers import integrate


ProgressCallback = Callable[[int, float], None]


def trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss: str = "mae",
) -> torch.Tensor:
    error = prediction - target
    if loss == "mae":
        return error.abs().mean()
    if loss == "mse":
        return error.square().mean()
    if loss == "rmse":
        return error.square().mean().sqrt()
    raise ValueError("loss must be one of: mae, mse, rmse")


def sample_time_batch(
    problem: Problem,
    config: TrainConfig,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample trajectory fragments as described in section 2.1 of the paper.

    Returns model-space initial states, the actual time grids for every fragment,
    and observed targets. Hidden-state initial values come from ``problem.x0``.
    """
    n_train = problem.train_size
    batch_time = min(config.batch_time, n_train)
    possible = n_train - batch_time + 1
    starts = torch.randint(
        possible,
        (config.batch_size,),
        generator=generator,
        device=config.device,
    )
    data = torch.as_tensor(problem.data, dtype=torch.float32, device=config.device)
    times = torch.as_tensor(problem.t, dtype=torch.float32, device=config.device)
    offsets = torch.arange(batch_time, device=config.device)[:, None]
    indices = starts[None, :] + offsets

    x0 = torch.as_tensor(problem.x0, dtype=torch.float32, device=config.device)
    initial = x0.expand(config.batch_size, -1).clone()
    initial[:, list(problem.observed)] = data[starts]
    batch_times = times[indices]
    targets = data[indices]
    return initial, batch_times, targets


def predict(
    model,
    problem: Problem,
    *,
    kind: str | None = None,
    t: np.ndarray | torch.Tensor | None = None,
    x0: np.ndarray | torch.Tensor | None = None,
    method: str = "rk4",
    max_step: float | None = None,
) -> torch.Tensor:
    """Integrate a fitted model over a requested time grid."""
    device = next(model.parameters(), torch.empty(0)).device
    times = torch.as_tensor(problem.t if t is None else t, dtype=torch.float32, device=device)
    initial_value = _initial_state(problem, kind or "unknown_hybrid") if x0 is None else x0
    initial = torch.as_tensor(initial_value, dtype=torch.float32, device=device)
    if max_step is None:
        max_step = problem.metadata.get("solver_max_step")
    return integrate(model, initial, times, method=method, max_step=max_step)


def score(problem: Problem, prediction: torch.Tensor) -> dict[str, float]:
    """Compute train/test MAE and RMSE on observed states."""
    observed = prediction[..., list(problem.observed)]
    target = torch.as_tensor(problem.data, dtype=observed.dtype, device=observed.device)
    split = problem.train_size

    def metrics(prefix: str, pred: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
        if actual.numel() == 0:
            return {f"{prefix}_mae": math.nan, f"{prefix}_rmse": math.nan}
        error = pred - actual
        return {
            f"{prefix}_mae": float(error.abs().mean().detach()),
            f"{prefix}_rmse": float(error.square().mean().sqrt().detach()),
        }

    values = metrics("train", observed[:split], target[:split])
    values.update(metrics("test", observed[split:], target[split:]))
    return values


def fit(
    problem: Problem,
    kind: str = "unknown_hybrid",
    config: TrainConfig | None = None,
    *,
    fixed_params: Mapping[str, float] | None = None,
    output: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> FitResult:
    """Calibrate one pure NDE, HNDE, or mechanistic model.

    Non-finite attempts are restarted up to ``config.max_attempts``. Mechanistic
    baselines use full-batch fitting; neural models use sampled time batches.
    """
    config = config or TrainConfig()
    if config.batch_time > problem.train_size:
        config = TrainConfig(**{**config.__dict__, "batch_time": problem.train_size})

    last_error: RuntimeError | None = None
    for attempt in range(1, config.max_attempts + 1):
        set_seed(config.seed + attempt - 1)
        model = build_model(problem, kind, config, fixed_params=fixed_params).to(config.device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            prediction = predict(model, problem, kind=kind).detach()
            metrics = score(problem, prediction)
            metrics["n_parameters"] = 0.0
            result = FitResult(problem, config, kind, model, prediction, [], metrics, model.mechanistic_parameters(), attempt)
            if output is not None:
                result.save(output)
            return result

        optimiser = torch.optim.Adam(trainable, lr=config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=config.lr_step, gamma=config.lr_gamma
        )
        generator = torch.Generator(device=config.device).manual_seed(config.seed + attempt - 1)
        losses: list[float] = []
        failed = False

        for iteration in range(1, config.iterations + 1):
            optimiser.zero_grad()
            if kind == "mechanistic":
                train_t = torch.as_tensor(
                    problem.t[: problem.train_size], dtype=torch.float32, device=config.device
                )
                initial = torch.as_tensor(
                    _initial_state(problem, kind), dtype=torch.float32, device=config.device
                )
                target = torch.as_tensor(
                    problem.data[: problem.train_size], dtype=torch.float32, device=config.device
                )
                raw_prediction = integrate(
                    model,
                    initial,
                    train_t,
                    max_step=config.solver_max_step or problem.metadata.get("solver_max_step"),
                )
            else:
                initial, train_t, target = sample_time_batch(problem, config, generator=generator)
                raw_prediction = integrate(
                    model,
                    initial,
                    train_t,
                    max_step=config.solver_max_step or problem.metadata.get("solver_max_step"),
                )

            observed_prediction = raw_prediction[..., list(problem.observed)]
            objective = trajectory_loss(observed_prediction, target, config.loss)
            if config.l1 and hasattr(model, "net"):
                objective = objective + config.l1 * sum(
                    parameter.abs().sum() for parameter in model.net.parameters()
                )
            if not torch.isfinite(objective):
                failed = True
                last_error = RuntimeError(
                    f"non-finite loss in attempt {attempt}, iteration {iteration}"
                )
                break
            objective.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimiser.step()
            scheduler.step()
            value = float(objective.detach())
            losses.append(value)
            if progress is not None:
                progress(iteration, value)

        if failed:
            continue

        model.eval()
        with torch.no_grad():
            prediction = predict(model, problem, kind=kind).detach()
        if not torch.isfinite(prediction).all():
            last_error = RuntimeError(
                f"non-finite full trajectory after calibration attempt {attempt}"
            )
            continue
        metrics = score(problem, prediction)
        metrics["loss"] = losses[-1]
        metrics["n_parameters"] = float(sum(parameter.numel() for parameter in trainable))
        result = FitResult(
            problem=problem,
            config=config,
            kind=kind,
            model=model,
            prediction=prediction,
            losses=losses,
            metrics=metrics,
            parameters=model.mechanistic_parameters(),
            attempt=attempt,
        )
        if output is not None:
            result.save(output)
        return result

    raise last_error or RuntimeError(f"calibration failed after {config.max_attempts} attempts")


def _initial_state(problem: Problem, kind: str):
    if kind == "mechanistic" and "mechanistic_x0" in problem.metadata:
        return problem.metadata["mechanistic_x0"]
    return problem.x0


def train_nn(
    problem: Problem,
    config: TrainConfig | None = None,
    out: str | Path | None = None,
) -> FitResult:
    """Compatibility wrapper for fitting a pure neural ODE."""
    return fit(problem, "nde", config, output=out)
