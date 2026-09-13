"""Composable analysis stages and high-level paper reproduction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any
import csv

import numpy as np

from .core import AnalysisResult, FitResult, Problem, ProfileResult, SweepResult, TrainConfig
from .io import save_analysis
from .training import fit


PAPER_HYPERPARAMETERS: dict[str, tuple[Any, ...]] = {
    "batch_time": (16, 32, 64),
    "batch_size": (64, 128, 256),
    "lr": (1e-1, 1e-2),
    "lr_step": (500, 1_000),
    "iterations": (500, 1_000, 2_000, 4_000),
    "width": (5, 15, 25),
}


def hyperparameter_sweep(
    problem: Problem,
    kind: str,
    grid: Mapping[str, Sequence[Any]],
    *,
    base_config: TrainConfig | None = None,
    replicates: int = 1,
    objective: str = "train_rmse",
    output: str | Path | None = None,
) -> SweepResult:
    """Fit every configuration in a grid and select by training performance."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    base = base_config or TrainConfig()
    unknown = set(grid) - set(asdict(base))
    if unknown:
        raise ValueError(f"unknown TrainConfig fields: {sorted(unknown)}")
    keys = tuple(grid)
    rows: list[dict[str, Any]] = []
    for combination in product(*(grid[key] for key in keys)):
        changes = dict(zip(keys, combination))
        for replicate in range(replicates):
            config = replace(base, **changes, seed=base.seed + replicate)
            result = fit(problem, kind, config)
            row = {
                "problem": problem.name,
                "kind": kind,
                "replicate": replicate,
                **changes,
                **result.metrics,
            }
            rows.append(row)
            if output is not None:
                _write_rows(Path(output), rows)

    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(float(row[objective]))
    best_values = min(grouped, key=lambda key: float(np.mean(grouped[key])))
    best = replace(base, **dict(zip(keys, best_values)))
    best_by_width: dict[int, TrainConfig] = {}
    if "width" in keys:
        width_index = keys.index("width")
        for width in grid["width"]:
            candidates = [key for key in grouped if key[width_index] == width]
            best_key = min(candidates, key=lambda key: float(np.mean(grouped[key])))
            best_by_width[int(width)] = replace(base, **dict(zip(keys, best_key)))
    return SweepResult(rows, best, objective, best_by_width)


def compare_models(
    problem: Problem,
    *,
    kinds: Sequence[str] = ("nde", "known_hybrid", "unknown_hybrid"),
    configs: TrainConfig | Mapping[str, TrainConfig | Mapping[int, TrainConfig]] | None = None,
    widths: Sequence[int] = (5, 15, 25),
    replicates: int = 1,
    output: str | Path | None = None,
) -> dict[str, list[FitResult]]:
    """Compare model structures and widths using repeated fits."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    fits: dict[str, list[FitResult]] = {}
    for kind in kinds:
        if isinstance(configs, Mapping):
            configured = configs[kind]
        else:
            configured = configs or TrainConfig()
        current: list[FitResult] = []
        for width in widths:
            base = configured.get(width, next(iter(configured.values()))) if isinstance(configured, Mapping) else configured
            for replicate in range(replicates):
                config = replace(base, width=width, seed=base.seed + replicate)
                fit_output = (
                    Path(output) / kind / f"fit_{len(current):03d}"
                    if output is not None
                    else None
                )
                current.append(fit(problem, kind, config, output=fit_output))
        fits[kind] = current
    return fits


def likelihood_profile(
    problem: Problem,
    *,
    kind: str = "unknown_hybrid",
    parameters: Iterable[str] | None = None,
    config: TrainConfig | None = None,
    values: Mapping[str, Sequence[float]] | None = None,
    samples: int = 11,
    log10_span: float = 1.5,
    replicates: int = 1,
    output: str | Path | None = None,
) -> ProfileResult:
    """Profile mechanistic parameters by fixing each value and recalibrating.

    This is the practical-identifiability method used in section 2.5. Values
    default to 11 log-spaced points over true value ±1.5 log10 units.
    """
    if kind not in {"unknown_hybrid", "mechanistic"}:
        raise ValueError("profiles require kind='unknown_hybrid' or 'mechanistic'")
    config = config or TrainConfig(width=5)
    names = tuple(parameters or problem.learnable_params)
    if not names:
        raise ValueError("the problem has no learnable mechanistic parameters")
    absent = set(names) - set(problem.params)
    if absent:
        raise ValueError(f"profile parameters absent from problem.params: {sorted(absent)}")
    rows: list[dict[str, Any]] = []

    for name in names:
        true_value = float(problem.params[name])
        profile_values = (
            np.asarray(values[name], dtype=float)
            if values is not None and name in values
            else np.logspace(
                np.log10(true_value) - log10_span,
                np.log10(true_value) + log10_span,
                samples,
            )
        )
        for value in profile_values:
            for replicate in range(replicates):
                current = replace(config, seed=config.seed + replicate)
                result = fit(problem, kind, current, fixed_params={name: float(value)})
                rows.append(
                    {
                        "problem": problem.name,
                        "kind": kind,
                        "parameter": name,
                        "value": float(value),
                        "log10_value": float(np.log10(value)),
                        "true_value": true_value,
                        "replicate": replicate,
                        **result.metrics,
                    }
                )
                if output is not None:
                    _write_rows(Path(output), rows)
    return ProfileResult(rows, names, kind)


def analyze(
    problem: Problem,
    *,
    config: TrainConfig | None = None,
    kinds: Sequence[str] = ("nde", "known_hybrid", "unknown_hybrid"),
    hyperparameters: Mapping[str, Sequence[Any]] | None = None,
    profiles: bool = False,
    profile_baseline: bool = True,
    widths: Sequence[int] = (15,),
    replicates: int = 1,
    sweep_replicates: int | None = None,
    profile_samples: int = 11,
    profile_replicates: int = 1,
    profile_baseline_replicates: int | None = None,
    output: str | Path | None = None,
) -> AnalysisResult:
    """Run a complete or selected analysis from a single :class:`Problem`.

    Supplying a hyperparameter grid selects a configuration separately for each
    model structure before the comparison. Setting ``profiles=True`` adds HNDE
    and complete-mechanistic likelihood profiles.
    """
    base = config or TrainConfig()
    output_root = Path(output) if output is not None else None
    sweeps: dict[str, SweepResult] = {}
    selected: dict[str, TrainConfig | Mapping[int, TrainConfig]] = {}
    if hyperparameters:
        for kind in kinds:
            sweep = hyperparameter_sweep(
                problem,
                kind,
                hyperparameters,
                base_config=base,
                replicates=sweep_replicates or replicates,
                output=output_root / f"hyperparameters_{kind}.csv" if output_root else None,
            )
            sweeps[kind] = sweep
            selected[kind] = sweep.best_by_width or sweep.best_config
    else:
        selected = {kind: base for kind in kinds}

    fitted = compare_models(
        problem,
        kinds=kinds,
        configs=selected,
        widths=widths,
        replicates=replicates,
        output=output_root / "fits" if output_root else None,
    )
    profile_results: dict[str, ProfileResult] = {}
    if profiles and problem.learnable_params:
        profile_selection = selected.get("unknown_hybrid", base)
        if isinstance(profile_selection, Mapping):
            profile_config = profile_selection.get(5, next(iter(profile_selection.values())))
        else:
            profile_config = replace(profile_selection, width=5)
        profile_results["unknown_hybrid"] = likelihood_profile(
            problem,
            kind="unknown_hybrid",
            config=profile_config,
            samples=profile_samples,
            replicates=profile_replicates,
            output=output_root / "profile_unknown_hybrid.csv" if output_root else None,
        )
        if profile_baseline:
            profile_results["mechanistic"] = likelihood_profile(
                problem,
                kind="mechanistic",
                config=profile_config,
                samples=profile_samples,
                replicates=profile_baseline_replicates or profile_replicates,
                output=output_root / "profile_mechanistic.csv" if output_root else None,
            )

    result = AnalysisResult(
        problem=problem,
        fits=fitted,
        sweeps=sweeps,
        profiles=profile_results,
        output_dir=Path(output) if output is not None else None,
    )
    from .insights import collect_insights

    result.insights = collect_insights(result)
    if output is not None:
        save_analysis(output, result)
    return result


def reproduce_paper(
    problem: Problem | None = None,
    *,
    output: str | Path = "results/paper",
    hyperparameter_replicates: int = 9,
    comparison_replicates: int = 50,
    profile_replicates: int = 20,
) -> dict[str, AnalysisResult] | AnalysisResult:
    """Run the experimental design stated in the paper.

    This is intentionally computationally expensive: the factorial grid has 432
    settings per model structure before replication.
    """
    from .examples import glycolysis, lotka_volterra

    def one(current: Problem) -> AnalysisResult:
        return analyze(
            current,
            hyperparameters=PAPER_HYPERPARAMETERS,
            profiles=True,
            widths=(5, 15, 25),
            replicates=comparison_replicates,
            sweep_replicates=hyperparameter_replicates,
            profile_samples=11,
            profile_replicates=profile_replicates,
            profile_baseline_replicates=3,
            output=Path(output) / current.name,
        )

    if problem is not None:
        return one(problem)
    return {"glycolysis": one(glycolysis()), "lotka_volterra": one(lotka_volterra())}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
