"""Turn analysis artifacts into compact, reproducible scientific summaries."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import Any

import numpy as np

from .core import AnalysisResult, ProfileResult, SweepResult


HYPERPARAMETER_FIELDS = (
    "batch_time",
    "batch_size",
    "lr",
    "lr_step",
    "iterations",
    "width",
)


def summarize_fits(result: AnalysisResult, metric: str = "test_rmse") -> list[dict[str, Any]]:
    """Mean and sample standard deviation by model kind and network width."""
    rows: list[dict[str, Any]] = []
    for kind, fits in result.fits.items():
        widths = sorted({fit.config.width for fit in fits})
        for width in widths:
            values = np.asarray(
                [fit.metrics[metric] for fit in fits if fit.config.width == width],
                dtype=float,
            )
            rows.append(
                {
                    "kind": kind,
                    "width": width,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            )
    return rows


def hyperparameter_importance(
    sweep: SweepResult,
    *,
    metric: str | None = None,
    fields: Iterable[str] = HYPERPARAMETER_FIELDS,
    interactions: bool = True,
) -> list[dict[str, Any]]:
    """Descriptive ANOVA-style effect fractions for a balanced sweep.

    The fraction is a factor or pairwise interaction sum of squares divided by
    total sum of squares. It is intended to rank tuning priorities, not replace
    a formal inferential ANOVA with its associated assumptions.
    """
    metric = metric or sweep.objective
    rows = sweep.rows
    y = np.asarray([row[metric] for row in rows], dtype=float)
    grand = float(np.mean(y))
    total = float(np.sum((y - grand) ** 2))
    names = [
        field
        for field in fields
        if field in rows[0] and len({row[field] for row in rows}) > 1
    ]
    marginal: dict[str, dict[Any, float]] = {}
    effects: list[dict[str, Any]] = []
    for name in names:
        levels = {row[name] for row in rows}
        means = {
            level: float(np.mean([row[metric] for row in rows if row[name] == level]))
            for level in levels
        }
        marginal[name] = means
        ss = sum(
            sum(row[name] == level for row in rows) * (mean - grand) ** 2
            for level, mean in means.items()
        )
        effects.append(
            {
                "effect": name,
                "type": "main",
                "sum_squares": float(ss),
                "fraction_total": float(ss / total) if total else 0.0,
            }
        )
    if interactions:
        for left, right in combinations(names, 2):
            pairs = {(row[left], row[right]) for row in rows}
            ss = 0.0
            for left_level, right_level in pairs:
                cell = [
                    row[metric]
                    for row in rows
                    if row[left] == left_level and row[right] == right_level
                ]
                interaction = (
                    float(np.mean(cell))
                    - marginal[left][left_level]
                    - marginal[right][right_level]
                    + grand
                )
                ss += len(cell) * interaction**2
            effects.append(
                {
                    "effect": f"{left}:{right}",
                    "type": "interaction",
                    "sum_squares": float(ss),
                    "fraction_total": float(ss / total) if total else 0.0,
                }
            )
    return sorted(effects, key=lambda row: row["fraction_total"], reverse=True)


def assess_identifiability(
    profile: ProfileResult,
    *,
    metric: str = "train_rmse",
    minimum_r2: float = 0.8,
) -> list[dict[str, Any]]:
    """Apply a transparent convexity heuristic to each loss profile.

    A profile is flagged identifiable when its median curve has an interior
    minimum near the true value and a convex quadratic explains it well. This
    operationalises the paper's visual criterion but is not structural
    identifiability analysis.
    """
    conclusions: list[dict[str, Any]] = []
    for name in profile.parameter_names:
        rows = profile.for_parameter(name)
        values = np.asarray(sorted({row["value"] for row in rows}), dtype=float)
        x = np.log10(values)
        y = np.asarray(
            [np.median([row[metric] for row in rows if row["value"] == value]) for value in values]
        )
        coefficients = np.polyfit(x, y, 2)
        fitted = np.polyval(coefficients, x)
        residual = float(np.sum((y - fitted) ** 2))
        total = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - residual / total if total else 0.0
        minimum_index = int(np.argmin(y))
        spacing = float(np.median(np.diff(x))) if len(x) > 1 else np.inf
        true_log = float(np.log10(rows[0]["true_value"]))
        near_true = abs(x[minimum_index] - true_log) <= spacing + 1e-12
        interior = 0 < minimum_index < len(values) - 1
        convex = bool(coefficients[0] > 0)
        identifiable = bool(interior and near_true and convex and r2 >= minimum_r2)
        conclusions.append(
            {
                "parameter": name,
                "kind": profile.kind,
                "identifiable_heuristic": identifiable,
                "minimum": float(values[minimum_index]),
                "true_value": float(rows[0]["true_value"]),
                "minimum_is_interior": interior,
                "minimum_near_true": near_true,
                "quadratic_curvature": float(coefficients[0]),
                "quadratic_r2": float(r2),
                "criterion": f"interior, near true, convex, quadratic_r2 >= {minimum_r2}",
            }
        )
    return conclusions


def collect_insights(result: AnalysisResult) -> dict[str, Any]:
    """Collect all summaries available in an analysis result."""
    return {
        "model_performance": summarize_fits(result),
        "hyperparameter_importance": {
            kind: hyperparameter_importance(sweep)
            for kind, sweep in result.sweeps.items()
        },
        "identifiability": {
            kind: assess_identifiability(profile)
            for kind, profile in result.profiles.items()
        },
    }
