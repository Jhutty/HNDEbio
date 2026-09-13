"""Portable result serialization."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any
import csv
import json

import numpy as np

from .core import write_json

if TYPE_CHECKING:
    from .core import AnalysisResult, FitResult


def save_fit(path, fit: "FitResult" | None = None) -> Path:
    """Save a fit as metadata, NumPy arrays, and a torch state dictionary.

    ``save_fit(store, fit)`` from the initial prototype remains supported.
    """
    if fit is None:
        raise TypeError("save_fit requires a path/store and FitResult")
    target = path.path if hasattr(path, "path") else Path(path)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "problem": fit.problem.name,
            "kind": fit.kind,
            "config": asdict(fit.config),
            "metrics": fit.metrics,
            "parameters": fit.parameters,
            "attempt": fit.attempt,
        },
        target / "result.json",
    )
    prediction = fit.prediction.detach().cpu().numpy()
    np.save(target / "prediction.npy", prediction)
    np.save(target / "losses.npy", np.asarray(fit.losses))
    import torch

    torch.save(fit.model.state_dict(), target / "model.pt")
    return target


def load_fit(path: str | Path, problem) -> "FitResult":
    """Restore a saved fit using its original :class:`Problem` definition."""
    from .core import FitResult, TrainConfig
    from .models import build_model

    root = Path(path)
    metadata = json.loads((root / "result.json").read_text())
    config = TrainConfig(**metadata["config"])
    import torch

    state = torch.load(root / "model.pt", map_location=config.device, weights_only=True)
    learned_names = {
        key.removeprefix("log_mechanistic.")
        for key in state
        if key.startswith("log_mechanistic.")
    }
    fixed_params = {
        name: value
        for name, value in metadata.get("parameters", {}).items()
        if name in problem.learnable_params and name not in learned_names
    }
    model = build_model(
        problem,
        metadata["kind"],
        config,
        fixed_params=fixed_params,
    ).to(config.device)
    model.load_state_dict(state)
    prediction = torch.as_tensor(
        np.load(root / "prediction.npy"), dtype=torch.float32, device=config.device
    )
    losses = np.load(root / "losses.npy").tolist()
    return FitResult(
        problem=problem,
        config=config,
        kind=metadata["kind"],
        model=model,
        prediction=prediction,
        losses=losses,
        metrics=metadata["metrics"],
        parameters=metadata.get("parameters", {}),
        attempt=metadata.get("attempt", 1),
    )


def save_analysis(path: str | Path, result: "AnalysisResult") -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for kind, fits in result.fits.items():
        for index, current in enumerate(fits):
            fit_path = root / "fits" / kind / f"fit_{index:03d}"
            save_fit(fit_path, current)
            summary.append(
                {
                    "problem": result.problem.name,
                    "kind": kind,
                    "fit": index,
                    "width": current.config.width,
                    "seed": current.config.seed,
                    **current.metrics,
                    **{f"parameter_{key}": value for key, value in current.parameters.items()},
                }
            )
    _save_rows(root / "fits.csv", summary)
    for kind, sweep in result.sweeps.items():
        _save_rows(root / f"hyperparameters_{kind}.csv", sweep.rows)
        write_json(asdict(sweep.best_config), root / f"best_config_{kind}.json")
        if sweep.best_by_width:
            write_json(
                {width: asdict(config) for width, config in sweep.best_by_width.items()},
                root / f"best_config_by_width_{kind}.json",
            )
    for kind, profile in result.profiles.items():
        _save_rows(root / f"profile_{kind}.csv", profile.rows)
    write_json(result.insights, root / "insights.json")
    write_json(
        {
            "problem": result.problem.name,
            "states": result.problem.states,
            "observed": result.problem.observed,
            "train_size": result.problem.train_size,
            "model_kinds": list(result.fits),
        },
        root / "analysis.json",
    )
    return root


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
