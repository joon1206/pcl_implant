"""Configuration and reproducible result export."""

from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .parameters import Geometry, ModelParameters, SimulationConfig
from .solver import SimulationResult


def _strict_dataclass(cls: type, values: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {', '.join(unknown)}")
    return cls(**values)


def load_configuration(path: str | Path) -> tuple[ModelParameters, SimulationConfig]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    parameters = _strict_dataclass(ModelParameters, raw.get("material", {}))
    simulation_values = dict(raw.get("simulation", {}))
    if "geometry" in simulation_values:
        simulation_values["geometry"] = Geometry(simulation_values["geometry"])
    config = _strict_dataclass(SimulationConfig, simulation_values)
    parameters.validate()
    config.validate()
    return parameters, config


def export_result(
    result: SimulationResult,
    parameters: ModelParameters,
    config: SimulationConfig,
    output_directory: str | Path,
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "material": parameters.as_dict(),
        "simulation": config.as_dict(),
        "diagnostics": result.diagnostics,
        "failure_time_days": result.failure_time_days,
        "final": {
            "mean_mn_kda": float(result.mean_mn_kda[-1]),
            "mass_retention": float(result.mass_retention[-1]),
            "modulus_parallel_fraction": float(result.modulus_parallel_fraction[-1]),
            "modulus_series_fraction": float(result.modulus_series_fraction[-1]),
            "strength_weakest_fraction": float(result.strength_weakest_fraction[-1]),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_days",
                "mean_mn_kda",
                "mass_retention",
                "modulus_parallel_fraction",
                "modulus_series_fraction",
                "strength_weakest_fraction",
            ]
        )
        writer.writerows(
            zip(
                result.time_days,
                result.mean_mn_kda,
                result.mass_retention,
                result.modulus_parallel_fraction,
                result.modulus_series_fraction,
                result.strength_weakest_fraction,
            )
        )

