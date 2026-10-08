"""Reproduce empirical, numerical, geometry, and CAD validation evidence."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcl_model.geometry import inspect_mesh
from pcl_model.kinetics import (
    autocatalytic_random_scission_mn,
    exponential_mn,
    fit_kinetic_models,
    random_scission_mn,
)
from pcl_model.parameters import Geometry, ModelParameters, SimulationConfig
from pcl_model.solver import simulate


def load_dataset(path: Path, environment: str = "water") -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["environment"] == environment]
    return (
        np.array([float(row["time_days"]) for row in rows]),
        np.array([float(row["mn_kda"]) for row in rows]),
    )


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    return {
        "rmse_kda": float(np.sqrt(np.mean(residual**2))),
        "mae_kda": float(np.mean(np.abs(residual))),
    }


def empirical_validation(data_path: Path, output: Path) -> dict[str, object]:
    time_days, observed = load_dataset(data_path)
    split = 6  # fit 0--400 d; hold out 500 and 650 d before fitting
    fits = fit_kinetic_models(time_days[:split], observed[:split])
    mn0 = float(observed[0])
    predictions: dict[str, np.ndarray] = {}
    for name, fit in fits.items():
        values = fit["parameters"]
        if name == "random_scission":
            prediction = random_scission_mn(time_days, mn0, values["k_inv_kda_day"])
        elif name == "exponential":
            prediction = exponential_mn(time_days, mn0, values["rate_per_day"])
        else:
            prediction = autocatalytic_random_scission_mn(
                time_days,
                mn0,
                values["base_rate_inv_kda_day"],
                values["feedback_kda"],
            )
        predictions[name] = prediction
        fit["fit_metrics"] = metrics(observed[:split], prediction[:split])
        fit["held_out_metrics"] = metrics(observed[split:], prediction[split:])
        fit.pop("prediction", None)

    dense_time = np.linspace(0.0, time_days[-1], 500)
    figure, (main_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True
    )
    main_axis.scatter(time_days[:split], observed[:split], color="black", label="fit data")
    main_axis.scatter(
        time_days[split:], observed[split:], facecolors="none", edgecolors="black", s=70, label="held-out data"
    )
    colors = {"random_scission": "tab:orange", "exponential": "tab:blue", "autocatalytic_random_scission": "tab:green"}
    labels = {"random_scission": "constant random scission", "exponential": "empirical exponential", "autocatalytic_random_scission": "autocatalytic random scission"}
    for name, fit in fits.items():
        values = fit["parameters"]
        if name == "random_scission":
            dense = random_scission_mn(dense_time, mn0, values["k_inv_kda_day"])
        elif name == "exponential":
            dense = exponential_mn(dense_time, mn0, values["rate_per_day"])
        else:
            dense = autocatalytic_random_scission_mn(
                dense_time, mn0, values["base_rate_inv_kda_day"], values["feedback_kda"]
            )
        main_axis.plot(dense_time, dense, color=colors[name], label=labels[name])
        residual_axis.plot(time_days, predictions[name] - observed, "o-", color=colors[name], label=labels[name])
    main_axis.set(ylabel=r"$M_n$ (kDa)")
    main_axis.grid(alpha=0.25)
    main_axis.legend(fontsize=8)
    residual_axis.axhline(0.0, color="black", linewidth=0.8)
    residual_axis.axvline(time_days[split] - 1, color="0.5", linestyle="--", linewidth=0.8)
    residual_axis.set(xlabel="Immersion time (days)", ylabel="Residual (kDa)")
    residual_axis.grid(alpha=0.25)
    figure.savefig(output / "model_vs_experiment.png", dpi=220)
    plt.close(figure)
    return {"dataset": str(data_path), "fit_through_day": 400.0, "held_out_days": [500.0, 650.0], "models": fits}


def convergence_validation(output: Path) -> dict[str, object]:
    material = ModelParameters(
        scission_rate_inv_kda_day=5e-5,
        autocatalysis_per_acid=3.0,
        acid_yield_kda=40.0,
        mass_transfer_mm_day=0.01,
        dissolution_rate_per_day=0.0,
    )
    base = SimulationConfig(duration_days=180.0, output_interval_days=2.0, maximum_step_days=1.0)
    cell_counts = [20, 40, 80, 160]
    results = [simulate(material, replace(base, cells=cells)) for cells in cell_counts]
    reference = results[-1]
    entries = []
    for cells, result in zip(cell_counts, results):
        entries.append(
            {
                "cells": cells,
                "mean_mn_final_kda": float(result.mean_mn_kda[-1]),
                "absolute_error_vs_160_cells_kda": float(abs(result.mean_mn_kda[-1] - reference.mean_mn_kda[-1])),
            }
        )

    step_sizes = [4.0, 2.0, 1.0, 0.5]
    temporal_results = [simulate(material, replace(base, cells=80, maximum_step_days=step)) for step in step_sizes]
    temporal_reference = temporal_results[-1]
    temporal = [
        {
            "maximum_step_days": step,
            "mean_mn_final_kda": float(result.mean_mn_kda[-1]),
            "absolute_error_vs_0.5_day_step_kda": float(abs(result.mean_mn_kda[-1] - temporal_reference.mean_mn_kda[-1])),
        }
        for step, result in zip(step_sizes, temporal_results)
    ]
    return {"spatial": entries, "temporal": temporal}


def limiting_cases() -> dict[str, object]:
    base = SimulationConfig(duration_days=100.0, output_interval_days=5.0, cells=40)
    no_hydrolysis = simulate(ModelParameters(scission_rate_inv_kda_day=1e-20), base)
    no_auto = simulate(ModelParameters(autocatalysis_per_acid=0.0), base)
    cleared = simulate(ModelParameters(mass_transfer_mm_day=10.0), base)
    retained = simulate(ModelParameters(mass_transfer_mm_day=0.0), base)
    return {
        "vanishing_hydrolysis_mn_change_kda": float(no_hydrolysis.mean_mn_kda[-1] - no_hydrolysis.mean_mn_kda[0]),
        "no_autocatalysis_final_mn_kda": float(no_auto.mean_mn_kda[-1]),
        "strong_clearance_final_mn_kda": float(cleared.mean_mn_kda[-1]),
        "no_clearance_final_mn_kda": float(retained.mean_mn_kda[-1]),
        "clearance_slows_degradation": bool(cleared.mean_mn_kda[-1] > retained.mean_mn_kda[-1]),
    }


def equal_sav_geometry_demo(output: Path) -> dict[str, object]:
    sav = 1.0
    material = ModelParameters(
        acid_diffusivity_mm2_day=0.002,
        mass_transfer_mm_day=0.02,
        autocatalysis_per_acid=5.0,
        acid_yield_kda=60.0,
        dissolution_rate_per_day=0.0,
    )
    curves = {}
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for geometry in Geometry:
        length = geometry.surface_to_volume_factor / sav
        config = SimulationConfig(
            geometry=geometry,
            characteristic_length_mm=length,
            duration_days=365.0,
            output_interval_days=5.0,
            cells=80,
            maximum_step_days=1.0,
        )
        result = simulate(material, config)
        axis.plot(result.time_days, result.mean_mn_kda, label=f"{geometry.value}: L={length:g} mm")
        curves[geometry.value] = {
            "characteristic_length_mm": length,
            "surface_to_volume_per_mm": config.surface_to_volume_per_mm,
            "final_mean_mn_kda": float(result.mean_mn_kda[-1]),
            "damkohler": result.diagnostics["damkohler_scission"],
        }
    axis.set(xlabel="Time (days)", ylabel=r"Volume-mean $M_n$ (kDa)")
    axis.set_title(r"Equal global SA/V = 1 mm$^{-1}$ does not fix transport length")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output / "equal_sav_geometry_comparison.png", dpi=220)
    plt.close(figure)
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/validation")
    parser.add_argument("--data", default="data/gil_castell_2019_pcl_mn.csv")
    parser.add_argument("--mesh", default="Snap-Fit v5.stl")
    args = parser.parse_args()
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "empirical": empirical_validation(Path(args.data), output),
        "convergence": convergence_validation(output),
        "limiting_cases": limiting_cases(),
        "equal_sav_geometry": equal_sav_geometry_demo(output),
        "cad_audit": inspect_mesh(args.mesh, "mm").as_dict(),
    }
    report["runtime_seconds"] = time.perf_counter() - started
    (output / "validation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

