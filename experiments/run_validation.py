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

from pcl_model.distribution import random_scission_distribution_moments
from pcl_model.geometry import inspect_mesh
from pcl_model.kinetics import (
    autocatalytic_random_scission_mn,
    exponential_mn,
    fit_autocatalytic_random_scission,
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


def model_prediction(
    name: str, time_days: np.ndarray, mn0_kda: float, parameters: dict[str, float]
) -> np.ndarray:
    if name == "random_scission":
        return random_scission_mn(time_days, mn0_kda, parameters["k_inv_kda_day"])
    if name == "exponential":
        return exponential_mn(time_days, mn0_kda, parameters["rate_per_day"])
    return autocatalytic_random_scission_mn(
        time_days,
        mn0_kda,
        parameters["base_rate_inv_kda_day"],
        parameters["feedback_kda"],
    )


def empirical_validation(data_path: Path, output: Path) -> dict[str, object]:
    time_days, observed = load_dataset(data_path)
    split = 6  # fit 0--400 d; hold out 500 and 650 d before fitting
    fits = fit_kinetic_models(time_days[:split], observed[:split])
    mn0 = float(observed[0])
    predictions: dict[str, np.ndarray] = {}
    for name, fit in fits.items():
        values = fit["parameters"]
        prediction = model_prediction(name, time_days, mn0, values)
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
        dense = model_prediction(name, dense_time, mn0, values)
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


def cross_environment_validation(data_path: Path, output: Path) -> dict[str, object]:
    """Test water-calibrated kinetics on the separate PBS arm."""

    water_time, water = load_dataset(data_path, "water")
    pbs_time, pbs = load_dataset(data_path, "pbs")
    split = 6
    water_fits = fit_kinetic_models(water_time[:split], water[:split])
    pbs_fits = fit_kinetic_models(pbs_time[:split], pbs[:split])
    result: dict[str, object] = {"calibration_cutoff_day": 400.0, "models": {}}
    dense_time = np.linspace(0.0, max(water_time[-1], pbs_time[-1]), 500)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    colors = {
        "random_scission": "tab:orange",
        "exponential": "tab:blue",
        "autocatalytic_random_scission": "tab:green",
    }
    for name, water_fit in water_fits.items():
        no_refit = model_prediction(name, pbs_time, float(water[0]), water_fit["parameters"])
        refit = model_prediction(name, pbs_time, float(pbs[0]), pbs_fits[name]["parameters"])
        result["models"][name] = {
            "water_calibrated_no_refit_all_nonzero": metrics(pbs[1:], no_refit[1:]),
            "water_calibrated_no_refit_late": metrics(pbs[split:], no_refit[split:]),
            "pbs_calibrated_fit": metrics(pbs[:split], refit[:split]),
            "pbs_calibrated_held_out": metrics(pbs[split:], refit[split:]),
            "pbs_fit_parameters": pbs_fits[name]["parameters"],
        }
        axes[0].plot(
            dense_time,
            model_prediction(name, dense_time, float(water[0]), water_fit["parameters"]),
            color=colors[name],
            label=name.replace("_", " "),
        )
        axes[1].plot(
            dense_time,
            model_prediction(name, dense_time, float(pbs[0]), pbs_fits[name]["parameters"]),
            color=colors[name],
        )
    for axis in axes:
        axis.scatter(pbs_time[:split], pbs[:split], color="black", s=24, label="PBS data through 400 d")
        axis.scatter(pbs_time[split:], pbs[split:], facecolors="none", edgecolors="black", s=60, label="PBS held out")
        axis.set(xlabel="Immersion time (days)", ylabel=r"$M_n$ (kDa)")
        axis.grid(alpha=0.25)
    axes[0].set_title("Water calibration applied to PBS without refitting")
    axes[1].set_title("PBS calibration through day 400")
    axes[0].legend(fontsize=7)
    figure.savefig(output / "cross_environment_validation.png", dpi=220)
    plt.close(figure)
    return result


def uncertainty_and_identifiability(
    data_path: Path, output: Path, bootstrap_samples: int = 2000, seed: int = 20261007
) -> dict[str, object]:
    """Residual bootstrap plus local log-parameter identifiability diagnostics."""

    time_days, observed = load_dataset(data_path, "water")
    split = 6
    train_time, train_observed = time_days[:split], observed[:split]
    mn0 = float(observed[0])
    fitted = fit_autocatalytic_random_scission(train_time, train_observed, mn0_kda=mn0)
    fitted_parameters = np.exp(fitted.x)
    train_prediction = autocatalytic_random_scission_mn(train_time, mn0, *fitted_parameters)
    # Molecular weight is positive and errors grow with scale, so resample in
    # log space rather than allowing an additive bootstrap to predict Mn < 0.
    centered_log_residuals = np.log(train_observed / train_prediction)
    centered_log_residuals -= np.mean(centered_log_residuals)
    rng = np.random.default_rng(seed)
    parameter_draws = []
    prediction_draws = []
    predictive_draws = []
    dense_time = np.linspace(0.0, time_days[-1], 500)
    dense_draws = []
    for _ in range(bootstrap_samples):
        synthetic = train_prediction * np.exp(
            rng.choice(centered_log_residuals, size=train_time.size, replace=True)
        )
        synthetic[0] = mn0
        draw_fit = fit_autocatalytic_random_scission(train_time, synthetic, mn0_kda=mn0)
        if not draw_fit.success:
            continue
        parameters = np.exp(draw_fit.x)
        parameter_draws.append(parameters)
        prediction = autocatalytic_random_scission_mn(time_days, mn0, *parameters)
        prediction_draws.append(prediction)
        predictive_draws.append(
            prediction
            * np.exp(rng.choice(centered_log_residuals, size=time_days.size, replace=True))
        )
        dense_draws.append(autocatalytic_random_scission_mn(dense_time, mn0, *parameters))
    parameter_array = np.asarray(parameter_draws)
    prediction_array = np.asarray(prediction_draws)
    predictive_array = np.asarray(predictive_draws)
    dense_array = np.asarray(dense_draws)
    if parameter_array.shape[0] < 0.95 * bootstrap_samples:
        raise RuntimeError("too many bootstrap fits failed")

    jacobian_condition = float(np.linalg.cond(fitted.jac))
    covariance_shape = np.linalg.pinv(fitted.jac.T @ fitted.jac)
    scale = np.sqrt(np.diag(covariance_shape))
    local_correlation = covariance_shape / np.outer(scale, scale)
    confidence_interval = np.percentile(prediction_array, [2.5, 50.0, 97.5], axis=0)
    predictive_interval = np.percentile(predictive_array, [2.5, 50.0, 97.5], axis=0)
    parameter_interval = np.percentile(parameter_array, [2.5, 50.0, 97.5], axis=0)

    dense_interval = np.percentile(dense_array, [2.5, 50.0, 97.5], axis=0)
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.fill_between(dense_time, dense_interval[0], dense_interval[2], color="tab:green", alpha=0.22, label="95% bootstrap confidence band")
    axis.plot(dense_time, dense_interval[1], color="tab:green", label="bootstrap median")
    axis.scatter(train_time, train_observed, color="black", label="fit data")
    axis.scatter(time_days[split:], observed[split:], facecolors="none", edgecolors="black", s=70, label="held-out data")
    axis.set(xlabel="Immersion time (days)", ylabel=r"$M_n$ (kDa)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output / "bootstrap_uncertainty.png", dpi=220)
    plt.close(figure)

    prediction_rows = []
    for index, (time, value) in enumerate(zip(time_days, observed)):
        prediction_rows.append(
            {
                "time_days": float(time),
                "observed_mn_kda": float(value),
                "confidence_interval_kda": confidence_interval[:, index].tolist(),
                "predictive_interval_kda": predictive_interval[:, index].tolist(),
                "inside_95_percent_predictive_interval": bool(
                    predictive_interval[0, index] <= value <= predictive_interval[2, index]
                ),
            }
        )
    return {
        "method": "fixed-initial-condition multiplicative residual bootstrap in log(Mn)",
        "seed": seed,
        "requested_samples": bootstrap_samples,
        "successful_samples": int(parameter_array.shape[0]),
        "parameters": {
            "base_rate_inv_kda_day": parameter_interval[:, 0].tolist(),
            "feedback_kda": parameter_interval[:, 1].tolist(),
            "percentile_order": [2.5, 50.0, 97.5],
            "bootstrap_log_parameter_correlation": float(np.corrcoef(np.log(parameter_array).T)[0, 1]),
        },
        "local_log_parameter_jacobian_condition_number": jacobian_condition,
        "local_log_parameter_correlation": float(local_correlation[0, 1]),
        "predictions": prediction_rows,
    }


def distribution_level_prediction(data_path: Path, output: Path) -> dict[str, object]:
    """Map observed Mn to the ideal random-scission Mw and dispersity hypothesis."""

    time_days, observed = load_dataset(data_path, "water")
    moments = random_scission_distribution_moments(observed, float(observed[0]))
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    axis.plot(time_days, moments.dispersity, "o-", color="tab:purple")
    axis.set(xlabel="Immersion time (days)", ylabel=r"Predicted dispersity $M_w/M_n$")
    axis.set_ylim(0.95, 2.05)
    axis.grid(alpha=0.25)
    figure.savefig(output / "ideal_random_scission_dispersity.png", dpi=220)
    plt.close(figure)
    return {
        "assumption": "initially monodisperse chains with independent, uniformly probable bond cleavage",
        "initial_degree_of_polymerization": moments.initial_degree_of_polymerization,
        "time_days": time_days.tolist(),
        "inferred_bond_break_probability": moments.bond_break_probability.tolist(),
        "predicted_mw_kda": moments.mw_kda.tolist(),
        "predicted_dispersity": moments.dispersity.tolist(),
        "validation_status": "analytical prediction only; experimental SEC distribution data are not present in this dataset",
    }


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
        "cross_environment": cross_environment_validation(Path(args.data), output),
        "uncertainty_identifiability": uncertainty_and_identifiability(Path(args.data), output),
        "distribution_prediction": distribution_level_prediction(Path(args.data), output),
        "convergence": convergence_validation(output),
        "limiting_cases": limiting_cases(),
        "equal_sav_geometry": equal_sav_geometry_demo(output),
        "cad_audit": inspect_mesh(args.mesh, "mm").as_dict(),
    }
    report["runtime_seconds"] = time.perf_counter() - started
    (output / "validation_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

