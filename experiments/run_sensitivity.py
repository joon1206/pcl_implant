"""Run a deterministic global sensitivity screen of the coupled PDE model."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcl_model.parameters import Geometry, ModelParameters, SimulationConfig
from pcl_model.sensitivity import ParameterRange, saltelli_matrices, sobol_indices
from pcl_model.solver import simulate


RANGES = [
    ParameterRange("scission_rate_inv_kda_day", 2e-5, 8e-5, logarithmic=True),
    ParameterRange("autocatalysis_per_acid", 0.5, 6.0),
    ParameterRange("acid_diffusivity_mm2_day", 0.002, 0.2, logarithmic=True),
    ParameterRange("mass_transfer_mm_day", 0.005, 0.5, logarithmic=True),
    ParameterRange("chemicrystallization_gain", 0.05, 0.30),
    ParameterRange("dissolution_rate_per_day", 0.01, 0.08, logarithmic=True),
]


def evaluate(rows: np.ndarray, config: SimulationConfig) -> dict[str, np.ndarray]:
    outputs = {"final_mn_kda": [], "mass_retention": [], "weakest_strength_fraction": []}
    for row in rows:
        values = dict(zip((item.name for item in RANGES), map(float, row)))
        result = simulate(replace(ModelParameters(), **values), config)
        outputs["final_mn_kda"].append(float(result.mean_mn_kda[-1]))
        outputs["mass_retention"].append(float(result.mass_retention[-1]))
        outputs["weakest_strength_fraction"].append(float(result.strength_weakest_fraction[-1]))
    return {name: np.asarray(values) for name, values in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/sensitivity")
    parser.add_argument("--base-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(
        geometry=Geometry.CYLINDER,
        characteristic_length_mm=2.0,
        duration_days=730.0,
        output_interval_days=730.0,
        cells=20,
        relative_tolerance=1e-5,
        absolute_tolerance=1e-8,
        maximum_step_days=5.0,
    )
    started = time.perf_counter()
    a, b, hybrids = saltelli_matrices(RANGES, args.base_samples, args.seed)
    matrices = [a, b, *hybrids]
    evaluated = []
    for index, matrix in enumerate(matrices, start=1):
        print(f"evaluating Saltelli block {index}/{len(matrices)}", flush=True)
        evaluated.append(evaluate(matrix, config))

    report: dict[str, object] = {
        "method": "scrambled Sobol sequence; Saltelli first-order and Jansen total-order estimators",
        "seed": args.seed,
        "base_samples": args.base_samples,
        "model_evaluations": args.base_samples * (len(RANGES) + 2),
        "simulation_config": config.as_dict(),
        "parameter_ranges": [item.__dict__ for item in RANGES],
        "outputs": {},
    }
    csv_rows = []
    for output_name in evaluated[0]:
        first, total = sobol_indices(
            evaluated[0][output_name],
            evaluated[1][output_name],
            [block[output_name] for block in evaluated[2:]],
        )
        convergence = None
        if args.base_samples >= 4:
            half = args.base_samples // 2
            first_half, total_half = sobol_indices(
                evaluated[0][output_name][:half],
                evaluated[1][output_name][:half],
                [block[output_name][:half] for block in evaluated[2:]],
            )
            convergence = {
                "first_order_absolute_change_from_half_sample": np.abs(first - first_half).tolist(),
                "total_order_absolute_change_from_half_sample": np.abs(total - total_half).tolist(),
            }
        report["outputs"][output_name] = {
            "first_order": first.tolist(),
            "total_order": total.tolist(),
            "parameter_order": [item.name for item in RANGES],
            "output_mean": float(np.mean(np.concatenate([evaluated[0][output_name], evaluated[1][output_name]]))),
            "output_standard_deviation": float(np.std(np.concatenate([evaluated[0][output_name], evaluated[1][output_name]]), ddof=1)),
            "sample_size_convergence": convergence,
        }
        report["outputs"][output_name]["estimator_quality"] = (
            "screening-grade"
            if np.all(np.isfinite(first))
            and np.all(np.isfinite(total))
            and np.max(np.abs(first)) <= 1.5
            and np.max(total) <= 1.5
            and convergence is not None
            and max(convergence["first_order_absolute_change_from_half_sample"]) <= 0.35
            and max(convergence["total_order_absolute_change_from_half_sample"]) <= 0.35
            else "unstable; do not rank parameters from these indices"
        )
        for item, first_value, total_value in zip(RANGES, first, total):
            csv_rows.append(
                {
                    "output": output_name,
                    "parameter": item.name,
                    "first_order": float(first_value),
                    "total_order": float(total_value),
                }
            )

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    labels = [item.name.replace("_", "\n") for item in RANGES]
    x = np.arange(len(RANGES))
    for axis, (output_name, values) in zip(axes, report["outputs"].items()):
        axis.bar(x - 0.18, values["first_order"], width=0.36, label="first order")
        axis.bar(x + 0.18, values["total_order"], width=0.36, label="total order")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
        axis.set_title(output_name.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
        if values["estimator_quality"].startswith("unstable"):
            axis.text(0.5, 0.98, "unstable estimator", transform=axis.transAxes, ha="center", va="top", color="crimson")
    axes[0].set_ylabel("Sobol index estimate")
    axes[0].legend(fontsize=8)
    figure.savefig(output / "sobol_indices.png", dpi=220)
    plt.close(figure)

    report["runtime_seconds"] = time.perf_counter() - started
    (output / "sensitivity_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    with (output / "sobol_indices.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["output", "parameter", "first_order", "total_order"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
