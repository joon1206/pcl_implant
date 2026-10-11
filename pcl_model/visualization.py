"""Scientific plotting functions; all labels include units or normalization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .solver import SimulationResult


def plot_summary(result: SimulationResult, output_directory: str | Path) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(result.time_days, result.mean_mn_kda, color="tab:blue")
    axes[0, 0].set(xlabel="Time (days)", ylabel=r"Volume-mean $M_n$ (kDa)")
    axes[0, 1].plot(result.time_days, result.mass_retention, color="tab:green")
    axes[0, 1].set(xlabel="Time (days)", ylabel="Dry-mass retention (-)", ylim=(0, 1.03))
    axes[1, 0].plot(result.time_days, result.modulus_parallel_fraction, label="parallel / Voigt")
    axes[1, 0].plot(result.time_days, result.modulus_series_fraction, label="series / Reuss")
    axes[1, 0].plot(result.time_days, result.strength_weakest_fraction, label="weakest strength")
    axes[1, 0].set(xlabel="Time (days)", ylabel="Mechanical retention (-)", ylim=(0, 1.15))
    axes[1, 0].legend(fontsize=8)
    indices = np.unique(np.linspace(0, len(result.time_days) - 1, 5, dtype=int))
    for index in indices:
        axes[1, 1].plot(
            result.radius_mm,
            result.mn_kda[index],
            label=f"{result.time_days[index]:g} d",
        )
    axes[1, 1].set(xlabel="Centre-to-surface coordinate (mm)", ylabel=r"Local $M_n$ (kDa)")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    path = output / "simulation_summary.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_profiles(result: SimulationResult, output_directory: str | Path) -> Path:
    output = Path(output_directory)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    indices = np.unique(np.linspace(0, len(result.time_days) - 1, 5, dtype=int))
    fields = [
        (result.acid, "Normalized acid concentration (-)"),
        (result.crystallinity, "Crystalline fraction (-)"),
        (result.solid_fraction, "Local solid fraction (-)"),
    ]
    for axis, (values, label) in zip(axes, fields):
        for index in indices:
            axis.plot(result.radius_mm, values[index], label=f"{result.time_days[index]:g} d")
        axis.set(xlabel="Centre-to-surface coordinate (mm)", ylabel=label)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    path = output / "spatial_profiles.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path

