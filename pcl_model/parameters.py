"""Validated, explicitly dimensioned model inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Geometry(str, Enum):
    """One-dimensional symmetric transport geometry."""

    SLAB = "slab"
    CYLINDER = "cylinder"
    SPHERE = "sphere"

    @property
    def radial_power(self) -> int:
        return {self.SLAB: 0, self.CYLINDER: 1, self.SPHERE: 2}[self]

    @property
    def surface_to_volume_factor(self) -> int:
        """SA/V = factor / characteristic_length for the represented solid."""
        return self.radial_power + 1


@dataclass(frozen=True)
class ModelParameters:
    """Material and environmental parameters.

    Canonical units are millimetres, days, kilodaltons, and megapascals.
    Acid concentration is normalized by a declared reference concentration.
    """

    mn0_kda: float = 42.03
    scission_rate_inv_kda_day: float = 4.72e-5
    autocatalysis_per_acid: float = 2.4
    acid_yield_kda: float = 47.8
    acid_diffusivity_mm2_day: float = 0.02
    mass_transfer_mm_day: float = 0.2
    bulk_clearance_per_day: float = 0.0
    external_acid: float = 0.0

    crystallinity0: float = 0.45
    chemicrystallization_gain: float = 0.20
    chemicrystallization_rate_per_day: float = 0.02
    crystal_hydrolysis_per_day: float = 0.0
    crystal_transport_barrier: float = 2.0
    porosity_transport_gain: float = 3.0

    soluble_mn_kda: float = 3.0
    dissolution_width_kda: float = 0.35
    dissolution_rate_per_day: float = 0.03

    modulus0_mpa: float = 350.0
    strength0_mpa: float = 16.0
    crystal_to_amorphous_modulus_ratio: float = 4.0
    critical_entanglement_mn_kda: float = 2.5
    modulus_tie_exponent: float = 0.25
    strength_tie_exponent: float = 1.0
    porosity_modulus_exponent: float = 2.0
    porosity_strength_exponent: float = 1.5

    temperature_c: float = 37.0
    reference_temperature_c: float = 37.0
    activation_energy_kj_mol: float = 60.0
    ph_rate_multiplier: float = 1.0

    def validate(self) -> None:
        positive = {
            "mn0_kda": self.mn0_kda,
            "scission_rate_inv_kda_day": self.scission_rate_inv_kda_day,
            "acid_diffusivity_mm2_day": self.acid_diffusivity_mm2_day,
            "modulus0_mpa": self.modulus0_mpa,
            "strength0_mpa": self.strength0_mpa,
            "dissolution_width_kda": self.dissolution_width_kda,
            "activation_energy_kj_mol": self.activation_energy_kj_mol,
            "ph_rate_multiplier": self.ph_rate_multiplier,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")
        nonnegative = {
            "autocatalysis_per_acid": self.autocatalysis_per_acid,
            "acid_yield_kda": self.acid_yield_kda,
            "mass_transfer_mm_day": self.mass_transfer_mm_day,
            "bulk_clearance_per_day": self.bulk_clearance_per_day,
            "chemicrystallization_gain": self.chemicrystallization_gain,
            "chemicrystallization_rate_per_day": self.chemicrystallization_rate_per_day,
            "crystal_hydrolysis_per_day": self.crystal_hydrolysis_per_day,
            "dissolution_rate_per_day": self.dissolution_rate_per_day,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative; got {value}")
        if not 0 < self.crystallinity0 < 1:
            raise ValueError("crystallinity0 must lie strictly between zero and one")
        if self.crystallinity0 + self.chemicrystallization_gain >= 1:
            raise ValueError("initial crystallinity plus gain must be below one")
        if not 0 <= self.external_acid:
            raise ValueError("external_acid must be nonnegative")
        if not 0 < self.critical_entanglement_mn_kda < self.mn0_kda:
            raise ValueError("critical entanglement molecular weight must be in (0, Mn0)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationConfig:
    geometry: Geometry = Geometry.SLAB
    characteristic_length_mm: float = 1.0
    duration_days: float = 730.0
    output_interval_days: float = 5.0
    cells: int = 80
    relative_tolerance: float = 1e-6
    absolute_tolerance: float = 1e-9
    maximum_step_days: float = 2.0
    functional_strength_fraction: float = 0.50

    def validate(self) -> None:
        if self.characteristic_length_mm <= 0:
            raise ValueError("characteristic_length_mm must be positive")
        if self.duration_days <= 0 or self.output_interval_days <= 0:
            raise ValueError("simulation times must be positive")
        if self.cells < 4:
            raise ValueError("at least four finite-volume cells are required")
        if self.relative_tolerance <= 0 or self.absolute_tolerance <= 0:
            raise ValueError("solver tolerances must be positive")
        if self.maximum_step_days <= 0:
            raise ValueError("maximum_step_days must be positive")
        if not 0 < self.functional_strength_fraction < 1:
            raise ValueError("functional_strength_fraction must lie in (0, 1)")

    @property
    def surface_to_volume_per_mm(self) -> float:
        return self.geometry.surface_to_volume_factor / self.characteristic_length_mm

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["geometry"] = self.geometry.value
        return result

