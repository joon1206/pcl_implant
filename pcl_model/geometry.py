"""CAD audit utilities with explicit mesh-unit conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class MeshReport:
    path: str
    assumed_input_unit: str
    vertices: int
    faces: int
    watertight: bool
    winding_consistent: bool
    connected_components: int
    euler_number: int
    surface_area_mm2: float
    volume_mm3: float
    surface_to_volume_per_mm: float
    hydraulic_length_mm: float
    bounding_box_mm: tuple[float, float, float]
    minimum_bbox_dimension_mm: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}


def inspect_mesh(path: str | Path, input_unit: str = "mm") -> MeshReport:
    if input_unit not in UNIT_TO_MM:
        raise ValueError(f"input_unit must be one of {sorted(UNIT_TO_MM)}")
    # STL stores independent triangle vertices. Processing merges coincident
    # vertices so watertightness and connected-component checks are meaningful.
    mesh = trimesh.load_mesh(str(path), force="mesh", process=True)
    if mesh.is_empty:
        raise ValueError(f"mesh is empty: {path}")
    scale = UNIT_TO_MM[input_unit]
    area = float(mesh.area) * scale**2
    volume = float(mesh.volume) * scale**3
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("mesh volume is not positive; check closure and face orientation")
    extents = tuple(float(value * scale) for value in mesh.extents)
    components = len(mesh.split(only_watertight=False))
    return MeshReport(
        path=str(Path(path)),
        assumed_input_unit=input_unit,
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        connected_components=int(components),
        euler_number=int(mesh.euler_number),
        surface_area_mm2=area,
        volume_mm3=volume,
        surface_to_volume_per_mm=area / volume,
        hydraulic_length_mm=volume / area,
        bounding_box_mm=extents,
        minimum_bbox_dimension_mm=min(extents),
    )

