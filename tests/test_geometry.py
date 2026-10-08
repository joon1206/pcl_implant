import numpy as np
import trimesh

from pcl_model.geometry import inspect_mesh


def test_mesh_units_and_sav(tmp_path) -> None:
    path = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[2.0, 2.0, 2.0]).export(path)
    report = inspect_mesh(path, "mm")
    assert report.watertight
    assert np.isclose(report.surface_area_mm2, 24.0)
    assert np.isclose(report.volume_mm3, 8.0)
    assert np.isclose(report.surface_to_volume_per_mm, 3.0)

