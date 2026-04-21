from __future__ import annotations

import argparse
import os
import struct
import xml.etree.ElementTree as ET
from base64 import b64encode
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from pcl_shrink_model.pcl_deg_model_voxel_unconstrained_best_fast import (
    AutocatDiffusionParams,
    CrystallinityParams,
    calibrate_k0,
    compute_export_fields,
    init_state,
    load_mesh_compute_sav,
    make_geometry_fields,
    step_degradation_inplace,
)


@dataclass
class RunParams:
    stl_path: str
    outdir: str = "output_best"
    mn0: float = 60000.0
    k0: float | None = None
    mn_target_ratio: float = 0.20
    t_degrade_yr: float = 3.0
    m: float = 1.0
    e0: float = 350.0
    alpha: float = 2.0
    k_phi: float = 0.040
    phi_max: float = 0.65
    beta_phi: float = 2.0
    xc0: float = 0.45
    xc_max: float = 0.70
    enable_crystallinity: bool = False
    acid_xc_coupling: float = 0.0
    Dw: float = 438.0
    Da: float = 91.25
    k_water_uptake: float = 730.0
    k_acid_out: float = 146.0
    k_auto: float = 4.5
    acid_order: float = 1.0
    acid_from_damage: float = 8.0
    acid_decay: float = 0.73
    water_reaction_sink: float = 2.5
    kd_surface_boost: float = 1.12
    kd_core_drop: float = 0.94
    curvature_boost: float = 0.0
    tmax: float = 8.0
    nt: int = 121
    pitch: float | None = None
    pad_voxels: int = 2
    mc_level: float = 0.35
    smooth_mesh_iters: int = 0
    min_component_voxels: int = 12
    preserve_largest_component: bool = False
    erosion_start_mn_ratio: float = 0.45
    max_voxel_remove_fraction_per_step: float = 0.02
    surface_bias: float = 0.90
    rng_seed: int = 1234
    save_every: int = 4
    export_stl: bool = False
    export_vtp: bool = True
    substeps: int = 1


def parse_args() -> RunParams:
    ap = argparse.ArgumentParser(description="Faster voxel shrink pipeline with autocatalysis + diffusion degradation.")
    ap.add_argument("stl_path", type=str)
    ap.add_argument("--outdir", type=str, default="output_best")
    ap.add_argument("--mn0", type=float, default=60000.0)
    ap.add_argument("--k0", type=float, default=None)
    ap.add_argument("--mn-target-ratio", type=float, default=0.20)
    ap.add_argument("--t-degrade-yr", type=float, default=3.0)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--e0", type=float, default=350.0)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--k-phi", type=float, default=0.040)
    ap.add_argument("--phi-max", type=float, default=0.65)
    ap.add_argument("--beta-phi", type=float, default=2.0)
    ap.add_argument("--xc0", type=float, default=0.45)
    ap.add_argument("--xc-max", type=float, default=0.70)
    ap.add_argument("--enable-crystallinity", action="store_true")
    ap.add_argument("--acid-xc-coupling", type=float, default=0.0)
    ap.add_argument("--Dw", type=float, default=438.0)
    ap.add_argument("--Da", type=float, default=91.25)
    ap.add_argument("--k-water-uptake", type=float, default=730.0)
    ap.add_argument("--k-acid-out", type=float, default=146.0)
    ap.add_argument("--k-auto", type=float, default=4.5)
    ap.add_argument("--acid-order", type=float, default=1.0)
    ap.add_argument("--acid-from-damage", type=float, default=8.0)
    ap.add_argument("--acid-decay", type=float, default=0.73)
    ap.add_argument("--water-reaction-sink", type=float, default=2.5)
    ap.add_argument("--kd-surface-boost", type=float, default=1.12)
    ap.add_argument("--kd-core-drop", type=float, default=0.94)
    ap.add_argument("--curvature-boost", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=8.0)
    ap.add_argument("--nt", type=int, default=121)
    ap.add_argument("--pitch", type=float, default=None)
    ap.add_argument("--pad-voxels", type=int, default=2)
    ap.add_argument("--mc-level", type=float, default=0.35)
    ap.add_argument("--smooth-mesh-iters", type=int, default=0)
    ap.add_argument("--min-component-voxels", type=int, default=12)
    ap.add_argument("--preserve-largest-component", action="store_true")
    ap.add_argument("--erosion-start-mn-ratio", type=float, default=0.45)
    ap.add_argument("--max-voxel-remove-fraction-per-step", type=float, default=0.02)
    ap.add_argument("--surface-bias", type=float, default=0.90)
    ap.add_argument("--rng-seed", type=int, default=1234)
    ap.add_argument("--save-every", type=int, default=4, help="Write mesh files every N steps.")
    ap.add_argument("--export-stl", action="store_true")
    ap.add_argument("--no-vtp", action="store_true")
    ap.add_argument("--substeps", type=int, default=1)
    ns = ap.parse_args()
    d = vars(ns)
    d["export_vtp"] = not d.pop("no_vtp")
    return RunParams(**d)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _b64f32(a: np.ndarray) -> str:
    raw = np.asarray(a, dtype=np.float32).tobytes()
    return b64encode(struct.pack("<I", len(raw)) + raw).decode()


def _b64i32(a: np.ndarray) -> str:
    raw = np.asarray(a, dtype=np.int32).tobytes()
    return b64encode(struct.pack("<I", len(raw)) + raw).decode()


def write_vtp(path: str, verts: np.ndarray, faces: np.ndarray, face_fields: dict[str, np.ndarray], time: float = 0.0) -> None:
    nv = len(verts)
    nf = len(faces)
    conn = faces.reshape(-1).astype(np.int32)
    offs = (np.arange(1, nf + 1) * 3).astype(np.int32)
    root = ET.Element("VTKFile", type="PolyData", version="0.1", byte_order="LittleEndian", header_type="UInt32", encoding="base64")
    pd = ET.SubElement(root, "PolyData")
    fd = ET.SubElement(pd, "FieldData")
    tv = ET.SubElement(fd, "DataArray", type="Float64", Name="TimeValue", NumberOfTuples="1", format="ascii")
    tv.text = f"{time:.6g}"
    piece = ET.SubElement(pd, "Piece", NumberOfPoints=str(nv), NumberOfVerts="0", NumberOfLines="0", NumberOfStrips="0", NumberOfPolys=str(nf))
    pts = ET.SubElement(piece, "Points")
    pa = ET.SubElement(pts, "DataArray", type="Float32", Name="Points", NumberOfComponents="3", format="binary")
    pa.text = _b64f32(verts)
    polys = ET.SubElement(piece, "Polys")
    ET.SubElement(polys, "DataArray", type="Int32", Name="connectivity", format="binary").text = _b64i32(conn)
    ET.SubElement(polys, "DataArray", type="Int32", Name="offsets", format="binary").text = _b64i32(offs)
    cd = ET.SubElement(piece, "CellData")
    for name, arr in face_fields.items():
        el = ET.SubElement(cd, "DataArray", type="Float32", Name=name, format="binary")
        el.text = _b64f32(np.asarray(arr, dtype=np.float32))
    ET.indent(root, space="  ")
    with open(path, "wb") as f:
        f.write(b'<?xml version="1.0"?>\n')
        ET.ElementTree(root).write(f, encoding="utf-8", xml_declaration=False)


def write_pvd(outdir: str, steps: list[tuple[float, str]]) -> None:
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    coll = ET.SubElement(root, "Collection")
    for t, fname in steps:
        ET.SubElement(coll, "DataSet", timestep=f"{t:.6g}", group="", part="0", file=fname)
    ET.indent(root, space="  ")
    out = os.path.join(outdir, "degradation.pvd")
    with open(out, "wb") as f:
        f.write(b'<?xml version="1.0"?>\n')
        ET.ElementTree(root).write(f, encoding="utf-8", xml_declaration=False)


def choose_pitch(mesh: trimesh.Trimesh, target_cells_long_axis: int = 72) -> float:
    extents = np.asarray(mesh.extents, dtype=float)
    long_axis = float(np.max(extents))
    return max(long_axis / max(target_cells_long_axis, 16), long_axis / 180.0, 1e-6)


def voxelize_mesh(mesh: trimesh.Trimesh, pitch: float, pad_voxels: int = 2) -> tuple[np.ndarray, np.ndarray]:
    vox = mesh.voxelized(pitch).fill()
    mat = np.asarray(vox.matrix, dtype=bool)
    if pad_voxels > 0:
        mat = np.pad(mat, int(pad_voxels), mode="constant", constant_values=False)
    origin = np.asarray(vox.translation, dtype=float) - int(pad_voxels) * float(pitch)
    return mat, origin


def keep_components(mask: np.ndarray, min_component_voxels: int, preserve_largest: bool) -> np.ndarray:
    labels, nlab = ndimage.label(mask)
    if nlab == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if preserve_largest:
        return labels == int(np.argmax(counts))
    keep = np.zeros_like(mask, dtype=bool)
    for lab in range(1, nlab + 1):
        if counts[lab] >= int(min_component_voxels):
            keep |= labels == lab
    return keep


def extract_surface_from_mask(mask: np.ndarray, origin: np.ndarray, pitch: float, level: float, smooth_iters: int, crop_margin: int = 1):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    lo = np.maximum(pts.min(axis=0) - crop_margin, 0)
    hi = np.minimum(pts.max(axis=0) + crop_margin + 1, np.array(mask.shape))
    cropped = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    vol = ndimage.gaussian_filter(cropped.astype(np.float32), sigma=0.5)
    if np.max(vol) <= 0:
        return None
    verts, faces, _normals, _values = measure.marching_cubes(vol, level=level, spacing=(pitch, pitch, pitch))
    verts = verts + (origin + lo.astype(np.float32) * np.float32(pitch))[np.newaxis, :]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if smooth_iters > 0:
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.20, nu=-0.21, iterations=smooth_iters)
        except Exception:
            pass
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh if len(mesh.vertices) >= 4 and len(mesh.faces) >= 4 else None


def sample_faces_from_grid(face_centers_world: np.ndarray, face_normals: np.ndarray, origin: np.ndarray, pitch: float, grid: np.ndarray, inward_offset_fraction: float = 0.35) -> np.ndarray:
    inward_points = face_centers_world - np.asarray(face_normals, dtype=np.float32) * (float(inward_offset_fraction) * float(pitch))
    idx = np.floor((inward_points - origin[np.newaxis, :]) / float(pitch)).astype(int)
    idx = np.clip(idx, 0, np.array(grid.shape) - 1)
    vals = grid[idx[:, 0], idx[:, 1], idx[:, 2]]

    # Fallback for any boundary faces that still land in exterior cells: sample a tiny local neighborhood
    bad = vals <= 0
    if np.any(bad):
        neigh_max = vals.copy()
        shape = np.array(grid.shape, dtype=int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nidx = idx.copy()
                    nidx[:, 0] = np.clip(nidx[:, 0] + dx, 0, shape[0] - 1)
                    nidx[:, 1] = np.clip(nidx[:, 1] + dy, 0, shape[1] - 1)
                    nidx[:, 2] = np.clip(nidx[:, 2] + dz, 0, shape[2] - 1)
                    neigh_vals = grid[nidx[:, 0], nidx[:, 1], nidx[:, 2]]
                    neigh_max = np.maximum(neigh_max, neigh_vals)
        vals = neigh_max
    return vals


def evolve_mask_continuous(current_mask: np.ndarray, mn_ratio: np.ndarray, local_mult: np.ndarray, surface_exposure: np.ndarray, mn_remove_thresh: float, max_remove_fraction: float, rng: np.random.Generator) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    surface = current_mask & (~ndimage.binary_erosion(current_mask, structure))
    eligible = current_mask & (mn_ratio <= mn_remove_thresh)
    if not np.any(eligible):
        return current_mask
    progress = np.zeros_like(mn_ratio, dtype=np.float32)
    progress[eligible] = (mn_remove_thresh - mn_ratio[eligible]) / max(mn_remove_thresh, 1e-8)
    speed = np.power(np.clip(progress, 0.0, 1.0), 1.25)
    p_remove = np.clip((0.15 + 0.85 * speed) * (0.30 + 0.70 * surface.astype(np.float32)) * (0.30 + 0.70 * surface_exposure.astype(np.float32)) * np.clip(local_mult.astype(np.float32), 0.25, 2.5), 0.0, max_remove_fraction)
    remove_mask = (rng.random(current_mask.shape, dtype=np.float32) < p_remove) & eligible
    new_mask = current_mask & (~remove_mask)
    return new_mask


def build_model_params(p: RunParams, k0: float) -> AutocatDiffusionParams:
    return AutocatDiffusionParams(
        mn0=p.mn0, e0=p.e0, alpha=p.alpha, beta_phi=p.beta_phi, k0=k0, m=p.m,
        Dw=p.Dw, Da=p.Da, k_water_uptake=p.k_water_uptake, k_acid_out=p.k_acid_out,
        k_auto=p.k_auto, acid_order=p.acid_order, acid_from_damage=p.acid_from_damage,
        acid_decay=p.acid_decay, water_reaction_sink=p.water_reaction_sink,
        k_phi=p.k_phi, phi_max=p.phi_max, kd_surface_boost=p.kd_surface_boost,
        kd_core_drop=p.kd_core_drop, curvature_boost=p.curvature_boost,
        enable_crystallinity=p.enable_crystallinity, acid_xc_coupling=p.acid_xc_coupling,
        cryst_params=CrystallinityParams(xc0=p.xc0, xc_max=p.xc_max),
    )


def maybe_write_step(outdir: str, step_i: int, t: float, mesh, face_fields: dict[str, np.ndarray], export_vtp: bool, export_stl: bool) -> list[tuple[float, str]]:
    written = []
    if export_stl:
        mesh.export(os.path.join(outdir, f"step_{step_i:03d}.stl"))
    if export_vtp:
        fname = f"step_{step_i:03d}.vtp"
        write_vtp(os.path.join(outdir, fname), np.asarray(mesh.vertices), np.asarray(mesh.faces), face_fields, time=float(t))
        written.append((float(t), fname))
    return written


def main() -> None:
    p = parse_args()
    ensure_dir(p.outdir)
    for name in os.listdir(p.outdir):
        if name.startswith("step_") and (name.endswith(".vtp") or name.endswith(".stl")):
            try:
                os.remove(os.path.join(p.outdir, name))
            except Exception:
                pass
    for name in ("degradation.pvd", "summary.csv"):
        path = os.path.join(p.outdir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    surf_mesh = trimesh.load(p.stl_path, force="mesh")
    if isinstance(surf_mesh, trimesh.Scene):
        surf_mesh = trimesh.util.concatenate([g for g in surf_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    surf_mesh.remove_unreferenced_vertices()
    surf_mesh.fix_normals()

    sa0, vol0, sav0 = load_mesh_compute_sav(p.stl_path)
    print(f"[mesh] SA={sa0:.6g}  V={vol0:.6g}  SA/V={sav0:.6g}")
    pitch = p.pitch if p.pitch is not None else choose_pitch(surf_mesh)
    mask0, origin = voxelize_mesh(surf_mesh, pitch=pitch, pad_voxels=p.pad_voxels)
    mask0 = keep_components(mask0, p.min_component_voxels, p.preserve_largest_component)
    initial_voxels = int(np.count_nonzero(mask0))
    print(f"[voxel] pitch={pitch:.6g}  grid={mask0.shape}  solid_voxels={initial_voxels}")

    k0 = calibrate_k0(p.mn0, p.mn_target_ratio, p.t_degrade_yr, sav0, p.m) if p.k0 is None else float(p.k0)
    print(f"[kin] k0={k0:.6g}  kd_base={k0 * (sav0 ** p.m):.6g}  tmax={p.tmax:.3f} yr")

    model_params = build_model_params(p, k0)
    geom = make_geometry_fields(mask0, p.kd_surface_boost, p.kd_core_drop, p.curvature_boost)
    state = init_state(mask0, model_params)
    times = np.linspace(0.0, p.tmax, p.nt)
    dt = float(times[1] - times[0]) if len(times) > 1 else float(p.tmax)
    current_mask = mask0.copy()
    rng = np.random.default_rng(p.rng_seed)
    pvd_steps: list[tuple[float, str]] = []

    summary_path = os.path.join(p.outdir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8") as csv:
        csv.write("step,t_yr,Mn_ratio_mean,Mn_ratio_min,Mn_ratio_max,E_rel_mean,mass_frac_real,risk_mean,voxels,components\n")
        current_time = 0.0
        for i, t in enumerate(times):
            if i > 0:
                step_degradation_inplace(state, current_mask, geom, model_params, dt, sav0, current_time, substeps=p.substeps)
                current_time = float(t)
                export = compute_export_fields(state, current_mask, model_params)
                current_mask = evolve_mask_continuous(current_mask, export.mn_ratio, geom.local_mult, geom.surface_exposure, p.erosion_start_mn_ratio, p.max_voxel_remove_fraction_per_step, rng)
                current_mask = keep_components(current_mask, p.min_component_voxels, p.preserve_largest_component)
                geom = make_geometry_fields(current_mask, p.kd_surface_boost, p.kd_core_drop, p.curvature_boost)
            export = compute_export_fields(state, current_mask, model_params)
            solid_idx = np.flatnonzero(current_mask.ravel())
            if solid_idx.size == 0:
                print(f"[stop] no solid material left at step {i:03d}, t={t:.3f} yr")
                break
            mr_live = export.mn_ratio.ravel()[solid_idx]
            e_live = export.e_rel.ravel()[solid_idx]
            risk_live = export.risk.ravel()[solid_idx]
            real_mass_frac = float(np.count_nonzero(current_mask)) / max(initial_voxels, 1)
            _labels, ncomp = ndimage.label(current_mask)
            print(f"step {i:03d}  t={t:.3f} yr  Mn/Mn0 mean={np.mean(mr_live):.3f} [{np.min(mr_live):.3f}-{np.max(mr_live):.3f}]  real_mf={real_mass_frac:.3f}  voxels={np.count_nonzero(current_mask)} comps={ncomp}")
            csv.write(f"{i},{t:.6f},{np.mean(mr_live):.6f},{np.min(mr_live):.6f},{np.max(mr_live):.6f},{np.mean(e_live):.6f},{real_mass_frac:.6f},{np.mean(risk_live):.6f},{np.count_nonzero(current_mask)},{ncomp}\n")

            should_write = (i == 0) or (i == len(times) - 1) or (p.save_every > 0 and i % p.save_every == 0)
            if should_write and (p.export_vtp or p.export_stl):
                mesh = extract_surface_from_mask(current_mask, origin, pitch, p.mc_level, p.smooth_mesh_iters)
                if mesh is not None:
                    face_centers = np.asarray(mesh.triangles_center, dtype=np.float32)
                    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
                    face_fields = {
                        "Mn": sample_faces_from_grid(face_centers, face_normals, origin, pitch, export.mn).astype(np.float32),
                        "Mn_ratio": sample_faces_from_grid(face_centers, face_normals, origin, pitch, export.mn_ratio).astype(np.float32),
                        "E_rel": sample_faces_from_grid(face_centers, face_normals, origin, pitch, export.e_rel).astype(np.float32),
                        "risk": sample_faces_from_grid(face_centers, face_normals, origin, pitch, export.risk).astype(np.float32),
                    }
                    pvd_steps.extend(maybe_write_step(p.outdir, i, float(t), mesh, face_fields, p.export_vtp, p.export_stl))

    if p.export_vtp and pvd_steps:
        write_pvd(p.outdir, pvd_steps)
    print(f"[done] wrote {len(pvd_steps)} surface steps to {os.path.abspath(p.outdir)}")


if __name__ == "__main__":
    main()
