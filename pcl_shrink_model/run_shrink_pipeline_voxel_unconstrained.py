from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
from base64 import b64encode
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from pcl_deg_model_voxel_unconstrained import (
    CrystallinityParams,
    TwoPhaseParams,
    _t_when_mn_hits_threshold,
    calibrate_k0,
    kd_from_sav,
    load_mesh_compute_sav,
    mass_fraction_two_phase,
    mn_two_phase,
    porosity_from_mn,
    crystallinity_bulk_pcl,
)


@dataclass
class RunParams:
    stl_path: str
    outdir: str = "output"
    mn0: float = 60000.0
    k0: float | None = None
    mn_target_ratio: float = 0.20
    t_degrade_yr: float = 3.0
    m: float = 1.0

    mn_crit_ratio: float = 0.20
    accel_factor: float = 1.08
    k_erosion: float = 0.10
    transition_width: float = 1.40
    phase2_max_fraction: float = 0.18
    erosion_onset_ratio: float = 0.12
    erosion_power: float = 2.2
    min_mass_fraction: float = 0.0

    kd_surface_boost: float = 1.10
    kd_core_drop: float = 0.95
    curvature_boost: float = 0.04

    e0: float = 350.0
    alpha: float = 2.0
    k_phi: float = 0.040
    phi_max: float = 0.65
    beta_phi: float = 2.0
    xc0: float = 0.45
    xc_max: float = 0.70

    tmax: float = 6.0
    nt: int = 121

    pitch: float | None = None
    pad_voxels: int = 6
    mc_level: float = 0.35
    smooth_mesh_iters: int = 1
    min_component_voxels: int = 12
    preserve_largest_component: bool = False

    damage_threshold: float = 1.6
    shell_sigma: float = 0.75
    shell_weight: float = 0.8
    max_voxel_removal_fraction_per_step: float = 0.01


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _b64f32(a: np.ndarray) -> str:
    raw = np.asarray(a, dtype=np.float32).tobytes()
    return b64encode(struct.pack("<I", len(raw)) + raw).decode()


def _b64i32(a: np.ndarray) -> str:
    raw = np.asarray(a, dtype=np.int32).tobytes()
    return b64encode(struct.pack("<I", len(raw)) + raw).decode()


def write_vtp(path: str, verts: np.ndarray, faces: np.ndarray, face_fields: dict, time: float = 0.0) -> None:
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
        a = np.asarray(arr, dtype=np.float32)
        if a.size != nf:
            a = np.full(nf, float(np.nanmean(a)), dtype=np.float32)
        el = ET.SubElement(cd, "DataArray", type="Float32", Name=name, format="binary")
        el.text = _b64f32(a)
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


def voxelize_mesh(mesh: trimesh.Trimesh, pitch: float, pad_voxels: int = 6):
    vox = mesh.voxelized(pitch).fill()
    mat = np.asarray(vox.matrix, dtype=bool)
    if pad_voxels > 0:
        mat = np.pad(mat, int(pad_voxels), mode="constant", constant_values=False)
    origin = np.asarray(vox.translation, dtype=float) - int(pad_voxels) * float(pitch)
    return mat, origin


def local_rate_multiplier(mask: np.ndarray, p: RunParams) -> tuple[np.ndarray, np.ndarray]:
    inside_dist = ndimage.distance_transform_edt(mask)
    maxd = float(np.max(inside_dist))
    core_norm = inside_dist / max(maxd, 1e-12)
    surface_exposure = 1.0 - core_norm
    smooth = ndimage.gaussian_filter(mask.astype(np.float32), sigma=1.0)
    gx, gy, gz = np.gradient(smooth)
    curv = np.sqrt(gx * gx + gy * gy + gz * gz)
    if np.max(curv) > 0:
        curv = curv / np.max(curv)
    mult = float(p.kd_core_drop) + (float(p.kd_surface_boost) - float(p.kd_core_drop)) * surface_exposure + float(p.curvature_boost) * curv
    mult = np.clip(mult, 0.88, 1.18).astype(np.float32)
    return mult, curv.astype(np.float32)


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


def extract_surface_from_mask(mask: np.ndarray, origin: np.ndarray, pitch: float, level: float, smooth_iters: int):
    vol = ndimage.gaussian_filter(mask.astype(np.float32), sigma=0.6)
    if np.max(vol) <= 0:
        return None
    verts, faces, _normals, _values = measure.marching_cubes(vol, level=level, spacing=(pitch, pitch, pitch))
    verts = verts + origin[np.newaxis, :]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    try:
        mesh.merge_vertices(digits_vertex=6)
    except TypeError:
        mesh.merge_vertices()
    if smooth_iters > 0:
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.20, nu=-0.21, iterations=smooth_iters)
        except Exception:
            pass
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        return None
    return mesh


def sample_faces_from_grid(face_centers_world: np.ndarray, origin: np.ndarray, pitch: float, grid: np.ndarray) -> np.ndarray:
    idx = np.floor((face_centers_world - origin[np.newaxis, :]) / float(pitch)).astype(int)
    idx = np.clip(idx, 0, np.array(grid.shape) - 1)
    return grid[idx[:, 0], idx[:, 1], idx[:, 2]]


def grid_fields_at_t(t_scalar: float, local_mult: np.ndarray, curv_grid: np.ndarray, p: RunParams,
                     base_k0: float, global_sav: float, two_phase: TwoPhaseParams,
                     cryst_params: CrystallinityParams):
    kd_base = float(kd_from_sav(base_k0, global_sav, p.m))
    kd1_local = kd_base * local_mult
    kd2_local = kd1_local * max(float(two_phase.accel_factor), 1.0)
    tcrit_local = np.full_like(kd1_local, np.inf, dtype=np.float32)
    finite = kd1_local > 1e-12
    tcrit_local[finite] = np.log(float(p.mn0) / float(two_phase.mn_crit)) / kd1_local[finite]

    t_hist = np.linspace(0.0, max(float(t_scalar), 1e-12), 240, dtype=np.float32)
    if t_hist.size == 1:
        integral = kd1_local * float(t_scalar)
    else:
        width = max(float(two_phase.transition_width), 1e-12)
        u_hist = (t_hist[:, None, None, None] - (tcrit_local[None, ...] - 0.5 * width)) / width
        z_hist = np.clip(4.5 * (u_hist - 0.5), -60.0, 60.0)
        w_hist = float(two_phase.phase2_max_fraction) * (1.0 / (1.0 + np.exp(-z_hist)))
        w_hist = np.where(np.isfinite(tcrit_local[None, ...]), w_hist, 0.0).astype(np.float32)
        kd_hist = (1.0 - w_hist) * kd1_local[None, ...] + w_hist * kd2_local[None, ...]
        dt = np.diff(t_hist).astype(np.float32)
        integral = np.sum(0.5 * (kd_hist[:-1] + kd_hist[1:]) * dt[:, None, None, None], axis=0)

    mn_ratio = np.clip(np.exp(-integral), 1e-6, 1.0).astype(np.float32)
    mn = (float(p.mn0) * mn_ratio).astype(np.float32)
    phi = porosity_from_mn(mn, p.mn0, k_phi=p.k_phi, phi_max=p.phi_max).astype(np.float32)
    e_rel = (np.maximum(mn_ratio, 1e-12) ** float(p.alpha)) * (np.maximum(1.0 - phi, 1e-8) ** float(p.beta_phi))
    deg = (1.0 - mn_ratio).astype(np.float32)
    risk = (deg * (1.0 + 1.5 * deg)).astype(np.float32)

    xc = np.zeros_like(mn_ratio, dtype=np.float32)
    flat_mn = mn.reshape(-1)
    flat_xc = xc.reshape(-1)
    t_hist2 = np.array([0.0, float(t_scalar)], dtype=float)
    for i in range(flat_mn.size):
        flat_xc[i] = float(crystallinity_bulk_pcl(t_hist2, np.array([float(p.mn0), float(flat_mn[i])]), cryst_params)[-1])
    xc = flat_xc.reshape(mn_ratio.shape)

    global_curve = mn_two_phase(np.linspace(0.0, max(float(t_scalar), 1e-12), 240), p.mn0, base_k0, global_sav, p.m, two_phase)
    global_mn_ratio = np.float32(global_curve[-1] / float(p.mn0))
    return {
        "Mn": mn,
        "Mn_ratio": mn_ratio,
        "Mn_ratio_global": np.full_like(mn_ratio, global_mn_ratio, dtype=np.float32),
        "porosity": phi,
        "E_rel": e_rel.astype(np.float32),
        "risk": risk,
        "crystallinity": xc,
        "local_kd_multiplier": local_mult.astype(np.float32),
        "curvature_proxy": curv_grid.astype(np.float32),
    }


def evolve_mask_to_target(current_mask, damage, target_keep_voxels, p):
    current_keep = int(np.count_nonzero(current_mask))
    if current_keep <= target_keep_voxels:
        return current_mask.copy()

    candidate_remove = current_mask & (damage >= float(p.damage_threshold))
    if not np.any(candidate_remove):
        return current_mask.copy()

    remove_budget = current_keep - int(target_keep_voxels)
    max_remove = int(np.ceil(float(p.max_voxel_removal_fraction_per_step) * current_keep))
    remove_budget = min(remove_budget, max_remove, int(np.count_nonzero(candidate_remove)))
    if remove_budget <= 0:
        return current_mask.copy()

    cand_idx = np.flatnonzero(candidate_remove.ravel())
    cand_damage = damage.ravel()[cand_idx]
    top = np.argpartition(cand_damage, -remove_budget)[-remove_budget:]

    new_mask = current_mask.ravel().copy()
    new_mask[cand_idx[top]] = False
    new_mask = new_mask.reshape(current_mask.shape)

    structure = ndimage.generate_binary_structure(3, 1)
    new_mask = ndimage.binary_opening(new_mask, structure=structure)
    new_mask = ndimage.binary_closing(new_mask, structure=structure)
    new_mask = keep_components(new_mask, p.min_component_voxels, p.preserve_largest_component)
    return new_mask

def update_damage(damage, alive_mask, mn_ratio_grid, shell_pref0, dt, p, two_phase):
    onset = max(float(two_phase.erosion_onset_ratio), 1e-6)
    onset_prog = np.clip((onset - mn_ratio_grid) / onset, 0.0, 1.0)

    live_surface = alive_mask & (~ndimage.binary_erosion(
        alive_mask,
        structure=ndimage.generate_binary_structure(3, 1),
        border_value=0
    ))

    shell_factor = 1.0 + float(p.shell_weight) * shell_pref0
    shell_factor = np.where(live_surface, shell_factor, 0.35 * shell_factor)

    damage_rate = float(two_phase.k_erosion) * np.power(
        onset_prog,
        float(two_phase.erosion_power)
    ) * shell_factor

    new_damage = damage.copy()
    new_damage[alive_mask] += float(dt) * damage_rate[alive_mask]
    return new_damage


def main() -> None:
    p = RunParams(stl_path="input/newpart.stl")
    ensure_dir(p.outdir)
    for name in os.listdir(p.outdir):
        if name.startswith("step_") and (name.endswith(".vtp") or name.endswith(".stl")):
            try: os.remove(os.path.join(p.outdir, name))
            except Exception: pass
    for name in ("degradation.pvd", "summary.csv"):
        path = os.path.join(p.outdir, name)
        if os.path.exists(path):
            try: os.remove(path)
            except Exception: pass

    surf_mesh = trimesh.load(p.stl_path, force="mesh")
    if isinstance(surf_mesh, trimesh.Scene):
        surf_mesh = trimesh.util.concatenate([g for g in surf_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    surf_mesh.remove_unreferenced_vertices()
    surf_mesh.fix_normals()

    sa0, vol0, sav0 = load_mesh_compute_sav(p.stl_path)
    print(f"[mesh] SA={sa0:.6g}  V={vol0:.6g}  SA/V={sav0:.6g}")
    pitch = p.pitch if p.pitch is not None else choose_pitch(surf_mesh)
    mask0, origin = voxelize_mesh(surf_mesh, pitch=pitch, pad_voxels=p.pad_voxels)
    initial_voxels = int(np.count_nonzero(mask0))
    print(f"[voxel] pitch={pitch:.6g}  grid={mask0.shape}  solid_voxels={initial_voxels}")

    mn_crit = p.mn_crit_ratio * p.mn0
    two_phase = TwoPhaseParams(
        mn_crit=mn_crit,
        accel_factor=p.accel_factor,
        k_erosion=p.k_erosion,
        floor_mass_fraction=p.min_mass_fraction,
        transition_width=p.transition_width,
        phase2_max_fraction=p.phase2_max_fraction,
        erosion_onset_ratio=p.erosion_onset_ratio,
        erosion_power=p.erosion_power,
    )
    cryst_params = CrystallinityParams(xc0=p.xc0, xc_max=p.xc_max)
    if p.k0 is None:
        k0 = calibrate_k0(p.mn0, p.mn_target_ratio, p.t_degrade_yr, sav0, p.m, two_phase, p.mn_crit_ratio)
    else:
        k0 = float(p.k0)
    kd0 = float(kd_from_sav(k0, sav0, p.m))
    tcrit = _t_when_mn_hits_threshold(p.mn0, kd0, mn_crit)
    print(f"[kin] k0={k0:.6g}  kd={kd0:.6g}  tcrit={tcrit:.4f} yr  tmax={p.tmax:.3f} yr")
    t_check = np.linspace(0.0, max(p.tmax, p.t_degrade_yr), 300)
    mn_check = mn_two_phase(t_check, p.mn0, k0, sav0, p.m, two_phase)
    mf_check = mass_fraction_two_phase(t_check, p.mn0, k0, sav0, p.m, two_phase)
    idx_3yr = int(np.argmin(np.abs(t_check - p.t_degrade_yr)))
    print(f"[check] Mn/Mn0 at {p.t_degrade_yr:.2f} yr = {mn_check[idx_3yr] / p.mn0:.4f} ; mass_fraction at {p.tmax:.2f} yr = {mf_check[-1]:.4f}")

    times = np.linspace(0.0, p.tmax, p.nt)
    current_mask = mask0.copy()
    pvd_steps: list[tuple[float, str]] = []
    summary_path = os.path.join(p.outdir, "summary.csv")

    damage = np.zeros(mask0.shape, dtype=np.float32)
    inside_dist0 = ndimage.distance_transform_edt(mask0)
    maxd0 = max(float(np.max(inside_dist0)), 1e-12)
    shell_pref0 = 1.0 - inside_dist0 / maxd0
    shell_pref0 = ndimage.gaussian_filter(shell_pref0.astype(np.float32), sigma=p.shell_sigma)
    shell_pref0 = np.clip(shell_pref0, 0.0, 1.0)

    local_mult0, curv_grid0 = local_rate_multiplier(mask0, p)
    prev_t = 0.0

    with open(summary_path, "w", encoding="utf-8") as csv:
        csv.write("step,t_yr,Mn_ratio_mean,Mn_ratio_min,Mn_ratio_max,E_rel_mean,mass_frac_target,mass_frac_real,porosity_mean,risk_mean,voxels,components\n")
        for i, t in enumerate(times):
            dt = float(t - prev_t) if i > 0 else 0.0

            # Chemistry fields from the fixed initial heterogeneity field
            grid_fields = grid_fields_at_t(
                float(t),
                local_mult0,
                curv_grid0,
                p,
                k0,
                sav0,
                two_phase,
                cryst_params,
            )

            t_grid = np.linspace(0.0, max(float(t), 1e-9), 240)
            global_mass_frac_target = float(
                mass_fraction_two_phase(t_grid, p.mn0, k0, sav0, p.m, two_phase)[-1]
            )
            global_mass_frac_target = max(global_mass_frac_target, p.min_mass_fraction)
            target_keep_voxels = int(round(global_mass_frac_target * initial_voxels))

            if i > 0:
                damage = update_damage(
                    damage,
                    current_mask,
                    grid_fields["Mn_ratio"],
                    shell_pref0,
                    dt,
                    p,
                    two_phase,
                )
                current_mask = evolve_mask_to_target(
                    current_mask,
                    damage,
                    target_keep_voxels,
                    p,
                )

            mesh = extract_surface_from_mask(current_mask, origin, pitch, p.mc_level, p.smooth_mesh_iters)
            if mesh is None:
                print(f"[stop] no solid material left at step {i:03d}, t={t:.3f} yr")
                break

            face_centers = np.asarray(mesh.triangles_center, dtype=np.float32)
            face_fields = {
                key: sample_faces_from_grid(face_centers, origin, pitch, np.asarray(arr, dtype=np.float32)).astype(np.float32)
                for key, arr in grid_fields.items()
            }
            face_fields["damage"] = sample_faces_from_grid(face_centers, origin, pitch, damage.astype(np.float32)).astype(np.float32)

            stl_out = os.path.join(p.outdir, f"step_{i:03d}.stl")
            vtp_out = os.path.join(p.outdir, f"step_{i:03d}.vtp")
            mesh.export(stl_out)
            write_vtp(vtp_out, np.asarray(mesh.vertices), np.asarray(mesh.faces), face_fields, time=float(t))
            pvd_steps.append((float(t), os.path.basename(vtp_out)))

            solid_idx = np.flatnonzero(current_mask.ravel())
            mr_live = grid_fields["Mn_ratio"].ravel()[solid_idx]
            e_live = grid_fields["E_rel"].ravel()[solid_idx]
            phi_live = grid_fields["porosity"].ravel()[solid_idx]
            risk_live = grid_fields["risk"].ravel()[solid_idx]
            real_mass_frac = float(np.count_nonzero(current_mask)) / max(initial_voxels, 1)
            _labels, ncomp = ndimage.label(current_mask)

            print(
                f"step {i:03d}  t={t:.3f} yr  Mn/Mn0 mean={np.mean(mr_live):.3f} "
                f"[{np.min(mr_live):.3f}-{np.max(mr_live):.3f}]  "
                f"target_mf={global_mass_frac_target:.3f} real_mf={real_mass_frac:.3f}  "
                f"voxels={np.count_nonzero(current_mask)} comps={ncomp}"
            )

            csv.write(
                f"{i},{t:.6f},{np.mean(mr_live):.6f},{np.min(mr_live):.6f},{np.max(mr_live):.6f},"
                f"{np.mean(e_live):.6f},{global_mass_frac_target:.6f},{real_mass_frac:.6f},"
                f"{np.mean(phi_live):.6f},{np.mean(risk_live):.6f},{np.count_nonzero(current_mask)},{ncomp}\n"
            )

            prev_t = float(t)
    write_pvd(p.outdir, pvd_steps)
    print(f"[done] wrote {len(pvd_steps)} steps to {os.path.abspath(p.outdir)}")


if __name__ == "__main__":
    main()
