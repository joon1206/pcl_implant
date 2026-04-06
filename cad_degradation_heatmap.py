
#!/usr/bin/env python3
"""
cad_degradation_heatmap.py

Prototype pipeline for overlaying degradation kinetics onto a CAD-derived surface mesh.

What it does
------------
1. Loads a surface mesh (STL/OBJ/PLY/GLB; STEP is not supported directly here, my bad :< ).
2. Voxelizes the geometry into a regular 3D grid.
3. Solves a simplified reaction-diffusion degradation model inside the voxelized device:
      - water ingress from exposed surfaces
      - retained acid / oligomer diffusion
      - autocatalytic molecular-weight decay
      - porosity growth with deg
      - modulus decay from Mn and porosity
4. Exports:
      - CSV of voxel centers and fields
      - NPZ arrays
      - colored PLY point clouds for Mn, porosity, modulus, and risk (should be good as graphics, imo)
      - slice plots as PNG
5. Highlights weak points using a simple risk proxy:
      risk = stress_proxy / local_strength_proxy (this is not a verified metric, arbitrarily set to show relatively dangerous areas)

Important notes
---------------
- This is a research prototype, not a validated medical-device simulator!! Maybe we'll get there. 
- For real CAD + coupled FEA, this can definitely be used as a preprocessor / screening tool. Preliminary only. 
- This script currently expects a closed watertight surface mesh. If you have STEP,
  convert it to STL first using CAD software. The files we're using are safe. 
- The structural part here is a simple geometric stress proxy, not full FEA.
  It is intentionally lightweight and dependency-minimal.

Dependencies
------------
numpy, scipy, matplotlib, trimesh

Example
-------
python cad_degradation_heatmap.py device.stl --voxel-pitch 0.4 --days 180 --dt 0.05 --outdir out_device

Outputs
-------
outdir/
  fields_final.csv
  fields_final.npz
  Mn_rel_final.ply
  porosity_final.ply
  modulus_final.ply
  risk_final.ply
  slice_Mn_rel.png
  slice_porosity.png
  slice_modulus.png
  slice_risk.png
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy import ndimage


@dataclass
class Params:
    voxel_pitch: float = 0.4            # mm
    days: float = 180.0
    dt: float = 0.05                    # days; explicit time step; if too small, it takes REALLY long I tried dt = 1.1574 e-5 because I got excited about sec-resolution
    save_every_days: float = 30.0

    # Kinetics / transport
    """
    Important info on kinetics / transport
    --------------------------------------
    Technically this has to be fitted from experimental parameters, but we can get away with extracting from lit data for now, I think.
    Any additional comments / concerns, added here.

    """
    
    Dw: float = 1.2                     # water diffusivity [voxel^2/day] after nondim scaling
    Da: float = 0.25                    # acid diffusivity [voxel^2/day]
    k_water_uptake: float = 2.0         # boundary exchange factor for water
    k_acid_out: float = 0.40            # boundary escape factor for acid

    k_hydro: float = 1.5e-3             # baseline hydrolysis [1/day]
    k_auto: float = 6.0e-3              # autocatalytic hydrolysis [1/day / conc]
    acid_order: float = 1.0

    acid_from_damage: float = 0.015     # acid generation from hydrolysis
    acid_decay: float = 0.002           # washout / neutralization term

    k_phi: float = 0.06                 # porosity growth
    phi_max: float = 0.85

    # Mechanics coupling
    E0_MPa: float = 300.0               # baseline young's modulus, from control sample OR from lit
    alpha_E_Mn: float = 1.15            # LIT / EXPERIMENTAL
    beta_E_phi: float = 2.5

    # Weak-point proxy
    stress_sensitivity_curvature: float = 1.0
    stress_sensitivity_thinness: float = 1.0

    # Numerical / safety
    smooth_curvature_sigma: float = 1.0
    minimum_Mn_rel: float = 1e-3
    minimum_strength_factor: float = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay simplified degradation kinetics onto a CAD mesh.")
    p.add_argument("mesh", type=str, help="Input mesh path (.stl, .obj, .ply, .glb, ...)")
    p.add_argument("--outdir", type=str, default="cad_deg_out", help="Output directory")
    p.add_argument("--voxel-pitch", type=float, default=0.4, help="Voxel size in mm")
    p.add_argument("--days", type=float, default=180.0, help="Simulation duration in days")
    p.add_argument("--dt", type=float, default=0.05, help="Time step in days")
    p.add_argument("--save-every-days", type=float, default=30.0, help="Save interval for snapshots")
    return p.parse_args()
# cad_degradation_heatmap.py device.stl --voxel-pitch 0.4 --days 180 --dt 0.05 --outdir out_device


def load_and_voxelize(mesh_path: Path, pitch: float):
    mesh = trimesh.load_mesh(mesh_path, process=True)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Loaded asset is not a triangle mesh.")

    if not mesh.is_watertight:
        print("Warning: mesh is not watertight. Filled voxelization may be imperfect.")

    vox = mesh.voxelized(pitch)
    vox_filled = vox.fill()

    occ = vox_filled.matrix.astype(bool)
    transform = vox_filled.transform.copy()

    # World-space centers of occupied voxels
    pts = vox_filled.points.copy()

    return mesh, vox_filled, occ, transform, pts


def build_boundary_masks(occ: np.ndarray):
    """
    Returns:
        interior_mask
        boundary_mask              : occupied voxels with at least one exterior neighbor
    """
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    eroded = ndimage.binary_erosion(occ, structure=struct, border_value=0)
    boundary = occ & (~eroded)
    interior = occ & eroded
    return interior, boundary


def laplacian_masked(field: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """
    6-neighbor masked Laplacian on occupied voxels.
    Exterior values are treated as zero.
    """
    out = np.zeros_like(field, dtype=np.float64)

    # axis 0
    out[1:, :, :] += field[:-1, :, :] * occ[:-1, :, :]
    out[:-1, :, :] += field[1:, :, :] * occ[1:, :, :]
    # axis 1
    out[:, 1:, :] += field[:, :-1, :] * occ[:, :-1, :]
    out[:, :-1, :] += field[:, 1:, :] * occ[:, 1:, :]
    # axis 2
    out[:, :, 1:] += field[:, :, :-1] * occ[:, :, :-1]
    out[:, :, :-1] += field[:, :, 1:] * occ[:, :, 1:]

    # number of occupied neighbors
    n = np.zeros_like(field, dtype=np.float64)
    n[1:, :, :] += occ[:-1, :, :]
    n[:-1, :, :] += occ[1:, :, :]
    n[:, 1:, :] += occ[:, :-1, :]
    n[:, :-1, :] += occ[:, 1:, :]
    n[:, :, 1:] += occ[:, :, :-1]
    n[:, :, :-1] += occ[:, :, 1:]

    out = out - n * field
    out[~occ] = 0.0
    return out


def nearest_surface_distance(boundary_mask: np.ndarray, occ: np.ndarray, voxel_pitch: float) -> np.ndarray:
    """
    Euclidean distance from each occupied voxel to the nearest boundary voxel, in mm.
    """
    # distance_transform_edt computes distance to nearest zero;
    # so invert boundary and sample boundary distance within occ?
    inv_boundary = ~boundary_mask
    dist_vox = ndimage.distance_transform_edt(inv_boundary)
    dist_mm = dist_vox * voxel_pitch
    dist_mm[~occ] = np.nan
    return dist_mm


def estimate_curvature_proxy(boundary_mask: np.ndarray, occ: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Very rough geometry-based curvature proxy from smoothed boundary occupancy.
    Higher near edges/corners / abrupt geometric changes.

    Honestly, this could use some work. I used a stress concentration analogue (like FEAs do) but there is next to no guarantee that this is right. 
    """
    b = boundary_mask.astype(np.float64)
    smooth = ndimage.gaussian_filter(b, sigma=sigma)

    gx = ndimage.sobel(smooth, axis=0)
    gy = ndimage.sobel(smooth, axis=1)
    gz = ndimage.sobel(smooth, axis=2)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    # Normalize over occupied boundary-support region
    region = occ
    if np.any(region):
        gmax = np.nanmax(grad_mag[region])
        if gmax > 0:
            grad_mag = grad_mag / gmax

    grad_mag[~occ] = np.nan
    return grad_mag


def export_point_cloud(points_xyz: np.ndarray, scalar: np.ndarray, outpath: Path, name: str):
    """
    Save colored point cloud as PLY using a blue-red map. Can change the color obvoisuly! 
    """
    vals = np.asarray(scalar, dtype=np.float64)
    valid = np.isfinite(vals)
    pts = points_xyz[valid]
    vals = vals[valid]

    if len(vals) == 0:
        return

    vmin = np.min(vals)
    vmax = np.max(vals)
    if vmax <= vmin:
        norm = np.zeros_like(vals)
    else:
        norm = (vals - vmin) / (vmax - vmin)

    """
    cmap = plt.get_cmap("turbo")
    rgba = (255 * cmap(norm)).astype(np.uint64)
    colors = rgba[:, :2]                            # didn't work??
    """

    cmap = plt.get_cmap("turbo")
    rgba = (255 * cmap(norm)).astype(np.uint8)
    colors = rgba[:, :3]

    pc = trimesh.points.PointCloud(pts, colors=colors)
    pc.metadata["name"] = name
    pc.export(outpath)


def save_mid_slice_image(field: np.ndarray, occ: np.ndarray, outpath: Path, title: str):
    """
    Save a middle-z slice image. For some reason this is notworking too well. In principle it is very simple
    """
    z_indices = np.where(np.any(np.any(occ, axis=0), axis=0))[0]
    if len(z_indices) == 0:
        return
    z = z_indices[len(z_indices) // 2]

    img = field[:, :, z].copy()
    mask = occ[:, :, z]
    img = np.where(mask, img, np.nan)

    plt.figure(figsize=(7, 6))
    plt.imshow(np.rot90(img), cmap="turbo")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def masked_mean(field: np.ndarray, occ: np.ndarray) -> float:
    vals = field[occ]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def simulate(occ: np.ndarray, voxel_pitch: float, params: Params, outdir: Path):
    occ = occ.astype(bool)
    _, boundary = build_boundary_masks(occ)

    # Geometry-derived fields
    dist_surface_mm = nearest_surface_distance(boundary, occ, voxel_pitch)
    curvature_proxy = estimate_curvature_proxy(
        boundary_mask=boundary,
        occ=occ,
        sigma=params.smooth_curvature_sigma
    )

    # thinness proxy: larger near interior cores
    max_dist = np.nanmax(dist_surface_mm[occ]) if np.any(occ) else 1.0
    thinness_proxy = 1.0 - (dist_surface_mm / max_dist if max_dist > 0 else 0.0)
    thinness_proxy[~occ] = np.nan

    # fields
    W = np.zeros_like(occ, dtype=np.float64)               # water
    A = np.zeros_like(occ, dtype=np.float64)               # acid / oligomers lumped
    Mn_rel = np.ones_like(occ, dtype=np.float64)           # Mn / Mn0
    phi = np.zeros_like(occ, dtype=np.float64)             # porosity
    E_rel = np.ones_like(occ, dtype=np.float64)            # E / E0

    # Exterior target water concentration at boundary
    W_ext = np.zeros_like(W)
    W_ext[boundary] = 1.0

    steps = int(math.ceil(params.days / params.dt))
    save_every = max(1, int(round(params.save_every_days / params.dt)))

    history = {
        "time_days": [],
        "Mn_rel_mean": [],
        "phi_mean": [],
        "A_mean": [],
        "E_rel_mean": [],
        "risk_mean": [],
    }

    snapshots = {}

    for step in range(steps + 1):
        t = step * params.dt

        # Boundary exchange terms
        water_exchange = np.zeros_like(W)
        acid_exchange = np.zeros_like(A)
        water_exchange[boundary] = params.k_water_uptake * (1.0 - W[boundary])
        acid_exchange[boundary] = -params.k_acid_out * A[boundary]

        # Diffusion
        lap_W = laplacian_masked(W, occ)
        lap_A = laplacian_masked(A, occ)

        # Reaction rate
        hydro_rate = (params.k_hydro + params.k_auto * (A ** params.acid_order)) * W * Mn_rel
        hydro_rate[~occ] = 0.0

        # Update PDEs / ODEs
        dW = params.Dw * lap_W + water_exchange - 0.01 * hydro_rate
        dA = params.Da * lap_A + acid_exchange + params.acid_from_damage * hydro_rate - params.acid_decay * A
        dMn = -hydro_rate * Mn_rel
        dphi = params.k_phi * hydro_rate * (1.0 - phi / params.phi_max)

        W[occ] += params.dt * dW[occ]
        A[occ] += params.dt * dA[occ]
        Mn_rel[occ] += params.dt * dMn[occ]
        phi[occ] += params.dt * dphi[occ]

        # Clamp
        W[occ] = np.clip(W[occ], 0.0, 1.5)
        A[occ] = np.clip(A[occ], 0.0, None)
        Mn_rel[occ] = np.clip(Mn_rel[occ], params.minimum_Mn_rel, 1.0)
        phi[occ] = np.clip(phi[occ], 0.0, params.phi_max)

        # Derived modulus
        E_rel[occ] = (
            np.power(Mn_rel[occ], params.alpha_E_Mn) *
            np.power(np.clip(1.0 - phi[occ], 1e-8, None), params.beta_E_phi)
        )
        E_rel[occ] = np.clip(E_rel[occ], 1e-4, 1.0)

        # Risk / weak-point proxy
        # Stress_proxy increases at edges/corners and thin ligaments
        stress_proxy = (
            1.0
            + params.stress_sensitivity_curvature * np.nan_to_num(curvature_proxy, nan=0.0)
            + params.stress_sensitivity_thinness * np.nan_to_num(thinness_proxy, nan=0.0)
        )
        # strength proxy tracks local modulus
        strength_proxy = np.clip(E_rel, params.minimum_strength_factor, None)
        risk = np.zeros_like(E_rel)
        risk[occ] = stress_proxy[occ] / strength_proxy[occ]
        risk[~occ] = np.nan

        # Save history
        if step % save_every == 0 or step == steps:
            history["time_days"].append(t)
            history["Mn_rel_mean"].append(masked_mean(Mn_rel, occ))
            history["phi_mean"].append(masked_mean(phi, occ))
            history["A_mean"].append(masked_mean(A, occ))
            history["E_rel_mean"].append(masked_mean(E_rel, occ))
            history["risk_mean"].append(masked_mean(risk, occ))
            snapshots[round(t, 6)] = {
                "W": W.copy(),
                "A": A.copy(),
                "Mn_rel": Mn_rel.copy(),
                "phi": phi.copy(),
                "E_rel": E_rel.copy(),
                "risk": risk.copy(),
            }
            print(
                f"t={t:7.2f} d | "
                f"Mn={history['Mn_rel_mean'][-1]:.4f} | "
                f"phi={history['phi_mean'][-1]:.4f} | "
                f"A={history['A_mean'][-1]:.4f} | "
                f"E={history['E_rel_mean'][-1]:.4f} | "
                f"risk={history['risk_mean'][-1]:.3f}"
            )

    return {
        "boundary": boundary,
        "dist_surface_mm": dist_surface_mm,
        "curvature_proxy": curvature_proxy,
        "thinness_proxy": thinness_proxy,
        "history": history,
        "snapshots": snapshots,
        "final": {
            "W": W,
            "A": A,
            "Mn_rel": Mn_rel,
            "phi": phi,
            "E_rel": E_rel,
            "risk": risk,
        },
    }


def voxel_centers_from_points(points_xyz: np.ndarray, occ: np.ndarray) -> np.ndarray:
    # trimesh voxel points are already occupied-voxel centers in the same order as occ.nonzero()
    # keep them as is
    return points_xyz


def flatten_field(field: np.ndarray, occ: np.ndarray) -> np.ndarray:
    return field[occ].astype(np.float64)


def save_csv(points_xyz: np.ndarray, fields: dict[str, np.ndarray], out_csv: Path):
    names = list(fields.keys())
    arrs = [np.asarray(fields[n]).reshape(-1) for n in names]

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_mm", "y_mm", "z_mm", *names])
        for i in range(len(points_xyz)):
            row = [points_xyz[i, 0], points_xyz[i, 1], points_xyz[i, 2]]
            row.extend(float(a[i]) for a in arrs)
            writer.writerow(row)


def save_history_plots(history: dict, outdir: Path):
    t = np.asarray(history["time_days"])

    def one(ykey: str, title: str, ylabel: str, filename: str):
        plt.figure(figsize=(7, 4.5))
        plt.plot(t, history[ykey], marker="o")
        plt.xlabel("Time [days]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()

    one("Mn_rel_mean", "Mean molecular weight ratio", "Mean Mn/Mn0", "history_Mn_rel.png")
    one("phi_mean", "Mean porosity", "Mean porosity", "history_porosity.png")
    one("A_mean", "Mean acid concentration", "Mean acid", "history_acid.png")
    one("E_rel_mean", "Mean modulus ratio", "Mean E/E0", "history_modulus.png")
    one("risk_mean", "Mean weak-point risk proxy", "Mean risk", "history_risk.png")


def main():
    args = parse_args()
    mesh_path = Path(args.mesh)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = Params(
        voxel_pitch=args.voxel_pitch,
        days=args.days,
        dt=args.dt,
        save_every_days=args.save_every_days,
    )

    print(f"Loading mesh: {mesh_path}")
    mesh, vox_filled, occ, transform, points_xyz = load_and_voxelize(mesh_path, params.voxel_pitch)
    print(f"Occupied voxels: {int(occ.sum())}")
    print(f"Voxel grid shape: {occ.shape}")

    result = simulate(occ=occ, voxel_pitch=params.voxel_pitch, params=params, outdir=outdir)

    final = result["final"]
    boundary = result["boundary"]

    occ_points = voxel_centers_from_points(points_xyz, occ)

    fields_flat = {
        "boundary": flatten_field(boundary.astype(float), occ),
        "dist_surface_mm": flatten_field(result["dist_surface_mm"], occ),
        "curvature_proxy": flatten_field(result["curvature_proxy"], occ),
        "thinness_proxy": flatten_field(result["thinness_proxy"], occ),
        "water": flatten_field(final["W"], occ),
        "acid": flatten_field(final["A"], occ),
        "Mn_rel": flatten_field(final["Mn_rel"], occ),
        "porosity": flatten_field(final["phi"], occ),
        "modulus_rel": flatten_field(final["E_rel"], occ),
        "risk": flatten_field(final["risk"], occ),
    }

    # Save CSV + NPZ
    save_csv(occ_points, fields_flat, outdir / "fields_final.csv")
    np.savez_compressed(outdir / "fields_final.npz", xyz=occ_points, **fields_flat)

    # Save colored point clouds
    export_point_cloud(occ_points, fields_flat["Mn_rel"], outdir / "Mn_rel_final.ply", "Mn_rel")
    export_point_cloud(occ_points, fields_flat["porosity"], outdir / "porosity_final.ply", "porosity")
    export_point_cloud(occ_points, fields_flat["modulus_rel"], outdir / "modulus_final.ply", "modulus_rel")
    export_point_cloud(occ_points, fields_flat["risk"], outdir / "risk_final.ply", "risk")

    # Save slice images
    save_mid_slice_image(final["Mn_rel"], occ, outdir / "slice_Mn_rel.png", "Mid-slice Mn/Mn0")
    save_mid_slice_image(final["phi"], occ, outdir / "slice_porosity.png", "Mid-slice porosity")
    save_mid_slice_image(final["E_rel"], occ, outdir / "slice_modulus.png", "Mid-slice modulus ratio")
    save_mid_slice_image(final["risk"], occ, outdir / "slice_risk.png", "Mid-slice risk proxy")

    save_history_plots(result["history"], outdir)

    # Save a quick summary txt
    occ_mask = occ
    with (outdir / "summary.txt").open("w") as f:
        f.write("CAD degradation heat-map prototype summary\n")
        f.write(f"Input mesh: {mesh_path}\n")
        f.write(f"Occupied voxels: {int(occ.sum())}\n")
        f.write(f"Voxel grid shape: {occ.shape}\n")
        f.write(f"Voxel pitch [mm]: {params.voxel_pitch}\n")
        f.write(f"Duration [days]: {params.days}\n")
        f.write(f"Final mean Mn/Mn0: {masked_mean(final['Mn_rel'], occ_mask):.6f}\n")
        f.write(f"Final mean porosity: {masked_mean(final['phi'], occ_mask):.6f}\n")
        f.write(f"Final mean modulus/E0: {masked_mean(final['E_rel'], occ_mask):.6f}\n")
        f.write(f"Final mean risk: {masked_mean(final['risk'], occ_mask):.6f}\n")
        f.write("\nInterpretation\n")
        f.write("- High risk indicates geometrically stressed regions that have also degraded locally.\n")
        f.write("- This is a screening metric, not a substitute for full FEA.\n")

    print(f"\nDone. Outputs written to: {outdir.resolve()}")
    print("Open the PLY point clouds in MeshLab / ParaView / Blender for colored heat-map visualization.")


if __name__ == "__main__":
    main()
