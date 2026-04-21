from __future__ import annotations

"""
speed/disk changes:
- in-place state updates instead of copying full 3D arrays every step
- PDE work cropped to the active occupied bounding box + small margin
- optional crystallinity only when enabled
- intended to pair with a run file that exports less often and skips STL by default
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import trimesh
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False

Array = np.ndarray


def kd_from_sav(k0: float, sav: float | np.ndarray, m: float) -> float | np.ndarray:
    sav_arr = np.asarray(sav, dtype=float)
    kd = float(k0) * np.power(np.maximum(sav_arr, 1e-12), float(m))
    return np.maximum(kd, 0.0)


@dataclass(frozen=True)
class CrystallinityParams:
    xc0: float = 0.45
    xc_max: float = 0.70
    a1: float = 0.75
    kc1: float = 4.0
    nav: float = 2.0
    a2: float = 0.25
    kc2: float = 0.35


@dataclass(frozen=True)
class AutocatDiffusionParams:
    mn0: float = 60000.0
    e0: float = 350.0
    alpha: float = 2.0
    beta_phi: float = 2.0
    k0: float = 0.0
    m: float = 1.0

    Dw: float = 438.0
    Da: float = 91.25
    k_water_uptake: float = 730.0
    k_acid_out: float = 146.0
    k_auto: float = 2.19
    acid_order: float = 1.0
    acid_from_damage: float = 5.475
    acid_decay: float = 0.73
    water_reaction_sink: float = 3.65

    k_phi: float = 0.040
    phi_max: float = 0.65

    kd_surface_boost: float = 1.12
    kd_core_drop: float = 0.94
    curvature_boost: float = 0.0

    minimum_mn_ratio: float = 1e-6
    max_water: float = 1.5
    acid_clip: float = 50.0

    enable_crystallinity: bool = False
    acid_xc_coupling: float = 0.0
    cryst_params: Optional[CrystallinityParams] = None


@dataclass
class DegradationState:
    water: Array
    acid: Array
    mn_ratio: Array
    porosity: Array
    crystallinity: Optional[Array] = None


@dataclass(frozen=True)
class GeometryFields:
    boundary: Array
    local_mult: Array
    curvature: Array
    surface_exposure: Array


@dataclass(frozen=True)
class StepResult:
    mn: Array
    mn_ratio: Array
    e_rel: Array
    risk: Array
    porosity: Array
    crystallinity: Optional[Array] = None


def mn_t(t: np.ndarray, mn0: float, k0: float, sav: float, m: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    kd = kd_from_sav(k0, sav, m)
    return mn0 * np.exp(-kd * np.maximum(t, 0.0))


def crystallinity_bulk_pcl(t: np.ndarray, mn: np.ndarray, p: CrystallinityParams) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    mn = np.asarray(mn, dtype=float)
    mn0_ref = max(float(mn[0]), 1e-12)
    deg = np.clip(1.0 - mn / mn0_ref, 0.0, 1.0)
    primary = 1.0 - np.exp(-max(p.kc1, 1e-12) * np.power(deg, max(p.nav, 1e-12)))
    secondary = deg * (1.0 - np.exp(-max(p.kc2, 1e-12) * np.maximum(t, 0.0)))
    response = (max(float(p.a1), 0.0) * primary + max(float(p.a2), 0.0) * secondary) / max(float(p.a1 + p.a2), 1e-12)
    xc = p.xc0 + (p.xc_max - p.xc0) * np.clip(response, 0.0, 1.0)
    return np.clip(xc, 0.0, p.xc_max)


def calibrate_k0(mn0: float, mn_target_ratio: float, t_target: float, sav: float, m: float) -> float:
    if not (0.0 < mn_target_ratio < 1.0):
        raise ValueError("mn_target_ratio must be in (0,1)")
    if t_target <= 0:
        raise ValueError("t_target must be > 0")
    if sav <= 0:
        raise ValueError("sav must be > 0")
    return -math.log(mn_target_ratio) / (float(sav) ** float(m) * float(t_target))


def load_mesh_compute_sav(path: str) -> Tuple[float, float, float]:
    if not _HAS_TRIMESH:
        raise ImportError("trimesh is required: pip install trimesh")
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")
    sa = float(mesh.area)
    vol = float(mesh.volume)
    if vol <= 0:
        raise ValueError(f"Non-positive volume ({vol}) for mesh {path}")
    return sa, vol, sa / vol


def build_boundary_mask(occ: Array) -> Array:
    from scipy import ndimage
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(occ, structure=structure, border_value=0)
    return occ & (~eroded)


def crop_bounds(mask: Array, margin: int = 1) -> tuple[slice, slice, slice]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return slice(0, 0), slice(0, 0), slice(0, 0)
    lo = np.maximum(pts.min(axis=0) - margin, 0)
    hi = np.minimum(pts.max(axis=0) + margin + 1, np.array(mask.shape))
    return slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2])


def laplacian_masked(field: Array, occ: Array, out: Optional[Array] = None, neighbor_count: Optional[Array] = None) -> Array:
    if out is None:
        out = np.zeros_like(field, dtype=np.float32)
    else:
        out.fill(0.0)
    if neighbor_count is None:
        neighbor_count = np.zeros_like(field, dtype=np.float32)
    else:
        neighbor_count.fill(0.0)

    out[1:, :, :] += field[:-1, :, :] * occ[:-1, :, :]
    out[:-1, :, :] += field[1:, :, :] * occ[1:, :, :]
    out[:, 1:, :] += field[:, :-1, :] * occ[:, :-1, :]
    out[:, :-1, :] += field[:, 1:, :] * occ[:, 1:, :]
    out[:, :, 1:] += field[:, :, :-1] * occ[:, :, :-1]
    out[:, :, :-1] += field[:, :, 1:] * occ[:, :, 1:]

    neighbor_count[1:, :, :] += occ[:-1, :, :]
    neighbor_count[:-1, :, :] += occ[1:, :, :]
    neighbor_count[:, 1:, :] += occ[:, :-1, :]
    neighbor_count[:, :-1, :] += occ[:, 1:, :]
    neighbor_count[:, :, 1:] += occ[:, :, :-1]
    neighbor_count[:, :, :-1] += occ[:, :, 1:]

    out -= neighbor_count * field
    out[~occ] = 0.0
    return out


def make_geometry_fields(mask: Array, kd_surface_boost: float, kd_core_drop: float, curvature_boost: float) -> GeometryFields:
    from scipy import ndimage
    inside_dist = ndimage.distance_transform_edt(mask).astype(np.float32)
    maxd = float(np.max(inside_dist)) if np.any(mask) else 1.0
    core_norm = inside_dist / max(maxd, 1e-12)
    surface_exposure = (1.0 - core_norm).astype(np.float32)

    smooth = ndimage.gaussian_filter(mask.astype(np.float32), sigma=1.0)
    gx, gy, gz = np.gradient(smooth)
    curv = np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)
    cmax = float(np.max(curv)) if np.any(mask) else 0.0
    if cmax > 0.0:
        curv /= cmax

    local_mult = (
        float(kd_core_drop)
        + (float(kd_surface_boost) - float(kd_core_drop)) * surface_exposure
        + float(curvature_boost) * curv
    )
    local_mult = np.clip(local_mult, 0.05, None).astype(np.float32)
    boundary = build_boundary_mask(mask)
    return GeometryFields(boundary=boundary, local_mult=local_mult, curvature=curv, surface_exposure=surface_exposure)


def init_state(mask: Array, p: AutocatDiffusionParams) -> DegradationState:
    base = np.zeros(mask.shape, dtype=np.float32)
    cryst = None
    if p.enable_crystallinity:
        cp = p.cryst_params or CrystallinityParams()
        cryst = np.full(mask.shape, cp.xc0, dtype=np.float32)
    return DegradationState(
        water=base.copy(),
        acid=base.copy(),
        mn_ratio=np.ones(mask.shape, dtype=np.float32),
        porosity=base.copy(),
        crystallinity=cryst,
    )


def step_degradation_inplace(
    state: DegradationState,
    occ: Array,
    geom: GeometryFields,
    params: AutocatDiffusionParams,
    dt: float,
    sav: float,
    current_time: float,
    substeps: int = 1,
) -> DegradationState:
    del current_time
    if not np.any(occ):
        return state

    sl = crop_bounds(occ, margin=1)
    occ_s = occ[sl]
    bnd_s = geom.boundary[sl]
    mult_s = geom.local_mult[sl]

    water = state.water[sl]
    acid = state.acid[sl]
    mn_ratio = state.mn_ratio[sl]
    porosity = state.porosity[sl]
    cryst = state.crystallinity[sl] if state.crystallinity is not None else None

    kd_base = np.float32(kd_from_sav(params.k0, sav, params.m))
    k_base_local = (kd_base * mult_s).astype(np.float32)

    lap_w = np.zeros_like(water, dtype=np.float32)
    lap_a = np.zeros_like(acid, dtype=np.float32)
    ngh_w = np.zeros_like(water, dtype=np.float32)
    ngh_a = np.zeros_like(acid, dtype=np.float32)
    water_exchange = np.zeros_like(water, dtype=np.float32)
    acid_exchange = np.zeros_like(acid, dtype=np.float32)

    dt_sub = np.float32(dt / max(substeps, 1))

    for _ in range(max(substeps, 1)):
        laplacian_masked(water, occ_s, out=lap_w, neighbor_count=ngh_w)
        laplacian_masked(acid, occ_s, out=lap_a, neighbor_count=ngh_a)

        water_exchange.fill(0.0)
        acid_exchange.fill(0.0)
        water_exchange[bnd_s] = np.float32(params.k_water_uptake) * (1.0 - water[bnd_s])
        acid_exchange[bnd_s] = -np.float32(params.k_acid_out) * acid[bnd_s]

        acid_term = 1.0 + np.float32(params.k_auto) * np.power(np.maximum(acid, 0.0), np.float32(params.acid_order))
        if params.enable_crystallinity and cryst is not None and params.acid_xc_coupling != 0.0:
            xc_denom = np.maximum(1.0 - cryst, 1e-5)
            acid_term *= 1.0 + np.float32(params.acid_xc_coupling) * (acid / xc_denom)

        hydro_rate = k_base_local * water * acid_term
        hydro_rate[~occ_s] = 0.0

        water[occ_s] += dt_sub * (np.float32(params.Dw) * lap_w[occ_s] + water_exchange[occ_s] - np.float32(params.water_reaction_sink) * hydro_rate[occ_s])
        acid[occ_s] += dt_sub * (np.float32(params.Da) * lap_a[occ_s] + acid_exchange[occ_s] + np.float32(params.acid_from_damage) * hydro_rate[occ_s] - np.float32(params.acid_decay) * acid[occ_s])
        mn_ratio[occ_s] += dt_sub * (-hydro_rate[occ_s] * mn_ratio[occ_s])
        porosity[occ_s] += dt_sub * (np.float32(params.k_phi) * hydro_rate[occ_s] * np.maximum(1.0 - porosity[occ_s] / np.float32(params.phi_max), 0.0))

        np.clip(water, 0.0, np.float32(params.max_water), out=water)
        np.clip(acid, 0.0, np.float32(params.acid_clip), out=acid)
        np.clip(mn_ratio, np.float32(params.minimum_mn_ratio), 1.0, out=mn_ratio)
        np.clip(porosity, 0.0, np.float32(params.phi_max), out=porosity)

    if params.enable_crystallinity and state.crystallinity is not None:
        cp = params.cryst_params or CrystallinityParams()
        # fast algebraic approximation instead of per-voxel history loop
        deg = np.clip(1.0 - mn_ratio, 0.0, 1.0)
        state.crystallinity[sl][occ_s] = (cp.xc0 + (cp.xc_max - cp.xc0) * deg[occ_s]).astype(np.float32)

    state.water[sl] = water
    state.acid[sl] = acid
    state.mn_ratio[sl] = mn_ratio
    state.porosity[sl] = porosity
    return state


def compute_export_fields(state: DegradationState, occ: Array, params: AutocatDiffusionParams) -> StepResult:
    mn_ratio = np.asarray(state.mn_ratio, dtype=np.float32).copy()
    mn_ratio[~occ] = 0.0
    mn = (np.float32(params.mn0) * mn_ratio).astype(np.float32)
    porosity = np.asarray(state.porosity, dtype=np.float32)
    e_rel = (np.power(np.maximum(mn_ratio, 1e-12), np.float32(params.alpha)) * np.power(np.maximum(1.0 - porosity, 1e-8), np.float32(params.beta_phi))).astype(np.float32)
    e_rel[~occ] = 0.0
    deg = (1.0 - mn_ratio).astype(np.float32)
    risk = (deg * (1.0 + 1.5 * deg)).astype(np.float32)
    risk[~occ] = 0.0
    return StepResult(mn=mn, mn_ratio=mn_ratio, e_rel=e_rel, risk=risk, porosity=porosity, crystallinity=state.crystallinity)


__all__ = [
    "AutocatDiffusionParams",
    "CrystallinityParams",
    "DegradationState",
    "GeometryFields",
    "StepResult",
    "build_boundary_mask",
    "calibrate_k0",
    "compute_export_fields",
    "crop_bounds",
    "crystallinity_bulk_pcl",
    "init_state",
    "kd_from_sav",
    "laplacian_masked",
    "load_mesh_compute_sav",
    "make_geometry_fields",
    "mn_t",
    "step_degradation_inplace",
]
