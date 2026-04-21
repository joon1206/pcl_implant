from __future__ import annotations

"""
two-phase proxy removed and replaces it with the local,
incremental update driven by:
    - baseline hydrolysis scaled by global SA/V
    - local surface/core geometry multiplier
    - water ingress
    - retained acidic product diffusion + washout
    - optional crystallinity coupling

memory adjustments:
    - no time x nx x ny x nz temporary arrays
    - only a handful of 3D working fields are kept per step
    - crystallinity is only computed when enabled
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

    # baseline global scaling
    k0: float = 0.0
    m: float = 1.0

    # transport and chemistry
    Dw: float = 438.0               # ~1.2 voxel^2/day expressed in voxel^2/year
    Da: float = 91.25               # ~0.25 voxel^2/day expressed in voxel^2/year
    k_water_uptake: float = 730.0   # ~2/day in 1/year
    k_acid_out: float = 146.0       # ~0.4/day in 1/year

    k_auto: float = 2.19            # ~6e-3/day / conc in 1/year / conc
    acid_order: float = 1.0
    acid_from_damage: float = 5.475 # ~0.015/day in 1/year
    acid_decay: float = 0.73        # ~0.002/day in 1/year
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
    w1 = max(float(p.a1), 0.0)
    w2 = max(float(p.a2), 0.0)
    response = (w1 * primary + w2 * secondary) / max(w1 + w2, 1e-12)
    xc = p.xc0 + (p.xc_max - p.xc0) * np.clip(response, 0.0, 1.0)
    return np.clip(xc, 0.0, p.xc_max)


def porosity_from_mn(mn: np.ndarray, mn0: float, k_phi: float = 0.040, phi_max: float = 0.65) -> np.ndarray:
    mn = np.asarray(mn, dtype=float)
    deg = np.clip(1.0 - mn / max(mn0, 1e-12), 0.0, 1.0)
    return phi_max * (1.0 - np.exp(-float(k_phi) * deg))


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


def laplacian_masked(field: Array, occ: Array, out: Optional[Array] = None) -> Array:
    if out is None:
        out = np.zeros_like(field, dtype=np.float32)
    else:
        out.fill(0.0)

    out[1:, :, :] += field[:-1, :, :] * occ[:-1, :, :]
    out[:-1, :, :] += field[1:, :, :] * occ[1:, :, :]
    out[:, 1:, :] += field[:, :-1, :] * occ[:, :-1, :]
    out[:, :-1, :] += field[:, 1:, :] * occ[:, 1:, :]
    out[:, :, 1:] += field[:, :, :-1] * occ[:, :, :-1]
    out[:, :, :-1] += field[:, :, 1:] * occ[:, :, 1:]

    neighbor_count = np.zeros_like(field, dtype=np.float32)
    neighbor_count[1:, :, :] += occ[:-1, :, :]
    neighbor_count[:-1, :, :] += occ[1:, :, :]
    neighbor_count[:, 1:, :] += occ[:, :-1, :]
    neighbor_count[:, :-1, :] += occ[:, 1:, :]
    neighbor_count[:, :, 1:] += occ[:, :, :-1]
    neighbor_count[:, :, :-1] += occ[:, :, 1:]

    out -= neighbor_count * field
    out[~occ] = 0.0
    return out.astype(np.float32, copy=False)


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
    return GeometryFields(
        boundary=boundary,
        local_mult=local_mult,
        curvature=curv,
        surface_exposure=surface_exposure,
    )


def init_state(mask: Array, p: AutocatDiffusionParams) -> DegradationState:
    base = np.zeros(mask.shape, dtype=np.float32)
    mn_ratio = np.ones(mask.shape, dtype=np.float32)
    porosity = np.zeros(mask.shape, dtype=np.float32)
    cryst = None
    if p.enable_crystallinity:
        cp = p.cryst_params or CrystallinityParams()
        cryst = np.full(mask.shape, cp.xc0, dtype=np.float32)
    return DegradationState(
        water=base.copy(),
        acid=base.copy(),
        mn_ratio=mn_ratio,
        porosity=porosity,
        crystallinity=cryst,
    )


def update_crystallinity_from_ratio(
    current_time: float,
    mn_ratio: Array,
    mn0: float,
    cp: CrystallinityParams,
) -> Array:
    t_hist = np.array([0.0, float(current_time)], dtype=float)
    mn_now = mn0 * np.asarray(mn_ratio, dtype=np.float32)
    flat_mn = mn_now.reshape(-1)
    out = np.empty_like(flat_mn, dtype=np.float32)
    for i in range(flat_mn.size):
        out[i] = float(crystallinity_bulk_pcl(t_hist, np.array([float(mn0), float(flat_mn[i])]), cp)[-1])
    return out.reshape(mn_ratio.shape)


def step_degradation(
    state: DegradationState,
    occ: Array,
    geom: GeometryFields,
    params: AutocatDiffusionParams,
    dt: float,
    sav: float,
    current_time: float,
) -> DegradationState:
    occ = np.asarray(occ, dtype=bool)
    boundary = geom.boundary

    water = state.water
    acid = state.acid
    mn_ratio = state.mn_ratio
    porosity = state.porosity
    crystallinity = state.crystallinity

    kd_base = np.float32(kd_from_sav(params.k0, sav, params.m))
    k_base_local = (kd_base * geom.local_mult).astype(np.float32)

    lap_w = laplacian_masked(water, occ)
    lap_a = laplacian_masked(acid, occ)

    water_exchange = np.zeros_like(water, dtype=np.float32)
    acid_exchange = np.zeros_like(acid, dtype=np.float32)
    water_exchange[boundary] = np.float32(params.k_water_uptake) * (1.0 - water[boundary])
    acid_exchange[boundary] = -np.float32(params.k_acid_out) * acid[boundary]

    acid_term = 1.0 + np.float32(params.k_auto) * np.power(np.maximum(acid, 0.0), np.float32(params.acid_order))
    if params.enable_crystallinity and crystallinity is not None and params.acid_xc_coupling != 0.0:
        xc_denom = np.maximum(1.0 - crystallinity, 1e-5)
        acid_term *= 1.0 + np.float32(params.acid_xc_coupling) * (acid / xc_denom)

    hydro_rate = k_base_local * water * acid_term
    hydro_rate[~occ] = 0.0

    dW = np.float32(params.Dw) * lap_w + water_exchange - np.float32(params.water_reaction_sink) * hydro_rate
    dA = (
        np.float32(params.Da) * lap_a
        + acid_exchange
        + np.float32(params.acid_from_damage) * hydro_rate
        - np.float32(params.acid_decay) * acid
    )
    dMn = -hydro_rate * mn_ratio
    dphi = np.float32(params.k_phi) * hydro_rate * np.maximum(1.0 - porosity / np.float32(params.phi_max), 0.0)

    water_new = water.copy()
    acid_new = acid.copy()
    mn_ratio_new = mn_ratio.copy()
    porosity_new = porosity.copy()

    water_new[occ] += np.float32(dt) * dW[occ]
    acid_new[occ] += np.float32(dt) * dA[occ]
    mn_ratio_new[occ] += np.float32(dt) * dMn[occ]
    porosity_new[occ] += np.float32(dt) * dphi[occ]

    water_new[occ] = np.clip(water_new[occ], 0.0, np.float32(params.max_water))
    acid_new[occ] = np.clip(acid_new[occ], 0.0, np.float32(params.acid_clip))
    mn_ratio_new[occ] = np.clip(mn_ratio_new[occ], np.float32(params.minimum_mn_ratio), 1.0)
    porosity_new[occ] = np.clip(porosity_new[occ], 0.0, np.float32(params.phi_max))

    cryst_new = None
    if params.enable_crystallinity:
        cp = params.cryst_params or CrystallinityParams()
        cryst_new = update_crystallinity_from_ratio(current_time + dt, mn_ratio_new, params.mn0, cp)

    return DegradationState(
        water=water_new,
        acid=acid_new,
        mn_ratio=mn_ratio_new,
        porosity=porosity_new,
        crystallinity=cryst_new,
    )


def compute_export_fields(state: DegradationState, occ: Array, params: AutocatDiffusionParams) -> StepResult:
    mn_ratio = np.asarray(state.mn_ratio, dtype=np.float32)
    mn = (np.float32(params.mn0) * mn_ratio).astype(np.float32)
    porosity = np.asarray(state.porosity, dtype=np.float32)

    e_rel = (
        np.power(np.maximum(mn_ratio, 1e-12), np.float32(params.alpha))
        * np.power(np.maximum(1.0 - porosity, 1e-8), np.float32(params.beta_phi))
    ).astype(np.float32)
    e_rel[~occ] = 0.0

    deg = (1.0 - mn_ratio).astype(np.float32)
    risk = (deg * (1.0 + 1.5 * deg)).astype(np.float32)
    risk[~occ] = 0.0
    mn[~occ] = 0.0
    mn_ratio = mn_ratio.copy()
    mn_ratio[~occ] = 0.0

    return StepResult(
        mn=mn,
        mn_ratio=mn_ratio,
        e_rel=e_rel,
        risk=risk,
        porosity=porosity,
        crystallinity=state.crystallinity,
    )


__all__ = [
    "AutocatDiffusionParams",
    "CrystallinityParams",
    "DegradationState",
    "GeometryFields",
    "StepResult",
    "build_boundary_mask",
    "calibrate_k0",
    "compute_export_fields",
    "crystallinity_bulk_pcl",
    "init_state",
    "kd_from_sav",
    "laplacian_masked",
    "load_mesh_compute_sav",
    "make_geometry_fields",
    "mn_t",
    "porosity_from_mn",
    "step_degradation",
    "update_crystallinity_from_ratio",
]
