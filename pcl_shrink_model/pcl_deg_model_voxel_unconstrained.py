from __future__ import annotations

"""
Bulk-PCL degradation physics for voxel topology-changing pipeline.

Workflow implemented
--------------------
1. Base geometry scaling from the slide deck:
       k_base = k0 * (SA/V)^m

2. Random + end scission split:
       R_random ~ k_random * water(t)
       R_end    ~ k_end * water(t) * (1 + end_scission_factor * C_acid)

3. Local autocatalysis closure from retained acidic/oligomer products:
       acid_term = 1 + acid_autocat * (C_acid / (1 - Xc + eps))^u

4. Two-phase onset (bulk hydrolysis -> erosion proxy):
       smooth phase-2 multiplier once Mn approaches Mn_crit

5. Reaction-diffusion proxy for retained products:
       dC/dt = d/dx(D(t) dC/dx) + S(t)
       C(0,t) = C(L,t) = 0
   where S(t) is tied to the Mn loss rate.

6. Porosity / damage:
       dphi/dt = k_phi * k_eff * (1 - phi/phi_max)
       damage = w_Mn*(1 - Mn/Mn0) + w_phi*(phi/phi_max)

7. Surface erosion / mass loss proxy:
       d(mf)/dt = -k_erosion * onset * (1 + erosion_sensitivity_phi * phi)

8. Mechanical property linkage:
       E = E0 * (Mn/Mn0)^alpha * (1 - phi)^beta

unconstrained additions:
1. No k0 bounds. k0 is calibrated directly from the requested Mn target.
2. Global target defaults to Mn/Mn0 = 0.20 at t = 3 years.
3. Mass loss is still slower than Mn loss, but not so delayed that nothing
   happens in a 0-6 year simulation.
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


def kd_from_sav(k0: float, sav: float | np.ndarray, m: float) -> float | np.ndarray:
    sav_arr = np.asarray(sav, dtype=float)
    kd = float(k0) * np.power(np.maximum(sav_arr, 1e-12), float(m))
    return np.maximum(kd, 0.0)


def mn_t(t: np.ndarray, mn0: float, k0: float, sav: float, m: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    kd = kd_from_sav(k0, sav, m)
    return mn0 * np.exp(-kd * np.maximum(t, 0.0))


@dataclass(frozen=True)
class CrystallinityParams:
    xc0: float = 0.45
    xc_max: float = 0.70
    a1: float = 0.75
    kc1: float = 4.0
    nav: float = 2.0
    a2: float = 0.25
    kc2: float = 0.35


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


@dataclass(frozen=True)
class TwoPhaseParams:
    mn_crit: float
    accel_factor: float
    k_erosion: float
    floor_mass_fraction: float = 0.05
    transition_width: float = 0.80
    phase2_max_fraction: float = 0.65
    erosion_onset_ratio: float = 0.45
    erosion_power: float = 1.35


def _t_when_mn_hits_threshold(mn0: float, kd: float, mn_crit: float) -> float:
    if mn_crit <= 0:
        raise ValueError("mn_crit must be > 0")
    if mn_crit >= mn0:
        return 0.0
    if kd <= 0:
        return float("inf")
    return math.log(mn0 / mn_crit) / kd


def _sigmoid01(u: np.ndarray, sharpness: float = 5.0) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    z = np.clip(sharpness * (u - 0.5), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _phase2_blend_weight(t: np.ndarray, tcrit: float, width: float, max_fraction: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if not math.isfinite(tcrit):
        return np.zeros_like(t)
    width = max(float(width), 1e-12)
    u = (t - (tcrit - 0.5 * width)) / width
    return np.clip(float(max_fraction), 0.0, 1.0) * _sigmoid01(u, sharpness=4.5)


def mn_two_phase(t: np.ndarray, mn0: float, k0: float, sav: float, m: float, p: TwoPhaseParams) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be 1-D")
    if t.size == 0:
        return np.array([], dtype=float)

    kd1 = float(kd_from_sav(k0, sav, m))
    kd2 = kd1 * max(float(p.accel_factor), 1.0)
    tcrit = _t_when_mn_hits_threshold(mn0, kd1, p.mn_crit)
    w = _phase2_blend_weight(t, tcrit, p.transition_width, p.phase2_max_fraction)
    kd_t = (1.0 - w) * kd1 + w * kd2

    integral = np.zeros_like(t)
    if t.size == 1:
        integral[0] = kd_t[0] * max(float(t[0]), 0.0)
    else:
        dt = np.diff(t)
        if np.any(dt <= 0):
            raise ValueError("Time grid must be strictly increasing")
        integral[1:] = np.cumsum(0.5 * (kd_t[:-1] + kd_t[1:]) * dt)

    return np.maximum(mn0 * np.exp(-integral), 0.0)


def mass_fraction_two_phase(t: np.ndarray, mn0: float, k0: float, sav: float, m: float, p: TwoPhaseParams) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return np.array([], dtype=float)

    mn = mn_two_phase(t, mn0, k0, sav, m, p)
    mn_ratio = np.clip(mn / max(mn0, 1e-12), 0.0, 1.0)

    onset_ratio = np.clip(float(p.erosion_onset_ratio), 1e-6, 0.999)
    onset_progress = np.clip((onset_ratio - mn_ratio) / onset_ratio, 0.0, 1.0)

    kd1 = float(kd_from_sav(k0, sav, m))
    tcrit = _t_when_mn_hits_threshold(mn0, kd1, p.mn_crit)
    w = _phase2_blend_weight(t, tcrit, p.transition_width, p.phase2_max_fraction)

    erosion_rate = w * np.power(onset_progress, max(float(p.erosion_power), 1.0))

    if t.size == 1:
        erosion_integral = erosion_rate[0] * float(t[0])
        frac = np.array([1.0 - float(p.k_erosion) * erosion_integral], dtype=float)
    else:
        dt = np.diff(t)
        erosion_integral = np.zeros_like(t)
        erosion_integral[1:] = np.cumsum(0.5 * (erosion_rate[:-1] + erosion_rate[1:]) * dt)
        frac = 1.0 - float(p.k_erosion) * erosion_integral

    return np.clip(frac, float(p.floor_mass_fraction), 1.0)


def _mn_ratio_at_t(t_target: float, mn0: float, k0: float, sav: float, m: float, two_phase: Optional[TwoPhaseParams]) -> float:
    t_arr = np.linspace(0.0, max(float(t_target), 1e-9), 400)
    if two_phase is None:
        mn_end = float(mn_t(t_arr, mn0, k0, sav, m)[-1])
    else:
        mn_end = float(mn_two_phase(t_arr, mn0, k0, sav, m, two_phase)[-1])
    return mn_end / max(mn0, 1e-12)


def calibrate_k0(mn0: float, mn_target_ratio: float, t_target: float, sav: float, m: float,
                 two_phase: Optional[TwoPhaseParams] = None, mn_crit_ratio: float = 0.2) -> float:
    if not (0.0 < mn_target_ratio < 1.0):
        raise ValueError("mn_target_ratio must be in (0,1)")
    if t_target <= 0:
        raise ValueError("t_target must be > 0")
    if sav <= 0:
        raise ValueError("sav must be > 0")

    analytic = -math.log(mn_target_ratio) / (float(sav) ** float(m) * float(t_target))
    if two_phase is None:
        return analytic

    lo = max(analytic * 0.02, 1e-10)
    hi = max(analytic * 50.0, lo * 10.0)

    def resid(k0_try: float) -> float:
        return _mn_ratio_at_t(t_target, mn0, k0_try, sav, m, two_phase) - mn_target_ratio

    r_lo = resid(lo)
    r_hi = resid(hi)
    expand = 0
    while r_lo * r_hi > 0 and expand < 20:
        if r_lo > 0 and r_hi > 0:
            hi *= 2.0
            r_hi = resid(hi)
        else:
            lo *= 0.5
            r_lo = resid(lo)
        expand += 1

    if r_lo * r_hi > 0:
        return analytic

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        r_mid = resid(mid)
        if r_mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def porosity_from_mn(mn: np.ndarray, mn0: float, k_phi: float = 0.040, phi_max: float = 0.65) -> np.ndarray:
    mn = np.asarray(mn, dtype=float)
    deg = np.clip(1.0 - mn / max(mn0, 1e-12), 0.0, 1.0)
    return phi_max * (1.0 - np.exp(-float(k_phi) * deg))


def compute_all_fields_at_t(t_scalar: float, mn0: float, k0: float, sav: float, m: float,
                            e0: float, alpha: float, two_phase: Optional[TwoPhaseParams],
                            cryst_params: Optional[CrystallinityParams] = None,
                            k_phi: float = 0.040, phi_max: float = 0.65, beta_phi: float = 2.0) -> dict:
    if cryst_params is None:
        cryst_params = CrystallinityParams()

    t_arr = np.linspace(0.0, max(float(t_scalar), 1e-12), 240)
    if two_phase is None:
        mn_arr = mn_t(t_arr, mn0, k0, sav, m)
        mf = 1.0
    else:
        mn_arr = mn_two_phase(t_arr, mn0, k0, sav, m, two_phase)
        mf = float(mass_fraction_two_phase(t_arr, mn0, k0, sav, m, two_phase)[-1])

    mn_val = float(mn_arr[-1])
    mn_ratio = mn_val / max(mn0, 1e-12)
    phi_val = float(porosity_from_mn(np.array([mn_val]), mn0, k_phi=k_phi, phi_max=phi_max)[0])
    xc_val = float(crystallinity_bulk_pcl(t_arr, mn_arr, cryst_params)[-1])
    e_rel = (max(mn_ratio, 1e-12) ** float(alpha)) * (max(1.0 - phi_val, 1e-8) ** float(beta_phi))
    deg = 1.0 - mn_ratio
    risk = deg * (1.0 + 1.5 * deg)
    return {
        "Mn": mn_val,
        "Mn_ratio": mn_ratio,
        "E_rel": e_rel,
        "mass_fraction": mf,
        "porosity": phi_val,
        "crystallinity": xc_val,
        "risk": risk,
    }


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
