"""
Degradation and mechanical strength scaling model (first order chain scission & SA/V scaling)

Core (phase-1, pseudo-first-order chain scission; geometry via SA/V):
    kd = k0 * (SA/V)^m
    Mn(t) = Mn0 * exp(-kd * t)

Mechanical correlation (semi-crystalline polymer power law):
    E(t) = E0 * (Mn(t)/Mn0)^alpha
         = E0 * exp(-alpha * kd * t)

Optional Add-ons:
1) Two-phase behavior (hydrolysis -> erosion onset):
   - Phase 1: pseudo-first order Mn decay until Mn drops below a threshold Mn_crit
   - Phase 2: optional accelerated Mn decay + linear mass loss (surface erosion proxy)

2) Simple reaction–diffusion proxy for acidic/monomeric products (1D, perfect sink):
   dC/dt = d/dx ( D(t) dC/dx ) + S(t)
   - Perfect sink boundary: C(0,t)=C(L,t)=0
   - Diffusivity increases with degradation: D(t)=D0*(1 + beta*(1 - Mn/Mn0))

Notes:
- IMPORTANT: k0 depends on the units of time and SA/V. MAKE SURE TO CHECK!!! This depends on the cad file.
  If time is in weeks and SA/V in 1/mm, k0 has units: (weeks^-1) * (mm^m) This script treats inputs as self-consistent.
- Robin mass-transfer boundary is currently omitted (perfect sink is a decent in-vivo proxy).

Outputs (saved to --outdir):
    Mn_vs_t.png
    E_vs_t_geometries.png
    E_vs_SAV_fixed_times.png
    loglog_E_vs_Mn.png
    sensitivity_alpha_m.png
    E_surface_3d.png
    mass_fraction_vs_t.png               (two-phase only)
    Xc_vs_t.png
    Xc_vs_Mn.png
    crystallinity_summary.txt
    C_profile_heatmap.png                (rxn-diffusion only)
    gif_E_vs_SAV_over_time.gif
    gif_E_surface_rotate.gif
    gif_C_profile_over_time.gif          (rxn-diffusion only)

CAD / Geometry stuff
    I envision this to support mesh files that are readable by trimesh: STL, OBJ, PLY, GLB/GLTF, etc. Check the help page for the package to find more, since I'm not an expert at this haha
    For STEP/IGES you would apparently need a CAD kernel (OCC/cadquery) to tesselate into STL first. Check with Connie for output types. 


INSTALL! These are thepackages you would need.
    
    pip install numpy matplotlib trimesh pillow


OPTIONAL INSTALL (if it comes to reading CAD, since for now we can get whatever, but in the future, this could be a VERY, and I mean VERY useful tool for any kind of geometry-degradation-mechanical strength model.
    
    pip install "trimesh[easy]"
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# -----------------------
# Core kinetics + mechanics
# -----------------------

def kd_from_sav(k0: float, sav: float | np.ndarray, m: float) -> float | np.ndarray:
    """Compute effective degradation rate kd = k0*(SA/V)^m (supports scalar or array SA/V)."""
    kd = k0 * (np.asarray(sav, dtype=float) ** m)
    if np.any(kd < 0):
        raise ValueError("kd must be non-negative. Check k0, SA/V, and m.")
    return kd


def mn_t(t: np.ndarray, mn0: float, k0: float, sav: float, m: float) -> np.ndarray:
    """Number-average molecular weight over time (phase-1)."""
    kd = kd_from_sav(k0, sav, m)
    return mn0 * np.exp(-kd * t)


def e_t(t: np.ndarray, e0: float, k0: float, sav: float, m: float, alpha: float) -> np.ndarray:
    """Young's modulus over time from Mn scaling."""
    kd = kd_from_sav(k0, sav, m)
    return e0 * np.exp(-(alpha * kd) * t)

def e_t_with_crystallinity(
    t: np.ndarray,
    mn: np.ndarray,
    mn0: float,
    e0: float,
    xc: np.ndarray,
    xc0: float,
    alpha: float,
    bxc: float = 1.5,
) -> np.ndarray:
    """
    Modulus model for bulk semicrystalline PCL.
    Mn loss reduces modulus; increasing crystallinity can stiffen the material.
    """
    mn_term = np.maximum(mn / max(mn0, 1e-12), 1e-12) ** alpha
    xc_term = 1.0 + bxc * ((xc - xc0) / max(xc0, 1e-12))
    return e0 * mn_term * np.maximum(xc_term, 0.1)

def characteristic_time_fraction(f: float, k0: float, sav: float, m: float, alpha: float) -> float:
    """
    Time to reach E(t)/E0 = f.
        f = exp(-alpha*k0*(SA/V)^m*t)  ->  t = -ln(f)/(alpha*k0*(SA/V)^m)
    """
    if not (0.0 < f < 1.0):
        raise ValueError("f must be between 0 and 1 exclusive")
    denom = alpha * kd_from_sav(k0, sav, m)
    if denom <= 0:
        raise ValueError("alpha*k0*(SA/V)^m must be > 0")
    return -math.log(f) / denom


# -----------------------
# Two-phase add-on (erosion onset proxy)
# -----------------------

@dataclass(frozen=True)
class TwoPhaseParams:
    mn_crit: float          # threshold Mn (same units as mn0)
    accel_factor: float     # kd multiplier in 2nd phase (>=1) (proxy for autocatalysis/erosion onset)
    k_erosion: float        # linear mass loss rate after onset (in 1/time)
    floor_mass_fraction: float = 0.0


@dataclass(frozen=True)
class CrystallinityParams:
    """
    Bulk-PCL crystallinity proxy.

    xc0: initial crystallinity fraction of processed bulk PCL
    xc_max: upper cap for bulk PCL during degradation/reorganization
    mn_mid: Mn where crystallinity increase becomes active
    mn_width: smoothness of the Mn-trigger window
    a1/kc1/nav: primary crystallization / reorganization proxy
    a2/kc2: slower secondary crystallization proxy
    """
    xc0: float = 0.45
    xc_max: float = 0.70
    mn_mid: float = 20000.0
    mn_width: float = 5000.0
    a1: float = 0.12
    kc1: float = 0.8
    nav: float = 2.0
    a2: float = 0.06
    kc2: float = 0.08


def smooth_switch_from_mn(mn: np.ndarray, mn_mid: float, mn_width: float) -> np.ndarray:
    """Smooth 0->1 switch as Mn drops below mn_mid."""
    mn = np.asarray(mn, dtype=float)
    width = max(float(mn_width), 1e-12)
    z = (mn_mid - mn) / width
    return 1.0 / (1.0 + np.exp(-z))


def crystallinity_bulk_pcl(
    t: np.ndarray,
    mn: np.ndarray,
    p: CrystallinityParams,
) -> np.ndarray:
    """
    Bulk PCL crystallinity model:
    - primary reorganization / crystallization rises as Mn drops
    - slower secondary crystallization continues later
    """
    t = np.asarray(t, dtype=float)
    mn = np.asarray(mn, dtype=float)

    s = smooth_switch_from_mn(mn, p.mn_mid, p.mn_width)

    x_primary = p.a1 * (1.0 - np.exp(-p.kc1 * (s ** p.nav) * t))
    x_secondary = p.a2 * s * np.sqrt(np.maximum(t, 0.0)) / np.sqrt(1.0 / max(p.kc2, 1e-12))

    xc = p.xc0 + x_primary + x_secondary
    return np.clip(xc, 0.0, p.xc_max)


def diffusivity_with_crystallinity(
    D0: float,
    beta_D: float,
    mn: float,
    mn0: float,
    xc: float,
    xc0: float,
    gamma_xc: float = 1.0,
) -> float:
    """
    Diffusivity proxy: degradation can increase transport, while rising crystallinity can reduce it.
    """
    base = D0 * (1.0 + beta_D * (1.0 - mn / max(mn0, 1e-12)))
    xc_penalty = math.exp(-gamma_xc * max(xc - xc0, 0.0))
    return max(float(base * xc_penalty), 0.0)


def _t_when_mn_hits_threshold(mn0: float, kd: float, mn_crit: float) -> float:
    if mn_crit <= 0:
        raise ValueError("mn_crit must be > 0")
    if mn_crit >= mn0:
        return 0.0
    if kd <= 0:
        return float("inf")
    return math.log(mn0 / mn_crit) / kd


def mn_two_phase(t: np.ndarray, mn0: float, k0: float, sav: float, m: float, p: TwoPhaseParams) -> np.ndarray:
    """
    Piecewise Mn(t):
      - phase 1: Mn0 * exp(-kd*t) until Mn reaches mn_crit
      - phase 2: Mn_crit * exp(-kd2*(t - tcrit)), with kd2 = accel_factor * kd
    """
    kd = kd_from_sav(k0, sav, m)
    tcrit = _t_when_mn_hits_threshold(mn0, kd, p.mn_crit)
    kd2 = max(p.accel_factor, 1.0) * kd

    t = np.asarray(t, dtype=float)
    Mn = np.empty_like(t)

    mask1 = t <= tcrit
    Mn[mask1] = mn0 * np.exp(-kd * t[mask1])

    mask2 = ~mask1
    if np.any(mask2):
        Mn[mask2] = p.mn_crit * np.exp(-kd2 * (t[mask2] - tcrit))

    return Mn


def mass_fraction_two_phase(t: np.ndarray, mn0: float, k0: float, sav: float, m: float, p: TwoPhaseParams) -> np.ndarray:
    """
    Linear mass loss after onset (proxy for erosion/cell-mediated processes):
      - before onset: mass_fraction = 1
      - after onset:  mass_fraction = 1 - k_erosion*(t - tcrit), clipped
    """
    kd = kd_from_sav(k0, sav, m)
    tcrit = _t_when_mn_hits_threshold(mn0, kd, p.mn_crit)

    t = np.asarray(t, dtype=float)
    frac = np.ones_like(t)

    mask = t >= tcrit
    if np.any(mask) and math.isfinite(tcrit):
        frac[mask] = 1.0 - p.k_erosion * (t[mask] - tcrit)

    return np.clip(frac, p.floor_mass_fraction, 1.0)


# -----------------------
# Reaction–diffusion add-on (1D, perfect sink)
# -----------------------

@dataclass(frozen=True)
class RxnDiff1DParams:
    length: float           # L
    nx: int                 # num of grid points
    D0: float               # baseline diffusivity
    beta_D: float           # D(t)=D0*(1+beta_D*(1 - Mn/Mn0))
    s0: float               # source scaling
    use_two_phase_mn: bool  # if True then source uses two-phase Mn(t)
    safety: float = 0.45    # Stability safety factor for explicit scheme



def simulate_rxn_diff_1d(
    t: np.ndarray,
    mn0: float,
    k0: float,
    sav: float,
    m: float,
    params: RxnDiff1DParams,
    two_phase: Optional[TwoPhaseParams] = None,
    cryst_params: Optional[CrystallinityParams] = None,
    gamma_xc: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D reaction–diffusion simulation for:
        dC/dt = d/dx( D(t) dC/dx ) + S(t)
    with perfect-sink (Dirichlet) boundaries:
        C(0,t) = C(L,t) = 0

    Implementation notes:
    - Uses a backward-Euler (implicit) diffusion step for numerical stability.
      (Explicit FD is very easy to destabilize for practical dt/dx and growing D(t).)
    - D(t) is treated as spatially uniform but time-varying.
    - Source term S(t) is spatially uniform and proportional to scission rate magnitude.

    Returns:
        x: (nx,) spatial grid
        C_hist: (nt, nx) concentration over time
    """
    t = np.asarray(t, dtype=float)
    nt = int(t.size)
    if nt < 2:
        raise ValueError("Need at least 2 time points for reaction-diffusion simulation.")
    if params.nx < 3:
        raise ValueError("nx must be >= 3.")
    if params.length <= 0:
        raise ValueError("length must be positive.")

    x = np.linspace(0.0, params.length, params.nx, dtype=float)
    dx = float(x[1] - x[0])

    if params.use_two_phase_mn:
        if two_phase is None:
            raise ValueError("two_phase parameters must be provided when use_two_phase_mn=True.")
        Mn = mn_two_phase(t, mn0, k0, sav, m, two_phase)
    else:
        Mn = mn_t(t, mn0, k0, sav, m)

    if cryst_params is None:
        cryst_params = CrystallinityParams()
    Xc = crystallinity_bulk_pcl(t, Mn, cryst_params)

    dMn_dt = np.gradient(Mn, t)

    S_t = params.s0 * np.maximum(0.0, -dMn_dt) / max(mn0, 1e-12)

    C = np.zeros(params.nx, dtype=float)
    C_hist = np.zeros((nt, params.nx), dtype=float)
    n_int = params.nx - 2

    def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Solve tridiagonal system:
            a[i-1]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
        where:
            a has shape (n-1,), b has shape (n,), c has shape (n-1,)
        """
        n = b.size
        ac, bc, cc, dc = a.astype(float).copy(), b.astype(float).copy(), c.astype(float).copy(), d.astype(float).copy()
        for i in range(1, n):
            w = ac[i-1] / bc[i-1]
            bc[i] -= w * cc[i-1]
            dc[i] -= w * dc[i-1]
        x_out = np.empty(n, dtype=float)
        x_out[-1] = dc[-1] / bc[-1]
        for i in range(n - 2, -1, -1):
            x_out[i] = (dc[i] - cc[i] * x_out[i + 1]) / bc[i]
        return x_out

    n_int = params.nx - 2
    if n_int <= 0:
        return x, C_hist

    for i in range(nt):
        C_hist[i] = C

        if i == nt - 1:
            break

        dt_i = float(t[i + 1] - t[i])
        if dt_i <= 0:
            raise ValueError("Time grid must be strictly increasing.")

        # time-varying diffusivity (spatially uniform proxy)
        D = diffusivity_with_crystallinity(
            D0=params.D0,
            beta_D=params.beta_D,
            mn=float(Mn[i]),
            mn0=mn0,
            xc=float(Xc[i]),
            xc0=cryst_params.xc0,
            gamma_xc=gamma_xc,
        )

        # Backward Euler:
        #   C^{n+1} - C^n = dt*(D*Laplacian(C^{n+1}) + S_t[i])
        # interior: -r*C_{j-1}^{n+1} + (1+2r)*C_j^{n+1} - r*C_{j+1}^{n+1} = C_j^n + dt*S
        r = D * dt_i / (dx * dx) if D > 0 else 0.0

        # tri-diagonal for interior unknowns
        if n_int == 1:
            b = np.array([1.0 + 2.0 * r], dtype=float)
            rhs = np.array([C[1] + dt_i * S_t[i]], dtype=float)
            C1 = rhs[0] / b[0] if b[0] != 0 else 0.0
            C_new = C.copy()
            C_new[0] = 0.0
            C_new[-1] = 0.0
            C_new[1] = max(C1, 0.0)
            C = C_new
            continue

        a = (-r) * np.ones(n_int - 1, dtype=float)
        b = (1.0 + 2.0 * r) * np.ones(n_int, dtype=float)
        c = (-r) * np.ones(n_int - 1, dtype=float)

        rhs = C[1:-1].copy()
        rhs += dt_i * S_t[i]


        C_int = _thomas(a, b, c, rhs)

        C_new = C.copy()
        C_new[0] = 0.0
        C_new[-1] = 0.0
        C_new[1:-1] = np.maximum(C_int, 0.0)
        C = C_new

    return x, C_hist

# -------------------
# Crystallinity add-on
# -------------------

@dataclass(frozen=True)
class CrystallinityParams:
    xc0: float = 0.45          # initial bulk crystallinity fraction
    xc_max: float = 0.70       # upper bound for bulk PCL
    mn_mid: float = 20000.0    # Mn where crystallinity increase becomes active
    mn_width: float = 5000.0   # smoothness of Mn-trigger
    a1: float = 0.12           # primary crystallinity gain
    kc1: float = 0.8           # primary crystallization/reorg rate
    nav: float = 1.24          # Avg Avrami-like exponent for bulk-PCL https://doi.org/10.3390/polym15143013
    a2: float = 0.06           # secondary crystallization gain
    kc2: float = 0.08          # secondary scaling

def smooth_switch_from_mn(mn: np.ndarray, mn_mid: float, mn_width: float) -> np.ndarray:
    """
    Returns a smooth 0->1 switch as Mn drops below mn_mid.
    High Mn: near 0
    Low Mn: near 1
    """
    z = (mn_mid - mn) / max(mn_width, 1e-12)
    return 1.0 / (1.0 + np.exp(-z))


def crystallinity_bulk_pcl(
    t: np.ndarray,
    mn: np.ndarray,
    p: CrystallinityParams,
) -> np.ndarray:
    """
    Bulk PCL crystallinity model:
    - primary reorganization / crystallization turns on as Mn drops
    - secondary crystallization continues more slowly afterward
    """
    t = np.asarray(t, dtype=float)
    mn = np.asarray(mn, dtype=float)

    s = smooth_switch_from_mn(mn, p.mn_mid, p.mn_width)

    # primary contribution: Avrami-like saturation
    x_primary = p.a1 * (1.0 - np.exp(-p.kc1 * (s ** p.nav) * t))

    # secondary contribution: slower concurrent growth
    x_secondary = p.a2 * s * np.sqrt(np.maximum(t, 0.0)) / np.sqrt(1.0 / max(p.kc2, 1e-12))

    xc = p.xc0 + x_primary + x_secondary
    return np.clip(xc, 0.0, p.xc_max)

# -------------------
# Geometry helper (optional trimesh)
# -------------------

def load_mesh_compute_sav(path: str) -> Tuple[float, float, float]:
    """
    Load a mesh and compute:
        SA = surface area
        V  = volume (reliable only for watertight meshes)
        SA/V

    Returns: (SA, V, SA/V)

    Units:
        SA/V is 1/length using the mesh's length unit. Keep this consistent with k0.
    """
    try:
        import trimesh  #lazy import so the rest of the script works without trimesh temp
    except Exception as e:
        raise ImportError(
            "trimesh is required for --cad. Install it in a venv (recommended) or via your environment manager."
        ) from e

    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {path}")

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    sa = float(mesh.area)

    if not mesh.is_watertight:
        print("[WARN] Mesh is not watertight; volume may be inaccurate. "
              "Export a closed/solid STL if possible.")

    vol = float(mesh.volume)
    if vol <= 0:
        raise ValueError(
            f"Computed volume is not positive ({vol}). Mesh may not be watertight/solid."
        )

    sav = sa / vol
    return sa, vol, sav


def compute_mn_curve(
    t: np.ndarray,
    mn0: float,
    k0: float,
    sav: float,
    m: float,
    two_phase: Optional[TwoPhaseParams] = None,
) -> np.ndarray:
    if two_phase is None:
        return mn_t(t, mn0, k0, sav, m)
    return mn_two_phase(t, mn0, k0, sav, m, two_phase)


def compute_xc_curve(
    t: np.ndarray,
    mn0: float,
    k0: float,
    sav: float,
    m: float,
    cryst_params: CrystallinityParams,
    two_phase: Optional[TwoPhaseParams] = None,
) -> np.ndarray:
    mn = compute_mn_curve(t, mn0, k0, sav, m, two_phase=two_phase)
    return crystallinity_bulk_pcl(t, mn, cryst_params)


def compute_e_curve(
    t: np.ndarray,
    mn0: float,
    e0: float,
    k0: float,
    sav: float,
    m: float,
    alpha: float,
    cryst_params: Optional[CrystallinityParams] = None,
    two_phase: Optional[TwoPhaseParams] = None,
    bxc: float = 1.5,
) -> np.ndarray:
    if cryst_params is None:
        return e_t(t, e0, k0, sav, m, alpha)
    mn = compute_mn_curve(t, mn0, k0, sav, m, two_phase=two_phase)
    xc = crystallinity_bulk_pcl(t, mn, cryst_params)
    return e_t_with_crystallinity(t, mn, mn0, e0, xc, cryst_params.xc0, alpha, bxc=bxc)


# ----------------------
# Plot helpers
# ----------------------

def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def savefig(path: str, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def make_gif(frames: List[str], out_gif: str, duration_ms: int = 120) -> None:
    imgs = [Image.open(p).convert("RGBA") for p in frames]
    imgs[0].save(
        out_gif,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


# -----------------------------
# Main plotting functions
# -----------------------------

def plot_Mn_vs_t(
    outdir: str,
    t: np.ndarray,
    mn0: float,
    k0: float,
    m: float,
    sav_list: List[float],
    labels: Optional[List[str]] = None,
    two_phase: Optional[TwoPhaseParams] = None,
    time_label: str = "time",
) -> str:
    plt.figure()
    for i, sav in enumerate(sav_list):
        if two_phase is None:
            y = mn_t(t, mn0, k0, sav, m)
        else:
            y = mn_two_phase(t, mn0, k0, sav, m, two_phase)
        lab = labels[i] if labels and i < len(labels) else f"SA/V={sav:.3g}"
        plt.plot(t, y, label=lab)

    plt.xlabel(time_label)
    plt.ylabel("Mn(t)")
    plt.title("Molecular weight vs time")
    plt.legend()
    outpath = os.path.join(outdir, "Mn_vs_t.png")
    savefig(outpath)
    return outpath


def plot_mass_fraction_vs_t(
    outdir: str,
    t: np.ndarray,
    mn0: float,
    k0: float,
    m: float,
    sav: float,
    two_phase: TwoPhaseParams,
    time_label: str = "time",
) -> str:
    frac = mass_fraction_two_phase(t, mn0, k0, sav, m, two_phase)

    plt.figure()
    plt.plot(t, frac)
    plt.xlabel(time_label)
    plt.ylabel("mass fraction (proxy)")
    plt.title("Linear mass loss after Mn threshold (two-phase proxy)")
    outpath = os.path.join(outdir, "mass_fraction_vs_t.png")
    savefig(outpath)
    return outpath


def plot_E_vs_t_geometries(
    outdir: str,
    t: np.ndarray,
    mn0: float,
    e0: float,
    k0: float,
    m: float,
    alpha: float,
    sav_list: List[float],
    labels: Optional[List[str]] = None,
    cryst_params: Optional[CrystallinityParams] = None,
    two_phase: Optional[TwoPhaseParams] = None,
    bxc: float = 1.5,
    time_label: str = "time",
) -> str:
    plt.figure()
    for i, sav in enumerate(sav_list):
        y = compute_e_curve(
            t=t, mn0=mn0, e0=e0, k0=k0, sav=sav, m=m, alpha=alpha,
            cryst_params=cryst_params, two_phase=two_phase, bxc=bxc
        )
        lab = labels[i] if labels and i < len(labels) else f"SA/V={sav:.3g}"
        plt.plot(t, y, label=lab)

    plt.xlabel(time_label)
    plt.ylabel("E(t)")
    title = "Young's modulus vs time for different geometries (SA/V)"
    if cryst_params is not None:
        title += "\n(with bulk-PCL crystallinity)"
    plt.title(title)
    plt.legend()
    outpath = os.path.join(outdir, "E_vs_t_geometries.png")
    savefig(outpath)
    return outpath


def plot_E_vs_SAV_fixed_times(
    outdir: str,
    sav_grid: np.ndarray,
    times: List[float],
    mn0: float,
    e0: float,
    k0: float,
    m: float,
    alpha: float,
    cryst_params: Optional[CrystallinityParams] = None,
    two_phase: Optional[TwoPhaseParams] = None,
    bxc: float = 1.5,
) -> str:
    plt.figure()
    for tt in times:
        y = np.array([
            compute_e_curve(
                t=np.array([tt]), mn0=mn0, e0=e0, k0=k0, sav=float(sav), m=m, alpha=alpha,
                cryst_params=cryst_params, two_phase=two_phase, bxc=bxc
            )[0]
            for sav in sav_grid
        ], dtype=float)
        plt.plot(sav_grid, y, label=f"t={tt:g}")

    plt.xlabel("SA/V")
    plt.ylabel("E(t)")
    plt.title("Young's modulus vs SA/V at fixed times")
    plt.legend()
    outpath = os.path.join(outdir, "E_vs_SAV_fixed_times.png")
    savefig(outpath)
    return outpath


def plot_loglog_E_vs_Mn(
    outdir: str,
    t: np.ndarray,
    mn0: float,
    e0: float,
    k0: float,
    sav: float,
    m: float,
    alpha: float,
    cryst_params: Optional[CrystallinityParams] = None,
    two_phase: Optional[TwoPhaseParams] = None,
    bxc: float = 1.5,
) -> str:
    Mn = compute_mn_curve(t, mn0, k0, sav, m, two_phase=two_phase)
    E = compute_e_curve(t, mn0, e0, k0, sav, m, alpha, cryst_params=cryst_params, two_phase=two_phase, bxc=bxc)

    x = Mn / mn0
    y = E / e0

    eps = 1e-16
    x = np.clip(x, eps, None)
    y = np.clip(y, eps, None)

    plt.figure()
    plt.loglog(x, y, marker="o", linewidth=1)

    lx = np.log(x)
    ly = np.log(y)
    slope = float(np.polyfit(lx, ly, 1)[0])

    plt.xlabel("Mn/Mn0 (log)")
    plt.ylabel("E/E0 (log)")
    title = f"Log-log check of E ~ Mn^alpha (fit slope ≈ {slope:.3g})"
    if cryst_params is not None:
        title += "\n(with crystallinity-coupled modulus)"
    plt.title(title)

    outpath = os.path.join(outdir, "loglog_E_vs_Mn.png")
    savefig(outpath)
    return outpath


def plot_sensitivity_alpha_m(
    outdir: str,
    tfinal: float,
    sav: float,
    k0: float,
    alpha_range: Tuple[float, float],
    m_range: Tuple[float, float],
    n: int = 60,
) -> str:
    alphas = np.linspace(alpha_range[0], alpha_range[1], n)
    ms = np.linspace(m_range[0], m_range[1], n)

    A, M = np.meshgrid(alphas, ms)
    frac = np.exp(-(A * k0 * (sav ** M)) * tfinal)

    plt.figure()
    im = plt.imshow(
        frac,
        origin="lower",
        aspect="auto",
        extent=[alphas[0], alphas[-1], ms[0], ms[-1]],
    )
    plt.colorbar(im, label="E(tfinal)/E0")
    plt.xlabel("alpha")
    plt.ylabel("m")
    plt.title(f"Sensitivity of E fraction at t={tfinal:g} (SA/V={sav:.3g})")

    outpath = os.path.join(outdir, "sensitivity_alpha_m.png")
    savefig(outpath)
    return outpath


def plot_E_surface_3d(
    outdir: str,
    t_grid: np.ndarray,
    sav_grid: np.ndarray,
    mn0: float,
    e0: float,
    k0: float,
    m: float,
    alpha: float,
    cryst_params: Optional[CrystallinityParams] = None,
    two_phase: Optional[TwoPhaseParams] = None,
    bxc: float = 1.5,
) -> str:
    T, S = np.meshgrid(t_grid, sav_grid)
    frac = np.zeros_like(T, dtype=float)
    for row, sav in enumerate(sav_grid):
        frac[row, :] = compute_e_curve(
            t=t_grid, mn0=mn0, e0=e0, k0=k0, sav=float(sav), m=m, alpha=alpha,
            cryst_params=cryst_params, two_phase=two_phase, bxc=bxc
        ) / max(e0, 1e-12)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T, S, frac, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.set_xlabel("time")
    ax.set_ylabel("SA/V")
    ax.set_zlabel("E/E0")
    ax.set_title("Degradation surface: E/E0 over time and SA/V")

    outpath = os.path.join(outdir, "E_surface_3d.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def gif_E_vs_SAV_over_time(
    outdir: str,
    sav_grid: np.ndarray,
    t_steps: np.ndarray,
    mn0: float,
    e0: float,
    k0: float,
    m: float,
    alpha: float,
    cryst_params: Optional[CrystallinityParams] = None,
    two_phase: Optional[TwoPhaseParams] = None,
    bxc: float = 1.5,
) -> str:
    frames = []
    tmpdir = os.path.join(outdir, "_frames_E_vs_SAV")
    os.makedirs(tmpdir, exist_ok=True)

    for i, tt in enumerate(t_steps):
        plt.figure()
        y = np.array([
            compute_e_curve(
                t=np.array([tt]), mn0=mn0, e0=e0, k0=k0, sav=float(sav), m=m, alpha=alpha,
                cryst_params=cryst_params, two_phase=two_phase, bxc=bxc
            )[0]
            for sav in sav_grid
        ], dtype=float)
        plt.plot(sav_grid, y)
        plt.xlabel("SA/V")
        plt.ylabel("E(t)")
        plt.title(f"E vs SA/V (time={tt:g})")
        frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
        savefig(frame_path, dpi=140)
        frames.append(frame_path)

    out_gif = os.path.join(outdir, "gif_E_vs_SAV_over_time.gif")
    make_gif(frames, out_gif, duration_ms=120)
    return out_gif


def gif_E_surface_rotate(
    outdir: str,
    t_grid: np.ndarray,
    sav_grid: np.ndarray,
    mn0: float,
    e0: float,
    k0: float,
    m: float,
    alpha: float,
    cryst_params: Optional[CrystallinityParams] = None,
    two_phase: Optional[TwoPhaseParams] = None,
    bxc: float = 1.5,
    n_frames: int = 48,
) -> str:
    T, S = np.meshgrid(t_grid, sav_grid)
    frac = np.zeros_like(T, dtype=float)
    for row, sav in enumerate(sav_grid):
        frac[row, :] = compute_e_curve(
            t=t_grid, mn0=mn0, e0=e0, k0=k0, sav=float(sav), m=m, alpha=alpha,
            cryst_params=cryst_params, two_phase=two_phase, bxc=bxc
        ) / max(e0, 1e-12)

    frames = []
    tmpdir = os.path.join(outdir, "_frames_surface_rotate")
    os.makedirs(tmpdir, exist_ok=True)

    for i in range(n_frames):
        angle = 360.0 * i / n_frames

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(T, S, frac, rstride=1, cstride=1, linewidth=0, antialiased=True)

        ax.set_xlabel("time")
        ax.set_ylabel("SA/V")
        ax.set_zlabel("E/E0")
        ax.set_title("E/E0 surface (rotating view)")
        ax.view_init(elev=25, azim=angle)

        frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=140)
        plt.close(fig)
        frames.append(frame_path)

    out_gif = os.path.join(outdir, "gif_E_surface_rotate.gif")
    make_gif(frames, out_gif, duration_ms=110)
    return out_gif


def plot_C_profile_heatmap(
    outdir: str,
    x: np.ndarray,
    t: np.ndarray,
    C_hist: np.ndarray,
) -> str:
    plt.figure()
    # show as time (rows) vs position (cols)
    im = plt.imshow(
        C_hist,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
    )
    plt.colorbar(im, label="C(x,t)")
    plt.xlabel("x")
    plt.ylabel("time")
    plt.title("Reaction–diffusion (1D) concentration history")
    outpath = os.path.join(outdir, "C_profile_heatmap.png")
    savefig(outpath)
    return outpath


def gif_C_profile_over_time(
    outdir: str,
    x: np.ndarray,
    t: np.ndarray,
    C_hist: np.ndarray,
    n_frames: int = 60,
) -> str:
    frames = []
    tmpdir = os.path.join(outdir, "_frames_C_profile")
    os.makedirs(tmpdir, exist_ok=True)

    idxs = np.linspace(0, len(t) - 1, n_frames).astype(int)
    for j, i in enumerate(idxs):
        plt.figure()
        plt.plot(x, C_hist[i])
        plt.xlabel("x")
        plt.ylabel("C(x,t)")
        plt.title(f"C profile (t={t[i]:g})")
        frame_path = os.path.join(tmpdir, f"frame_{j:04d}.png")
        savefig(frame_path, dpi=140)
        frames.append(frame_path)

    out_gif = os.path.join(outdir, "gif_C_profile_over_time.gif")
    make_gif(frames, out_gif, duration_ms=120)
    return out_gif


def plot_Xc_vs_t(
    outdir: str,
    t: np.ndarray,
    mn0: float,
    k0: float,
    m: float,
    sav_list: List[float],
    cryst_params: CrystallinityParams,
    labels: Optional[List[str]] = None,
    two_phase: Optional[TwoPhaseParams] = None,
    time_label: str = "time",
) -> str:
    plt.figure()
    for i, sav in enumerate(sav_list):
        xc = compute_xc_curve(t, mn0, k0, sav, m, cryst_params, two_phase=two_phase)
        lab = labels[i] if labels and i < len(labels) else f"SA/V={sav:.3g}"
        plt.plot(t, xc, label=lab)

    plt.xlabel(time_label)
    plt.ylabel("Xc(t)")
    plt.title("Bulk-PCL crystallinity vs time")
    plt.legend()
    outpath = os.path.join(outdir, "Xc_vs_t.png")
    savefig(outpath)
    return outpath


def plot_Xc_vs_Mn(
    outdir: str,
    t: np.ndarray,
    mn0: float,
    k0: float,
    m: float,
    sav: float,
    cryst_params: CrystallinityParams,
    two_phase: Optional[TwoPhaseParams] = None,
) -> str:
    mn = compute_mn_curve(t, mn0, k0, sav, m, two_phase=two_phase)
    xc = crystallinity_bulk_pcl(t, mn, cryst_params)

    plt.figure()
    plt.plot(mn, xc)
    plt.xlabel("Mn(t)")
    plt.ylabel("Xc(t)")
    plt.title("Bulk-PCL crystallinity vs molecular weight")
    outpath = os.path.join(outdir, "Xc_vs_Mn.png")
    savefig(outpath)
    return outpath


def write_crystallinity_summary(
    outdir: str,
    t: np.ndarray,
    mn0: float,
    k0: float,
    m: float,
    sav_list: List[float],
    cryst_params: CrystallinityParams,
    two_phase: Optional[TwoPhaseParams] = None,
) -> str:
    outpath = os.path.join(outdir, "crystallinity_summary.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("Bulk PCL crystallinity summary\n")
        f.write(f"xc0={cryst_params.xc0}\n")
        f.write(f"xc_max={cryst_params.xc_max}\n")
        f.write(f"mn_mid={cryst_params.mn_mid}\n")
        f.write(f"mn_width={cryst_params.mn_width}\n")
        f.write(f"a1={cryst_params.a1}, kc1={cryst_params.kc1}, nav={cryst_params.nav}\n")
        f.write(f"a2={cryst_params.a2}, kc2={cryst_params.kc2}\n\n")
        for sav in sav_list:
            mn = compute_mn_curve(t, mn0, k0, sav, m, two_phase=two_phase)
            xc = crystallinity_bulk_pcl(t, mn, cryst_params)
            f.write(
                f"SA/V={sav:.6g}: Xc(t=0)={xc[0]:.6g}, Xc(t=end)={xc[-1]:.6g}, Mn(end)={mn[-1]:.6g}\n"
            )
    return outpath


# -----------------------------
# CLI
# -----------------------------

def parse_floats(csv: str) -> List[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="PCL degradation + mechanics scaling model (SA/V).")

    ap.add_argument("--cad", type=str, default=None,
                    help="Path to mesh CAD file (STL/OBJ/PLY/GLB...). If omitted, use --sav-list.")
    ap.add_argument("--sav-list", type=str, default="1.4,2,3",
                    help="Comma-separated SA/V values (used if --cad not provided).")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory.")

    # model selection
    ap.add_argument("--mode", type=str, default="base",
                    choices=["base", "two_phase", "rxn_diff1d"],
                    help="Which add-ons to enable.")

    # core params
    ap.add_argument("--mn0", type=float, default=60000.0, help="Initial Mn (e.g., g/mol).")
    ap.add_argument("--e0", type=float, default=1000.0, help="Initial E (e.g., MPa).")
    ap.add_argument("--k0", type=float, default=0.45, #for 80-90% Mn decrease in 2-3 years
                    help="Material/environment constant in kd = k0*(SA/V)^m (units depend on your choices).")
    ap.add_argument("--m", type=float, default=1.0, help="Geometry exponent (m≈1 surface-controlled; m≈0 bulk).")
    ap.add_argument("--alpha", type=float, default=2.0, help="E~Mn^alpha scaling exponent (often 1-3).")
    ap.add_argument("--bxc", type=float, default=1.5, help="Strength of crystallinity stiffening in modulus model.")

    # crystallinity params (bulk PCL)
    ap.add_argument("--xc0", type=float, default=0.45, help="Initial bulk-PCL crystallinity fraction.")
    ap.add_argument("--xc-max", type=float, default=0.70, help="Maximum bulk-PCL crystallinity fraction.")
    ap.add_argument("--mn-mid", type=float, default=20000.0, help="Mn where crystallinity increase becomes active.")
    ap.add_argument("--mn-width", type=float, default=5000.0, help="Width of the Mn-trigger window for crystallinity.")
    ap.add_argument("--xc-a1", type=float, default=0.12, help="Primary crystallinity gain.")
    ap.add_argument("--xc-kc1", type=float, default=0.8, help="Primary crystallization/reorganization rate.")
    ap.add_argument("--xc-nav", type=float, default=2.0, help="Avrami-like exponent for primary crystallinity growth.")
    ap.add_argument("--xc-a2", type=float, default=0.06, help="Secondary crystallinity gain.")
    ap.add_argument("--xc-kc2", type=float, default=0.08, help="Secondary crystallization scaling.")
    ap.add_argument("--gamma-xc", type=float, default=1.0, help="How strongly rising crystallinity reduces diffusivity.")

    # Time axis (years scale for PCL degradation)
    ap.add_argument("--tmax", type=float, default=5.0, help="Max time for plots in years.")
    ap.add_argument("--nt", type=int, default=300, help="Number of time points.")
    ap.add_argument("--time-label", type=str, default="time (years)",
                    help="X-axis label.")

    # E vs SA/V at fixed times
    ap.add_argument("--fixed-times", type=str, default="0,0.5,1,2,3,4",
                    help="Comma-separated times to plot E vs SA/V.")
    ap.add_argument("--sav-min", type=float, default=1.0, help="Min SA/V for grids.")
    ap.add_argument("--sav-max", type=float, default=5.0, help="Max SA/V for grids.")
    ap.add_argument("--nsav", type=int, default=200, help="Number of SA/V points in grids.")

    # Sensitivity
    ap.add_argument("--sens-tfinal", type=float, default=3.0, help="Time for sensitivity heatmap.")
    ap.add_argument("--sens-alpha", type=str, default="1,3", help="alpha range 'min,max' for sensitivity.")
    ap.add_argument("--sens-m", type=str, default="0,2", help="m range 'min,max' for sensitivity.")
    ap.add_argument("--sens-n", type=int, default=60, help="Resolution for sensitivity grid.")

    # Characteristic time fraction
    ap.add_argument("--fraction", type=float, default=0.5, help="Fraction f for characteristic time (E/E0=f).")

    # GIF controls
    ap.add_argument("--gif-frames", type=int, default=60, help="Number of frames for time-progression GIF.")

    # two-phase params
    ap.add_argument("--mn-crit", type=float, default=8000.0,
                    help="Mn threshold for onset (two-phase mode).")
    ap.add_argument("--accel-factor", type=float, default=3.0,
                    help="Multiplier on kd after onset (two-phase mode).")
    ap.add_argument("--k-erosion", type=float, default=0.002,
                    help="Linear mass-loss rate after onset, in 1/time (two-phase mode).")

    # rxn-diffusion params
    ap.add_argument("--L", type=float, default=1.0, help="1D length for diffusion domain.")
    ap.add_argument("--nx", type=int, default=101, help="Grid points for diffusion domain.")
    ap.add_argument("--D0", type=float, default=1e-3, help="Baseline diffusivity.")
    ap.add_argument("--beta-D", type=float, default=5.0, help="Diffusivity growth factor vs degradation.")
    ap.add_argument("--s0", type=float, default=1.0, help="Source scaling for degradation products.")
    ap.add_argument("--rxn-use-two-phase-mn", action="store_true",
                    help="Use two-phase Mn(t) for the diffusion source term.")

    args = ap.parse_args()
    ensure_outdir(args.outdir)

    # determine SA/V values
    sav_values: List[float] = []
    sav_labels: List[str] = []

    if args.cad is not None:
        sa, vol, sav = load_mesh_compute_sav(args.cad)
        sav_values = [sav]
        sav_labels = [f"{os.path.basename(args.cad)} (SA/V={sav:.3g})"]
        with open(os.path.join(args.outdir, "geometry_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"CAD: {args.cad}\n")
            f.write(f"Surface area (SA): {sa}\n")
            f.write(f"Volume (V): {vol}\n")
            f.write(f"SA/V: {sav}\n")
    else:
        sav_values = parse_floats(args.sav_list)
        sav_labels = [f"SA/V={v:.3g}" for v in sav_values]

    # vectors
    t = np.linspace(0.0, args.tmax, args.nt)

    # build optional params objects
    cryst_params = CrystallinityParams(
        xc0=args.xc0,
        xc_max=args.xc_max,
        mn_mid=args.mn_mid,
        mn_width=args.mn_width,
        a1=args.xc_a1,
        kc1=args.xc_kc1,
        nav=args.xc_nav,
        a2=args.xc_a2,
        kc2=args.xc_kc2,
    )

    two_phase = None
    if args.mode in ("two_phase", "rxn_diff1d") or args.rxn_use_two_phase_mn:
        two_phase = TwoPhaseParams(
            mn_crit=args.mn_crit,
            accel_factor=args.accel_factor,
            k_erosion=args.k_erosion,
        )

    # Diagnostics: report whether two-phase onset is reached for the first SA/V
    if args.mode == "two_phase" and two_phase is not None:
        kd0 = kd_from_sav(args.k0, sav_values[0], args.m)
        tcrit0 = _t_when_mn_hits_threshold(args.mn0, kd0, args.mn_crit)
        mn_end = float(mn_two_phase(np.array([args.tmax]), args.mn0, args.k0, sav_values[0], args.m, two_phase)[0])
        print(f"[INFO] two-phase: SA/V={sav_values[0]:.6g}, kd={kd0:.6g}, Mn_crit={args.mn_crit:g}, tcrit={tcrit0:.6g}, Mn(tmax)={mn_end:.6g}")
        if (not np.isfinite(tcrit0)) or (tcrit0 > args.tmax):
            print("[INFO] two-phase onset NOT reached within tmax; outputs will match base model. Increase --tmax, --k0, SA/V, or raise --mn-crit.")

    # --- Plots: Mn + E
    plot_Mn_vs_t(
        outdir=args.outdir,
        t=t,
        mn0=args.mn0,
        k0=args.k0,
        m=args.m,
        sav_list=sav_values,
        labels=sav_labels,
        two_phase=two_phase if args.mode in ("two_phase", "rxn_diff1d") else None,
        time_label=args.time_label,
    )
    # relabel time axes in saved plots (simple approach: set global label via rcParams is messy;
    # instead, we just overwrite xlabel in key plots by replotting? low priority. Keep label via args.)
    # For now, we keep "time" in plots; user can edit labels if needed.

    plot_E_vs_t_geometries(
        outdir=args.outdir,
        t=t,
        mn0=args.mn0,
        e0=args.e0,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
        sav_list=sav_values,
        labels=sav_labels,
        cryst_params=cryst_params,
        two_phase=two_phase if args.mode in ("two_phase", "rxn_diff1d") else None,
        bxc=args.bxc,
        time_label=args.time_label,
    )

    plot_Xc_vs_t(
        outdir=args.outdir,
        t=t,
        mn0=args.mn0,
        k0=args.k0,
        m=args.m,
        sav_list=sav_values,
        cryst_params=cryst_params,
        labels=sav_labels,
        two_phase=two_phase if args.mode in ("two_phase", "rxn_diff1d") else None,
        time_label=args.time_label,
    )
    plot_Xc_vs_Mn(
        outdir=args.outdir,
        t=t,
        mn0=args.mn0,
        k0=args.k0,
        m=args.m,
        sav=sav_values[0],
        cryst_params=cryst_params,
        two_phase=two_phase if args.mode in ("two_phase", "rxn_diff1d") else None,
    )
    write_crystallinity_summary(
        outdir=args.outdir,
        t=t,
        mn0=args.mn0,
        k0=args.k0,
        m=args.m,
        sav_list=sav_values,
        cryst_params=cryst_params,
        two_phase=two_phase if args.mode in ("two_phase", "rxn_diff1d") else None,
    )

    sav_grid = np.linspace(args.sav_min, args.sav_max, args.nsav)
    fixed_times = parse_floats(args.fixed_times)
    plot_E_vs_SAV_fixed_times(
        outdir=args.outdir,
        sav_grid=sav_grid,
        times=fixed_times,
        mn0=args.mn0,
        e0=args.e0,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
        cryst_params=cryst_params,
        two_phase=two_phase if args.mode in ("two_phase", "rxn_diff1d") else None,
        bxc=args.bxc,
    )

    plot_loglog_E_vs_Mn(
        outdir=args.outdir,
        t=t,
        mn0=args.mn0,
        e0=args.e0,
        k0=args.k0,
        sav=sav_values[0],
        m=args.m,
        alpha=args.alpha,
        cryst_params=cryst_params,
        two_phase=two_phase if args.mode in ("two_phase", "rxn_diff1d") else None,
        bxc=args.bxc,
    )

    alpha_min, alpha_max = parse_floats(args.sens_alpha)
    m_min, m_max = parse_floats(args.sens_m)
    plot_sensitivity_alpha_m(
        outdir=args.outdir,
        tfinal=args.sens_tfinal,
        sav=sav_values[0],
        k0=args.k0,
        alpha_range=(alpha_min, alpha_max),
        m_range=(m_min, m_max),
        n=args.sens_n,
    )

    t_grid = np.linspace(0.0, args.tmax, 120)
    sav_grid_surface = np.linspace(args.sav_min, args.sav_max, 120)
    plot_E_surface_3d(
        outdir=args.outdir,
        t_grid=t_grid,
        sav_grid=sav_grid_surface,
        mn0=args.mn0,
        e0=args.e0,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
        cryst_params=cryst_params,
        two_phase=two_phase if args.mode in ("two_phase", "rxn_diff1d") else None,
        bxc=args.bxc,
    )

    # characteristic time
    t_char = characteristic_time_fraction(
        f=args.fraction, k0=args.k0, sav=sav_values[0], m=args.m, alpha=args.alpha
    )
    with open(os.path.join(args.outdir, "characteristic_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"Target fraction f = {args.fraction}\n")
        f.write(f"SA/V used = {sav_values[0]}\n")
        f.write(f"alpha = {args.alpha}, m = {args.m}, k0 = {args.k0}\n")
        f.write(f"t_char (E/E0=f) = {t_char}\n")

    # two-phase extra plot
    if args.mode == "two_phase" and two_phase is not None:
        plot_mass_fraction_vs_t(
            outdir=args.outdir,
            t=t,
            mn0=args.mn0,
            k0=args.k0,
            m=args.m,
            sav=sav_values[0],
            two_phase=two_phase,
            time_label=args.time_label,
        )

    # GIFs
    t_steps = np.linspace(0.0, args.tmax, args.gif_frames)
    gif_E_vs_SAV_over_time(
        outdir=args.outdir,
        sav_grid=sav_grid,
        t_steps=t_steps,
        mn0=args.mn0,
        e0=args.e0,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
        cryst_params=cryst_params,
        two_phase=two_phase if args.mode in ("two_phase", "rxn_diff1d") else None,
        bxc=args.bxc,
    )
    gif_E_surface_rotate(
        outdir=args.outdir,
        t_grid=t_grid,
        sav_grid=sav_grid_surface,
        mn0=args.mn0,
        e0=args.e0,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
        cryst_params=cryst_params,
        two_phase=two_phase if args.mode in ("two_phase", "rxn_diff1d") else None,
        bxc=args.bxc,
        n_frames=48,
    )

    # rxn-diffusion module
    if args.mode == "rxn_diff1d":
        rd = RxnDiff1DParams(
            length=args.L,
            nx=args.nx,
            D0=args.D0,
            beta_D=args.beta_D,
            s0=args.s0,
            use_two_phase_mn=bool(args.rxn_use_two_phase_mn),
        )
        x, C_hist = simulate_rxn_diff_1d(
            t=t,
            mn0=args.mn0,
            k0=args.k0,
            sav=sav_values[0],
            m=args.m,
            params=rd,
            two_phase=two_phase,
            cryst_params=cryst_params,
            gamma_xc=args.gamma_xc,
        )
        plot_C_profile_heatmap(args.outdir, x, t, C_hist)
        gif_C_profile_over_time(args.outdir, x, t, C_hist, n_frames=min(args.gif_frames, 80))

    print(f"Done. Outputs saved to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
    
