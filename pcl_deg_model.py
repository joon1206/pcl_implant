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

    mask = t > tcrit
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

    dMn_dt = np.gradient(Mn, t)

    S_t = params.s0 * np.maximum(0.0, -dMn_dt) / max(mn0, 1e-12)

    C = np.zeros(params.nx, dtype=float)
    C_hist = np.zeros((nt, params.nx), dtype=float)

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
        frac_deg = 1.0 - (Mn[i] / max(mn0, 1e-12))
        D = params.D0 * (1.0 + params.beta_D * frac_deg)
        D = max(float(D), 0.0)

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
) -> str:
    plt.figure()
    for i, sav in enumerate(sav_list):
        if two_phase is None:
            y = mn_t(t, mn0, k0, sav, m)
        else:
            y = mn_two_phase(t, mn0, k0, sav, m, two_phase)
        lab = labels[i] if labels and i < len(labels) else f"SA/V={sav:.3g}"
        plt.plot(t, y, label=lab)

    plt.xlabel("time")
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
) -> str:
    frac = mass_fraction_two_phase(t, mn0, k0, sav, m, two_phase)

    plt.figure()
    plt.plot(t, frac)
    plt.xlabel("time")
    plt.ylabel("mass fraction (proxy)")
    plt.title("Linear mass loss after Mn threshold (two-phase proxy)")
    outpath = os.path.join(outdir, "mass_fraction_vs_t.png")
    savefig(outpath)
    return outpath


def plot_E_vs_t_geometries(
    outdir: str,
    t: np.ndarray,
    e0: float,
    k0: float,
    m: float,
    alpha: float,
    sav_list: List[float],
    labels: Optional[List[str]] = None,
) -> str:
    plt.figure()
    for i, sav in enumerate(sav_list):
        y = e_t(t, e0, k0, sav, m, alpha)
        lab = labels[i] if labels and i < len(labels) else f"SA/V={sav:.3g}"
        plt.plot(t, y, label=lab)

    plt.xlabel("time")
    plt.ylabel("E(t)")
    plt.title("Young's modulus vs time for different geometries (SA/V)")
    plt.legend()
    outpath = os.path.join(outdir, "E_vs_t_geometries.png")
    savefig(outpath)
    return outpath


def plot_E_vs_SAV_fixed_times(
    outdir: str,
    sav_grid: np.ndarray,
    times: List[float],
    e0: float,
    k0: float,
    m: float,
    alpha: float,
) -> str:
    plt.figure()
    for tt in times:
        y = e_t(np.array([tt]), e0, k0, sav_grid, m, alpha).reshape(-1)
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
) -> str:
    Mn = mn_t(t, mn0, k0, sav, m)
    E = e_t(t, e0, k0, sav, m, alpha)

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
    plt.title(f"Log-log check of E ~ Mn^alpha (fit slope ≈ {slope:.3g})")

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
    k0: float,
    m: float,
    alpha: float,
) -> str:
    T, S = np.meshgrid(t_grid, sav_grid)
    frac = np.exp(-(alpha * k0 * (S ** m)) * T)

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
    e0: float,
    k0: float,
    m: float,
    alpha: float,
) -> str:
    frames = []
    tmpdir = os.path.join(outdir, "_frames_E_vs_SAV")
    os.makedirs(tmpdir, exist_ok=True)

    for i, tt in enumerate(t_steps):
        plt.figure()
        y = e_t(np.array([tt]), e0, k0, sav_grid, m, alpha).reshape(-1)
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
    k0: float,
    m: float,
    alpha: float,
    n_frames: int = 48,
) -> str:
    T, S = np.meshgrid(t_grid, sav_grid)
    frac = np.exp(-(alpha * k0 * (S ** m)) * T)

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


# -----------------------------
# CLI
# -----------------------------

def parse_floats(csv: str) -> List[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="PCL degradation + mechanics scaling model (SA/V).")

    ap.add_argument("--cad", type=str, default=None,
                    help="Path to mesh CAD file (STL/OBJ/PLY/GLB...). If omitted, use --sav-list.")
    ap.add_argument("--sav-list", type=str, default="5,10,20",
                    help="Comma-separated SA/V values (used if --cad not provided).")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory.")

    # model selection
    ap.add_argument("--mode", type=str, default="base",
                    choices=["base", "two_phase", "rxn_diff1d"],
                    help="Which add-ons to enable.")

    # core params
    ap.add_argument("--mn0", type=float, default=60000.0, help="Initial Mn (e.g., g/mol).")
    ap.add_argument("--e0", type=float, default=1000.0, help="Initial E (e.g., MPa).")
    ap.add_argument("--k0", type=float, default=1e-4,
                    help="Material/environment constant in kd = k0*(SA/V)^m (units depend on your choices).")
    ap.add_argument("--m", type=float, default=1.0, help="Geometry exponent (m≈1 surface-controlled; m≈0 bulk).")
    ap.add_argument("--alpha", type=float, default=2.0, help="E~Mn^alpha scaling exponent (often 1-3).")

    # Time axis
    ap.add_argument("--tmax", type=float, default=120.0, help="Max time for plots (self-consistent units).")
    ap.add_argument("--nt", type=int, default=200, help="Number of time points.")
    ap.add_argument("--time-label", type=str, default="time",
                    help="X-axis label (e.g., 'time (weeks)' or 'time (years)').")

    # E vs SA/V at fixed times
    ap.add_argument("--fixed-times", type=str, default="0,12,24,48,96,120",
                    help="Comma-separated times to plot E vs SA/V.")
    ap.add_argument("--sav-min", type=float, default=1.0, help="Min SA/V for grids.")
    ap.add_argument("--sav-max", type=float, default=50.0, help="Max SA/V for grids.")
    ap.add_argument("--nsav", type=int, default=200, help="Number of SA/V points in grids.")

    # Sensitivity
    ap.add_argument("--sens-tfinal", type=float, default=48.0, help="Time for sensitivity heatmap.")
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
    )
    # relabel time axes in saved plots (simple approach: set global label via rcParams is messy;
    # instead, we just overwrite xlabel in key plots by replotting? low priority. Keep label via args.)
    # For now, we keep "time" in plots; user can edit labels if needed.

    plot_E_vs_t_geometries(
        outdir=args.outdir,
        t=t,
        e0=args.e0,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
        sav_list=sav_values,
        labels=sav_labels,
    )

    sav_grid = np.linspace(args.sav_min, args.sav_max, args.nsav)
    fixed_times = parse_floats(args.fixed_times)
    plot_E_vs_SAV_fixed_times(
        outdir=args.outdir,
        sav_grid=sav_grid,
        times=fixed_times,
        e0=args.e0,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
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
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
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
        )

    # GIFs
    t_steps = np.linspace(0.0, args.tmax, args.gif_frames)
    gif_E_vs_SAV_over_time(
        outdir=args.outdir,
        sav_grid=sav_grid,
        t_steps=t_steps,
        e0=args.e0,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
    )
    gif_E_surface_rotate(
        outdir=args.outdir,
        t_grid=t_grid,
        sav_grid=sav_grid_surface,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
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
        )
        plot_C_profile_heatmap(args.outdir, x, t, C_hist)
        gif_C_profile_over_time(args.outdir, x, t, C_hist, n_frames=min(args.gif_frames, 80))

    print(f"Done. Outputs saved to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
    