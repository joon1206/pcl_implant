#!/usr/bin/env python3

"""
This is my attempt at trying out a hybrid PCL "implant-like" degradation model, for a 1D slab (approximate geometry). 

The core ideas implemented are... (by the way, on their own all of these are pretty lightweight,
and therefore adding any theoretical and/or mathematical limitations to the model is actually
quite trivial. Either way, here are the concepts!)
    1) Bulk hydrolysis with autocatalysis via acid/oligomer concentration C(x, t) (This is
        simply a pseudo-first order molecular weight decay model, I have the skeletons for the code
        in the other project file)
    2) Acid/oligomer diffusion with source term from chain scission (This I think should just be 
        Fick's second law)
    3) In vivo-esque clearance via Robin BC (need to add to theory documentation) 
    4) Optional enzymatic surfaceerosion via a recession velocity v_n, this is Michaelis-Menten-like
    5) HEAVILY OPTIONAL introducing continuous reaction-diffusion & chain scission distribution,
        this lets us predict MWD/PDI (though this is only validated w/ PE data). This woul dbe 
        CRAZY cool because it predicts all of the above fully given initial degradation settings.
        Though it would not be the best without rigorous validation. 


The fun part is that there are many models for hydrolytic degredation and diffusion already. 
We are just adding some on top. 

State variables:
  M(x,t) : local (normalized) molecular weight fraction, M = Mn/Mn0 (dimensionless)
  C(x,t) : local acid/oligomer concentration (arbitrary units)

Governing equations (1D, x in [0, L], surfaces at x=0 and x=L):
  dM/dt = -k_h * M * (1 + beta*C)                    (bulk hydrolysis + autocatalysis)
  dC/dt = d/dx ( D(M) dC/dx ) + y * (-dM/dt) - k_cl*C  (diffusion + source + optional bulk clearance)

  D(M) = D0 * exp( R * (1 - M) )   (diffusivity increases with degradation / porosity proxy)

Boundary conditions (Robin):
  At x=0:  -D dC/dx = h (C - C_inf)
  At x=L:  +D dC/dx = h (C - C_inf)   (sign flips with outward normal)
Enzymatic surface erosion (optional):
  v_n = k_enz * E / (K_M + E)    (if you treat enzyme concentration E as constant, v_n is constant)
  L(t) decreases as: dL/dt = -2*v_n   (both faces erode inward)
  Mass loss proxy: m(t)/m0 ~ L(t)/L0  (slab area constant)

Mechanical coupling:
  E_local(x,t)/E0 = M(x,t)^alpha
  E_effective(t) can be:
    - volume-average of E_local
    - or minimum (weakest-link) or surface-average, depending on your narrative.
  Here we compute volume-average by default.

Outputs (in --outdir):
  - profiles_M_C.png                (snapshots of M(x) and C(x))
  - E_effective_vs_time.png
  - sensitivity_clearance_h.png     (how clearance changes E(t))
  - landscape_E_time_vs_L0.png      (E(t) for different initial thicknesses)
  - gif_profiles.gif                (M(x), C(x) evolving)
  - gif_landscape.gif               (E fraction vs time as thickness varies, animated)


Install:
    pip install numpy matplotlib pillow

"""


from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------
# Utilities
# -----------------------------
def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def savefig(path: str, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def make_gif(frame_paths: List[str], out_gif: str, duration_ms: int = 120) -> None:
    imgs = [Image.open(p).convert("RGBA") for p in frame_paths]
    imgs[0].save(
        out_gif,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


# -----------------------------
# Model parameters
# -----------------------------
@dataclass
class Params:
    # Hydrolysis / autocatalysis
    k_h: float = 1e-3        # base hydrolysis rate (1/time)
    beta: float = 5.0        # autocatalysis strength (1/concentration units)

    # Diffusion
    D0: float = 1e-2         # base diffusivity (length^2/time)
    R: float = 4.0           # diffusivity growth with degradation

    # Acid/oligomer source & clearance
    y: float = 1.0           # yield from chain scission to C (concentration per M-loss)
    k_cl: float = 0.0        # optional bulk clearance / buffering (1/time)

    # Robin boundary (perfusion / sink)
    h: float = 1e-1          # mass transfer coefficient (length/time)
    C_inf: float = 0.0       # external concentration (sink)

    # Mechanics
    alpha: float = 2.0       # modulus scaling exponent: E/E0 = M^alpha
    E0: float = 1.0          # normalized (we plot E/E0)

    # Enzymatic erosion (optional)
    erosion_on: bool = False
    k_enz: float = 0.0       # length/time
    Enz: float = 0.0         # enzyme concentration (arbitrary)
    Km: float = 1.0          # Michaelis constant (same units as Enz)

    # Numerics
    dt: float = 0.1          # time step
    tmax: float = 100.0      # end time
    nx: int = 101            # spatial nodes


# -----------------------------
# Core solver
# -----------------------------
def diffusivity(M: np.ndarray, p: Params) -> np.ndarray:
    # D increases as M decreases (damage/porosity proxy)
    return p.D0 * np.exp(p.R * (1.0 - M))


def apply_robin_bc(C: np.ndarray, D: np.ndarray, dx: float, p: Params) -> Tuple[float, float]:
    """
    Compute boundary flux terms using Robin BC in ghost-node style.
    Returns equivalent ghost values C_-1 and C_{nx} to use in second derivative stencils.

    For x=0, outward normal points to -x:
      -D * dC/dn = -D*(-dC/dx) = D dC/dx = h (C0 - C_inf)
      => dC/dx at 0 = h (C0 - C_inf) / D0

    For x=L, outward normal points to +x:
      -D * dC/dn = -D*(dC/dx) = h (CL - C_inf)
      => dC/dx at L = -h (CL - C_inf) / DL

    With one-sided derivative:
      dC/dx|0 ≈ (C1 - C_-1)/(2dx)
      dC/dx|L ≈ (C_ghost - C_{nx-2})/(2dx)
    """
    C0 = C[0]
    CL = C[-1]

    D0 = max(D[0], 1e-30)
    DL = max(D[-1], 1e-30)

    dCdx_0 = p.h * (C0 - p.C_inf) / D0
    dCdx_L = -p.h * (CL - p.C_inf) / DL

    C_ghost_left = C[1] - 2.0 * dx * dCdx_0
    C_ghost_right = C[-2] + 2.0 * dx * dCdx_L
    return C_ghost_left, C_ghost_right


def step_explicit(M: np.ndarray, C: np.ndarray, L: float, p: Params) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    One explicit Euler step for M and C on a uniform grid.
    Also optionally updates L(t) for enzymatic surface erosion.
    """
    nx = p.nx
    dx = L / (nx - 1)

    # Hydrolysis + autocatalysis
    rate = p.k_h * (1.0 + p.beta * C)
    dMdt = -rate * M
    M_new = M + p.dt * dMdt
    M_new = np.clip(M_new, 0.0, 1.0)

    # Diffusion coefficient depends on current M (could also use M_new)
    D = diffusivity(M, p)

    # Robin BC via ghost nodes
    CgL, CgR = apply_robin_bc(C, D, dx, p)

    # Build C with ghost points for Laplacian
    C_ext = np.empty(nx + 2, dtype=float)
    C_ext[0] = CgL
    C_ext[1:-1] = C
    C_ext[-1] = CgR

    # Variable-coefficient diffusion term in conservative form:
    # d/dx ( D dC/dx ) ~ (1/dx^2) * [ D_{i+1/2}(C_{i+1}-C_i) - D_{i-1/2}(C_i-C_{i-1}) ]
    D_iphalf = 0.5 * (D[1:] + D[:-1])
    D_imhalf = D_iphalf  # same array, just shifted usage

    diffusion = np.zeros_like(C)
    # interior nodes i=1..nx-2
    for i in range(1, nx - 1):
        # half indices: i+1/2 corresponds to D_iphalf[i]
        flux_r = D_iphalf[i] * (C_ext[i + 2] - C_ext[i + 1])
        flux_l = D_iphalf[i - 1] * (C_ext[i + 1] - C_ext[i])
        diffusion[i] = (flux_r - flux_l) / (dx * dx)

    # boundaries i=0 and i=nx-1 using ghost points
    # i=0:
    flux_r = D_iphalf[0] * (C_ext[2] - C_ext[1])
    flux_l = D[0] * (C_ext[1] - C_ext[0])  # uses D0 at boundary
    diffusion[0] = (flux_r - flux_l) / (dx * dx)
    # i=nx-1:
    flux_r = D[-1] * (C_ext[-1] - C_ext[-2])
    flux_l = D_iphalf[-1] * (C_ext[-2] - C_ext[-3])
    diffusion[-1] = (flux_r - flux_l) / (dx * dx)

    # Source from chain scission (positive when M decreases)
    source = p.y * (-dMdt)  # since dMdt is negative
    # Optional bulk clearance/buffering
    clearance = -p.k_cl * C

    dCdt = diffusion + source + clearance
    C_new = C + p.dt * dCdt
    C_new = np.clip(C_new, 0.0, None)

    # Optional enzymatic surface erosion (recess both faces)
    L_new = L
    if p.erosion_on and p.k_enz > 0.0:
        vn = p.k_enz * (p.Enz / (p.Km + p.Enz + 1e-30))
        L_new = max(L - 2.0 * vn * p.dt, 1e-9)  # don't go negative

    return M_new, C_new, L_new


def simulate(p: Params, L0: float) -> dict:
    """
    Returns time series and snapshots.
    """
    nx = p.nx
    nsteps = int(np.ceil(p.tmax / p.dt)) + 1

    # initial conditions
    M = np.ones(nx, dtype=float)
    C = np.zeros(nx, dtype=float)
    L = float(L0)

    ts = np.zeros(nsteps)
    Es = np.zeros(nsteps)
    Ls = np.zeros(nsteps)

    # store a few snapshots for plotting
    snap_indices = np.linspace(0, nsteps - 1, 6, dtype=int)
    snaps = []

    for k in range(nsteps):
        t = k * p.dt
        ts[k] = t
        Ls[k] = L

        # mechanics: local E/E0 = M^alpha ; effective = average over thickness
        E_local = M ** p.alpha
        Es[k] = float(np.mean(E_local))

        if k in snap_indices:
            x = np.linspace(0.0, L, nx)
            snaps.append((t, x.copy(), M.copy(), C.copy(), L))

        if k == nsteps - 1:
            break

        M, C, L = step_explicit(M, C, L, p)

    return {
        "t": ts,
        "E": Es,
        "L": Ls,
        "snapshots": snaps,
    }


# -----------------------------
# Plotting
# -----------------------------
def plot_profiles(outdir: str, sim: dict) -> str:
    snaps = sim["snapshots"]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for (t, x, M, C, L) in snaps:
        plt.plot(x, M, label=f"t={t:g}")
    plt.xlabel("x (thickness direction)")
    plt.ylabel("M = Mn/Mn0")
    plt.title("Molecular weight fraction profiles")
    plt.ylim(0, 1.05)
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for (t, x, M, C, L) in snaps:
        plt.plot(x, C, label=f"t={t:g}")
    plt.xlabel("x (thickness direction)")
    plt.ylabel("C (acid/oligomer)")
    plt.title("Acid/oligomer concentration profiles")
    plt.legend(fontsize=8)

    outpath = os.path.join(outdir, "profiles_M_C.png")
    savefig(outpath)
    return outpath


def plot_E_vs_time(outdir: str, sim: dict) -> str:
    t = sim["t"]
    E = sim["E"]
    L = sim["L"]

    plt.figure()
    plt.plot(t, E, label="E_eff/E0")
    plt.xlabel("time")
    plt.ylabel("E_eff/E0")
    plt.title("Effective modulus vs time (volume-averaged)")
    plt.ylim(0, 1.05)
    plt.legend()

    outpath = os.path.join(outdir, "E_effective_vs_time.png")
    savefig(outpath)
    return outpath


def plot_sensitivity_clearance(outdir: str, base: Params, L0: float, h_values: List[float]) -> str:
    plt.figure()
    for h in h_values:
        p = Params(**{**base.__dict__, "h": h})
        sim = simulate(p, L0=L0)
        plt.plot(sim["t"], sim["E"], label=f"h={h:g}")

    plt.xlabel("time")
    plt.ylabel("E_eff/E0")
    plt.title("Sensitivity to clearance strength (Robin h)")
    plt.ylim(0, 1.05)
    plt.legend()

    outpath = os.path.join(outdir, "sensitivity_clearance_h.png")
    savefig(outpath)
    return outpath


def plot_landscape_E_time_vs_L0(outdir: str, base: Params, L0_values: List[float]) -> str:
    plt.figure()
    for L0 in L0_values:
        sim = simulate(base, L0=L0)
        plt.plot(sim["t"], sim["E"], label=f"L0={L0:g}")

    plt.xlabel("time")
    plt.ylabel("E_eff/E0")
    plt.title("E(t) for different implant thicknesses (size effect)")
    plt.ylim(0, 1.05)
    plt.legend()

    outpath = os.path.join(outdir, "landscape_E_time_vs_L0.png")
    savefig(outpath)
    return outpath


def gif_profiles(outdir: str, base: Params, L0: float, nframes: int = 60) -> str:
    tmp = os.path.join(outdir, "_frames_profiles")
    os.makedirs(tmp, exist_ok=True)

    # run one simulation but capture frames at selected steps
    nx = base.nx
    nsteps = int(np.ceil(base.tmax / base.dt)) + 1
    frame_steps = np.linspace(0, nsteps - 1, nframes, dtype=int)

    M = np.ones(nx, dtype=float)
    C = np.zeros(nx, dtype=float)
    L = float(L0)

    frames = []
    for k in range(nsteps):
        if k in frame_steps:
            t = k * base.dt
            x = np.linspace(0.0, L, nx)

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.plot(x, M)
            plt.xlabel("x")
            plt.ylabel("M")
            plt.title(f"M profile (t={t:g})")
            plt.ylim(0, 1.05)

            plt.subplot(1, 2, 2)
            plt.plot(x, C)
            plt.xlabel("x")
            plt.ylabel("C")
            plt.title(f"C profile (t={t:g})")

            fp = os.path.join(tmp, f"frame_{len(frames):04d}.png")
            savefig(fp, dpi=140)
            frames.append(fp)

        if k == nsteps - 1:
            break
        M, C, L = step_explicit(M, C, L, base)

    outgif = os.path.join(outdir, "gif_profiles.gif")
    make_gif(frames, outgif, duration_ms=110)
    return outgif


def gif_landscape(outdir: str, base: Params, L0_min: float, L0_max: float, nL: int = 20, nframes: int = 40) -> str:
    tmp = os.path.join(outdir, "_frames_landscape")
    os.makedirs(tmp, exist_ok=True)

    L0_vals = np.linspace(L0_min, L0_max, nL)
    # compute E(t) curves for each L0
    sims = [simulate(base, L0=float(L0)) for L0 in L0_vals]
    t = sims[0]["t"]

    # animate a "moving time cursor" over a thickness-vs-E plot
    frames = []
    frame_idx = np.linspace(0, len(t) - 1, nframes, dtype=int)

    for j, idx in enumerate(frame_idx):
        E_at_time = np.array([s["E"][idx] for s in sims])

        plt.figure()
        plt.plot(L0_vals, E_at_time, marker="o", linewidth=1)
        plt.xlabel("Initial thickness L0")
        plt.ylabel("E_eff/E0")
        plt.title(f"Size effect snapshot at t={t[idx]:g}")
        plt.ylim(0, 1.05)

        fp = os.path.join(tmp, f"frame_{j:04d}.png")
        savefig(fp, dpi=140)
        frames.append(fp)

    outgif = os.path.join(outdir, "gif_landscape.gif")
    make_gif(frames, outgif, duration_ms=140)
    return outgif


# -----------------------------
# CLI
# -----------------------------
def parse_floats(csv: str) -> List[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs_hybrid")
    ap.add_argument("--L0", type=float, default=2.0, help="Initial thickness (length units)")
    ap.add_argument("--nx", type=int, default=101)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--tmax", type=float, default=100.0)

    # Hydrolysis/autocatalysis
    ap.add_argument("--k_h", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=5.0)

    # Diffusion
    ap.add_argument("--D0", type=float, default=1e-2)
    ap.add_argument("--R", type=float, default=4.0)

    # Source/clearance
    ap.add_argument("--y", type=float, default=1.0)
    ap.add_argument("--k_cl", type=float, default=0.0)

    # Robin boundary
    ap.add_argument("--h", type=float, default=1e-1)
    ap.add_argument("--C_inf", type=float, default=0.0)

    # Mechanics
    ap.add_argument("--alpha", type=float, default=2.0)

    # Enzymatic erosion toggles
    ap.add_argument("--erosion", action="store_true", help="Enable enzymatic surface erosion")
    ap.add_argument("--k_enz", type=float, default=0.0)
    ap.add_argument("--Enz", type=float, default=0.0)
    ap.add_argument("--Km", type=float, default=1.0)

    # Sweeps / sensitivity
    ap.add_argument("--h_sweep", type=str, default="0.01,0.1,1.0",
                    help="Comma-separated h values for clearance sensitivity plot")
    ap.add_argument("--L_sweep", type=str, default="0.5,1,2,4,8",
                    help="Comma-separated L0 values for size-effect plot")

    # GIF controls
    ap.add_argument("--gif_frames", type=int, default=60)

    args = ap.parse_args()
    ensure_outdir(args.outdir)

    p = Params(
        k_h=args.k_h,
        beta=args.beta,
        D0=args.D0,
        R=args.R,
        y=args.y,
        k_cl=args.k_cl,
        h=args.h,
        C_inf=args.C_inf,
        alpha=args.alpha,
        erosion_on=bool(args.erosion),
        k_enz=args.k_enz,
        Enz=args.Enz,
        Km=args.Km,
        dt=args.dt,
        tmax=args.tmax,
        nx=args.nx,
    )

    # Base run
    sim = simulate(p, L0=args.L0)

    # Plots
    plot_profiles(args.outdir, sim)
    plot_E_vs_time(args.outdir, sim)

    # Sensitivity to clearance
    h_vals = parse_floats(args.h_sweep)
    plot_sensitivity_clearance(args.outdir, p, L0=args.L0, h_values=h_vals)

    # Thickness sweep
    L_vals = parse_floats(args.L_sweep)
    plot_landscape_E_time_vs_L0(args.outdir, p, L0_values=L_vals)

    # GIFs
    gif_profiles(args.outdir, p, L0=args.L0, nframes=args.gif_frames)
    gif_landscape(args.outdir, p, L0_min=min(L_vals), L0_max=max(L_vals), nL=24, nframes=40)

    # Save a quick run summary
    with open(os.path.join(args.outdir, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Hybrid implant-like degradation model (1D slab)\n")
        f.write(f"L0={args.L0}, nx={args.nx}, dt={args.dt}, tmax={args.tmax}\n")
        f.write(f"k_h={p.k_h}, beta={p.beta}\n")
        f.write(f"D0={p.D0}, R={p.R}\n")
        f.write(f"y={p.y}, k_cl={p.k_cl}\n")
        f.write(f"h={p.h}, C_inf={p.C_inf}\n")
        f.write(f"alpha={p.alpha}\n")
        f.write(f"erosion_on={p.erosion_on}, k_enz={p.k_enz}, Enz={p.Enz}, Km={p.Km}\n")

    print(f"Done. Outputs saved to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()


