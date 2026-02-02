"""
Degradation and mechanical strength scaling model (first order chain scission & SA/V scaling

The model does something like this:
    Mn(t) = Mn0 + exp(-k0 * (SA/V)^m * t)
    E(t) = E0 * (Mn(t)/Mn0)^alpha = E0 * exp(-alpha * k0 * (SA/V)^m * t)

Outputs (saved to --outdir):
    E_vs_t_geometries.png
    E_vs_SAV_fixed_times.png
    loglog_E_vs_Mn.png
    sensitivity_alpha_m.png
    E_surface_3d.png
    gif_E_vs_SAV_over_time.gif
    gif_E_surface_rotate.gif

CAD / Geometry stuff
    I envision this to support mesh files that are readable by trimesh: STL, OBJ, PLY, GLB/GLTF, etc. Check the help page for the package to find more, since I'm not an expert at this haha
    For STEP/IGES you would apparently need a CAD kernel (OCC/cadquery) to tesselate into STL first. Check with Connie for output types. 


INSTALL! These are thepackages you would need.
    
    pip install numpy matplotlib trimesh pillow



OPTIONAL INSTALL (if it comes to reading CAD, since for now we can get whatever, but in the future, this could be a VERY, and I mean VERY useful tool for any kind of geometry-degradation-mechanical strength model.
    
    pip install "trimesh[easy]"

That is it for now. Best, Joon J. 
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# this is for making GIFs
from PIL import Image

# for CAD mesh loading if it comes to that, and to find area/volume
import trimesh

#-----------------------
# FIRST MODEL, JOON. 12/24/2025
#-----------------------


"""
Number average molecular weight over time. 
"""
def mn_t (t: np.ndarray, mn0: float, k0: float, sav: float, m: float) -> np.ndarray :
    kd = k0 * (sav ** m)
    return mn0 * np.exp(-kd * t)

"""
Young's modulus over time from Mn scaling
"""
def e_t (t: np.ndarray, e0: float, k0: float, sav: float, m: float,  alpha: float) -> np.ndarray :
    kd = k0 * (sav ** m) 
    return (e0 * np.exp(-(alpha * kd) * t))


def characteristic_time_fraction(
        f: float, k0: float, sav: float, m: float, alpha: float) -> float:
    """
    time to reach E(t)/E0 = f (where f is between 0 and 1, since E0 is original)
        f = exp (-alpha * k0 * (sav^m) * t --> t = -ln(f) / (alpha * k0 * sav^m) 
        where  ln(f) = ln (E(t) / E0)
    """
    denom = alpha * k0 * sav ** m

    if not (0.0 < f < 1.0):
        raise ValueError ("f must be between 0 and 1 exclusive")
    
    if denom <= 0:
        raise ValueError("alpha * k0 * (sav^m) must be over 0")

    return (-1 * (math.log(f) / denom))

#-------------------
# now for the geometry calculations!
#-------------------

def load_mesh_compute_sav(path: str) -> Tuple():
    """
    Okay so this is a little funky, you start by loading a mesh, and then computing the following:
    SA = surface area
    V = volume (requires a watertight mesh to be reliable, I tried some stuff and if the mesh doesn't connect fully (ie. tight corners not rendered correctly) it doesn't compute properly. Then again it was my old bike project mesh so it could just be completely incorrect. 
    SA/V
    Returns (SA, V, SA/V)

    ***IMPORTANT***
    the SA/V will be 1/length, where the length is whatever units your mesh is in. This is important with the final product, make sure that all the units match. We can come back to this to standardize it, maybe. 
    """

    mesh = trimesh.load(path, force="mesh")

    if mesh.is_empty:
        raise ValueError("Mesh is empty: {path}")

    #make sure you have a trimesh and not a scene

    if isinstance(mesh, trimesh.Scene):
        # combine into one mesh... while technically you lose some details this is the best effort solution for now. If there is anything better please employ. 

        mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                )

    sa = float(mesh.area)
    
    #now for volume... technically you need a "closed" ("watertight) mesh, but trimesh will
    #return a value, but also warn if not watertight. 

    if not mesh.is_watertight:
        print("[WARN] Mesh is not watertight, volume may be inaccurate. /n"
        "   Consider exporting a closed STL or using a CAD tool to make it completely solid")

    vol = float(mesh.volume)
    if vol <= 0: 
        raise ValueError(f"computed value is not positive ({vol})."
                "Your mesh may be not watertight / not solid!")

    sav = sa / vol
    return sa, vol, sav




#----------------------
# Now for the plot helpers! This is your typical stuff.
#----------------------

def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok = True)

def savefig(path: str, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi = dpi)
    plt.close()

def make_gif(frames: List[str], out_gif: str, duration_ms: int = 120) -> None:
    imgs = [Image.open(p).convert("RGBA") for p in frames]
    imgs[0].save(
                out_gif, 
                save_all = True,
                append_images=imgs[1:],
                duration=duration_ms,
                loop = 0,
                disposal = 2,
                )
   


        

# -----------------------------
# Main plotting functions
# -----------------------------
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

    # normalized, so log-log is clean:
    x = Mn / mn0
    y = E / e0

    # Avoid log(0)
    eps = 1e-16
    x = np.clip(x, eps, None)
    y = np.clip(y, eps, None)

    plt.figure()
    plt.loglog(x, y, marker="o", linewidth=1)

    # Fit log(y)=alpha*log(x) ideally (intercept ~ 0)
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
    e0: float,
    k0: float,
    alpha_range: Tuple[float, float],
    m_range: Tuple[float, float],
    n: int = 60,
) -> str:
    alphas = np.linspace(alpha_range[0], alpha_range[1], n)
    ms = np.linspace(m_range[0], m_range[1], n)

    # E/E0 at tfinal:
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
    e0: float,
    k0: float,
    m: float,
    alpha: float,
) -> str:
    # 3D surface of E/E0 over (t, SA/V)
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
    e0: float,
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


# -----------------------------
# CLI
# -----------------------------
def parse_floats(csv: str) -> List[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Degradation + mechanics scaling model (SA/V).")
    ap.add_argument("--cad", type=str, default=None,
                    help="Path to mesh CAD file (STL/OBJ/PLY/GLB...). If omitted, use --sav-list.")
    ap.add_argument("--sav-list", type=str, default="5,10,20",
                    help="Comma-separated SA/V values (used if --cad not provided).")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory.")

    # Parameters
    ap.add_argument("--mn0", type=float, default=60000.0, help="Initial Mn (e.g., g/mol).")
    ap.add_argument("--e0", type=float, default=1000.0, help="Initial E (e.g., MPa).")

    # IMPORTANT: k0 depends on your units of time and SA/V.
    # If time is in weeks and SA/V in 1/mm, k0 has units: (weeks^-1) * (mm^m).
    ap.add_argument("--k0", type=float, default=1e-4,
                    help="Material/environment constant in kd = k0*(SA/V)^m (units depend on your choices).")
    ap.add_argument("--m", type=float, default=1.0, help="Geometry exponent (m≈1 surface-controlled; m≈0 bulk).")
    ap.add_argument("--alpha", type=float, default=2.0, help="E~Mn^alpha scaling exponent (often 1-3).")

    # Time axis
    ap.add_argument("--tmax", type=float, default=120.0, help="Max time for plots (same units as k0).")
    ap.add_argument("--nt", type=int, default=200, help="Number of time points.")

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
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    # Determine SA/V values
    sav_values: List[float] = []
    sav_labels: List[str] = []

    if args.cad is not None:
        sa, vol, sav = load_mesh_compute_sav(args.cad)
        sav_values = [sav]
        sav_labels = [f"{os.path.basename(args.cad)} (SA/V={sav:.3g})"]
        # Save a quick text summary
        with open(os.path.join(args.outdir, "geometry_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"CAD: {args.cad}\n")
            f.write(f"Surface area (SA): {sa}\n")
            f.write(f"Volume (V): {vol}\n")
            f.write(f"SA/V: {sav}\n")
    else:
        sav_values = parse_floats(args.sav_list)
        sav_labels = [f"SA/V={v:.3g}" for v in sav_values]

    # Time vector
    t = np.linspace(0.0, args.tmax, args.nt)

    # Plots
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

    # Log-log check using the *first* SA/V as a representative geometry
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

    # Sensitivity map (alpha,m)
    alpha_min, alpha_max = parse_floats(args.sens_alpha)
    m_min, m_max = parse_floats(args.sens_m)
    plot_sensitivity_alpha_m(
        outdir=args.outdir,
        tfinal=args.sens_tfinal,
        sav=sav_values[0],
        e0=args.e0,
        k0=args.k0,
        alpha_range=(alpha_min, alpha_max),
        m_range=(m_min, m_max),
        n=args.sens_n,
    )

    # 3D surface plot across (t, SA/V)
    t_grid = np.linspace(0.0, args.tmax, 120)
    sav_grid_surface = np.linspace(args.sav_min, args.sav_max, 120)
    plot_E_surface_3d(
        outdir=args.outdir,
        t_grid=t_grid,
        sav_grid=sav_grid_surface,
        e0=args.e0,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
    )

    # Characteristic time for chosen fraction
    t_char = characteristic_time_fraction(
        f=args.fraction, k0=args.k0, sav=sav_values[0], m=args.m, alpha=args.alpha
    )
    with open(os.path.join(args.outdir, "characteristic_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"Target fraction f = {args.fraction}\n")
        f.write(f"SA/V used = {sav_values[0]}\n")
        f.write(f"alpha = {args.alpha}, m = {args.m}, k0 = {args.k0}\n")
        f.write(f"t_char (E/E0=f) = {t_char}\n")

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
        e0=args.e0,
        k0=args.k0,
        m=args.m,
        alpha=args.alpha,
        n_frames=48,
    )

    print(f"Done. Outputs saved to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()






























