"""
Post-processing and visualization script for lid-driven cavity flow solutions.

Loads precomputed flow fields from a NumPy .npz file and generates
high-quality showcase plots with LaTeX typography and clean legend positioning.

Usage:
    python plot_from_npz.py --file data/flow_fields_Re3200_N257.npz
"""

import os
import sys
import io
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import verified benchmark data and styling from core solver module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lid_driven_cavity_fdm import GHIA_DATA, _apply_latex_style

# UNICODE FIX FOR TERMINALS
if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    try:
        current_encoding = getattr(sys.stdout, 'encoding', None)
        if current_encoding and current_encoding.lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


def detect_vortices(psi, x, y):
    """Detect primary vortex and secondary corner eddies from streamfunction array."""
    vortices = {}
    # Primary gyre (minimum psi)
    idx_p = np.unravel_index(np.argmin(psi), psi.shape)
    vortices['primary'] = {'x': float(x[idx_p[1]]), 'y': float(y[idx_p[0]]), 'psi': float(psi[idx_p])}

    N = len(x)
    quarter = N // 3

    # Bottom-Right (BR1): positive psi local max
    br_region = psi[:quarter, -quarter:]
    if br_region.max() > 1e-7:
        idx_br = np.unravel_index(np.argmax(br_region), br_region.shape)
        vortices['BR1'] = {
            'x': float(x[N - quarter + idx_br[1]]),
            'y': float(y[idx_br[0]]),
            'psi': float(br_region[idx_br])
        }

    # Bottom-Left (BL1): positive psi local max
    bl_region = psi[:quarter, :quarter]
    if bl_region.max() > 1e-7:
        idx_bl = np.unravel_index(np.argmax(bl_region), bl_region.shape)
        vortices['BL1'] = {
            'x': float(x[idx_bl[1]]),
            'y': float(y[idx_bl[0]]),
            'psi': float(bl_region[idx_bl])
        }

    # Top-Left (TL1): positive psi local max
    tl_region = psi[-quarter:, :quarter]
    if tl_region.max() > 1e-7:
        idx_tl = np.unravel_index(np.argmax(tl_region), tl_region.shape)
        vortices['TL1'] = {
            'x': float(x[idx_tl[1]]),
            'y': float(y[N - quarter + idx_tl[0]]),
            'psi': float(tl_region[idx_tl])
        }

    return vortices


def create_showcase_plots(data, filename=None, save=True, show=True, out_path=None):
    """Create and display 2-panel publication showcase plot with LaTeX fonts and legends below x-axis."""
    _apply_latex_style()

    x, y = data["x"], data["y"]
    if "X" in data and "Y" in data:
        X, Y = data["X"], data["Y"]
    else:
        X, Y = np.meshgrid(x, y, indexing='xy')
    psi, u, v = data["psi"], data["u"], data["v"]
    L = float(x[-1])
    N = len(x)

    # Reynolds number
    if "Re" in data:
        Re_val = int(data["Re"])
    elif filename:
        m = re.search(r"Re(\d+)", str(filename))
        Re_val = int(m.group(1)) if m else 3200
    else:
        Re_val = 3200

    vortices = detect_vortices(psi, x, y)
    p = vortices['primary']

    fig, axs = plt.subplots(1, 2, figsize=(18, 8.5), dpi=300)

    # ==================================================
    # (1) STREAMLINES (Continuous Field, Zero Blank Space)
    # ==================================================
    ax = axs[0]
    psi_min = float(psi.min())
    psi_max = float(psi.max())

    # High-density power-law levels to fill outer wall boundaries & core
    n_neg = 55
    neg_p = np.linspace(0.03, 1.0, n_neg)**2.2
    levels_neg = np.sort(- neg_p * (-psi_min))

    ax.contourf(X, Y, psi, levels=levels_neg, cmap="viridis", alpha=0.9)
    ax.contour(X, Y, psi, levels=levels_neg, colors="black", linewidths=0.5, alpha=0.45)

    # Multi-decade logarithmic levels for all secondary & tertiary eddies
    if psi_max > 1e-9:
        levels_pos = np.logspace(-9, np.log10(max(psi_max, 1e-6)), 35)
        ax.contourf(X, Y, psi, levels=levels_pos, cmap="autumn_r", alpha=0.95)
        ax.contour(X, Y, psi, levels=levels_pos, colors="darkred", linewidths=0.6, alpha=0.6)

    # Clean separatrix
    ax.contour(X, Y, psi, levels=[0.0], colors="white", linewidths=1.5, linestyles="--")

    ax.set_title(r"\textbf{Streamlines} ($\psi$) --- \textbf{Primary, Secondary \& Quaternary Eddies}", fontsize=14)
    ax.set_xlabel(r"$x / L$", fontsize=12)
    ax.set_ylabel(r"$y / L$", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)

    # ==================================================
    # (2) CENTERLINE VELOCITY PROFILE
    # ==================================================
    ax = axs[1]
    x_mid = np.argmin(np.abs(x - L / 2.0))
    ax.plot(u[:, x_mid], y, "k-", lw=2.8, label=rf"$u(y)$ at $x = {L/2:.2f}$ (FDM)")

    if f"u_Re{Re_val}" in GHIA_DATA:
        ghia_u = GHIA_DATA[f"u_Re{Re_val}"]
        ax.scatter(ghia_u, GHIA_DATA["y_u"], facecolors="none", edgecolors="red", s=45, lw=1.5,
                   label=r"Ghia et al. (1982) Benchmark")

    ax.set_title(r"\textbf{Centerline Velocity Profile} $u(y)$", fontsize=14)
    ax.set_xlabel(r"$u / U$", fontsize=12)
    ax.set_ylabel(r"$y / L$", fontsize=12)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=11, framealpha=0.95)

    # ==================================================
    # SUPERTITLE & EXPORT
    # ==================================================
    fig.suptitle(
        rf"\textbf{{Lid-Driven Cavity Flow}} --- $Re = {Re_val}$, \textbf{{Grid}} $= {N} \times {N}$ (\textbf{{Constant Lid}})",
        fontsize=16,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if save:
        if out_path is not None:
            out = out_path
        else:
            out = f"figures/lid_driven_cavity_showcase_Re{Re_val}.png"
        parent_dir = os.path.dirname(out)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[SUCCESS] Saved curated showcase: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    running_in_notebook = "ipykernel" in sys.modules

    if running_in_notebook:
        file_path = "data/flow_fields_Re3200_N257.npz"
        print(f"[INFO] Notebook detected. Using default file: {file_path}")
        show_plot = True
    else:
        parser = argparse.ArgumentParser(
            description="Plot lid-driven cavity solution from .npz file"
        )
        parser.add_argument(
            "--file", type=str, required=True,
            help="Path to .npz file containing saved flow fields"
        )
        parser.add_argument(
            "--no_show", action="store_true",
            help="Save plot to file without displaying interactive GUI window"
        )
        args = parser.parse_args()
        file_path = args.file
        show_plot = not args.no_show

    data = np.load(file_path, allow_pickle=True)
    create_showcase_plots(data, filename=file_path, show=show_plot)
