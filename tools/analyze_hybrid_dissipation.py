"""
Quantitative Analysis & Spatial Mapping of Hybrid Convective Differencing
and Numerical Dissipation at Extreme Reynolds Number (Re = 100,000).

Addresses Technical Review Finding 1 by providing exact spatial distributions
of the local cell Peclet number and effective numerical viscosity ratio nu_num / nu.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lid_driven_cavity_fdm import _apply_latex_style


def analyze_hybrid_scheme(npz_path):
    data = np.load(npz_path)
    u = data['u']
    v = data['v']
    x = data['x']
    y = data['y']
    Re = float(data['Re'])
    N = int(data['N'])
    h = 1.0 / (N - 1)
    nu = 1.0 / Re

    # Signed cell Peclet numbers and magnitudes
    Pe_tilde_x = u * h / nu
    Pe_tilde_y = v * h / nu
    Pex = np.abs(Pe_tilde_x)
    Pey = np.abs(Pe_tilde_y)
    Pemax = np.maximum(Pex, Pey)

    # Added artificial numerical viscosity under operator-split ADI upwinding:
    # When Pe <= 2, pure central differencing preserves zero artificial diffusion (nu_art = 0).
    # When Pe > 2, 1st-order upwinding introduces leading truncation diffusion |u|*h/2.
    # Because molecular diffusion nu is retained in the ADI implicit solve, the added viscosity is:
    # nu_art_x = (|u|*h/2) * 1_{Pe_x > 2}, nu_art_y = (|v|*h/2) * 1_{Pe_y > 2}.
    # The directional average added viscosity ratio is:
    # bar_nu_art / nu = (1/4) * [ Pe_x * 1_{Pe_x > 2} + Pe_y * 1_{Pe_y > 2} ].
    nu_art_x = np.where(Pex > 2.0, np.abs(u) * h / 2.0, 0.0)
    nu_art_y = np.where(Pey > 2.0, np.abs(v) * h / 2.0, 0.0)
    nu_num_eff = 0.5 * (nu_art_x + nu_art_y)
    nu_ratio = nu_num_eff / nu

    central_fraction = np.mean(Pemax <= 2.0) * 100.0
    upwind_fraction = np.mean(Pemax > 2.0) * 100.0

    # Center value at geometric center (0.5, 0.5)
    ci, cj = N // 2, N // 2
    pe_center = float(Pemax[ci, cj])
    nu_ratio_center = float(nu_ratio[ci, cj])

    stats = {
        'N': N,
        'Re': Re,
        'h': h,
        'nu': nu,
        'central_pct': central_fraction,
        'upwind_pct': upwind_fraction,
        'pe_max': float(np.max(Pemax)),
        'pe_center': pe_center,
        'nu_ratio_max': float(np.max(nu_ratio)),
        'nu_ratio_center': nu_ratio_center,
        'nu_ratio_mean': float(np.mean(nu_ratio)),
        'Pex': Pex,
        'Pey': Pey,
        'Pemax': Pemax,
        'nu_ratio': nu_ratio,
        'x': x,
        'y': y,
        'u': u,
        'v': v,
    }
    return stats


def generate_figures():
    os.makedirs('figures', exist_ok=True)
    
    stats_513 = analyze_hybrid_scheme('data/flow_fields_Re100000_N513.npz')
    stats_1025 = analyze_hybrid_scheme('data/flow_fields_Re100000_N1025.npz')

    print("=" * 70)
    print("HYBRID DIFFERENCING & NUMERICAL DISSIPATION AUDIT (Re = 100,000)")
    print("=" * 70)
    for s in [stats_513, stats_1025]:
        print(f"Grid: {s['N']}x{s['N']} (h = {s['h']:.6f}) | Re = {s['Re']:.0f}")
        print(f"  Central Differencing Area (Pe <= 2): {s['central_pct']:6.2f}%")
        print(f"  Upwind Differencing Area (Pe > 2):   {s['upwind_pct']:6.2f}%")
        print(f"  Max Cell Peclet:                    {s['pe_max']:6.2f}")
        print(f"  Peclet at Geometric Center (0.5,0.5):{s['pe_center']:6.2f}")
        print(f"  Max Numerical Viscosity Ratio:      {s['nu_ratio_max']:6.2f}x nu")
        print(f"  Center Viscosity Ratio (0.5, 0.5):  {s['nu_ratio_center']:6.2f}x nu")
        print(f"  Mean Viscosity Ratio over Cavity:   {s['nu_ratio_mean']:6.2f}x nu")
        print("-" * 70)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lid_driven_cavity_fdm import _apply_latex_style
    _apply_latex_style()

    # 3-Panel Publication Figure with perfectly aligned subplots
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0), dpi=300)
    plt.subplots_adjust(wspace=0.34, bottom=0.22, top=0.88, left=0.06, right=0.96)

    X513, Y513 = np.meshgrid(stats_513['x'], stats_513['y'])

    # Panel (a): Maximum Directional Cell Peclet Number
    ax0 = axes[0]
    levels_pe = np.linspace(0, 100, 21)
    cf0 = ax0.contourf(X513, Y513, np.clip(stats_513['Pemax'], 0, 100), levels=levels_pe, cmap='viridis')
    cs0 = ax0.contour(X513, Y513, stats_513['Pemax'], levels=[2.0], colors='red', linewidths=1.6, linestyles='--')
    cb0 = fig.colorbar(cf0, ax=ax0, shrink=0.82, pad=0.04)
    cb0.set_label(r'$\max(Pe_x, Pe_y)$', fontsize=11)
    ax0.set_title(r'Cell P\'eclet Number $Pe_{h}$ ($513 \times 513$)', fontsize=12, pad=10)
    ax0.set_xlabel(r'$x / L$', fontsize=11)
    ax0.set_ylabel(r'$y / L$', fontsize=11)
    ax0.set_aspect('equal')

    # Panel (b): Numerical-to-Molecular Viscosity Ratio nu_num / nu
    ax1 = axes[1]
    levels_nu = np.linspace(0, 50, 21)
    cf1 = ax1.contourf(X513, Y513, np.clip(stats_513['nu_ratio'], 0, 50), levels=levels_nu, cmap='inferno')
    cs1 = ax1.contour(X513, Y513, stats_513['nu_ratio'], levels=[1.0, 5.0, 20.0], colors='white', linewidths=1.0, alpha=0.8)
    ax1.clabel(cs1, fmt=r'%1.0f$\times$', fontsize=8, colors='white')
    cb1 = fig.colorbar(cf1, ax=ax1, shrink=0.82, pad=0.04)
    cb1.set_label(r'$\nu_{\mathrm{num}} / \nu$', fontsize=11)
    ax1.set_title(r'Numerical Dissipation $\nu_{\mathrm{num}}/\nu$ ($513 \times 513$)', fontsize=12, pad=10)
    ax1.set_xlabel(r'$x / L$', fontsize=11)
    ax1.set_ylabel(r'$y / L$', fontsize=11)
    ax1.set_aspect('equal')

    # Panel (c): Vertical Centerline Profiles (x = 0.5) comparing 513 vs 1025
    ax2 = axes[2]
    mid_j_513 = len(stats_513['x']) // 2
    mid_j_1025 = len(stats_1025['x']) // 2

    y513 = stats_513['y']
    y1025 = stats_1025['y']
    pe_cut_513 = stats_513['Pemax'][:, mid_j_513]
    pe_cut_1025 = stats_1025['Pemax'][:, mid_j_1025]
    nu_cut_513 = stats_513['nu_ratio'][:, mid_j_513]
    nu_cut_1025 = stats_1025['nu_ratio'][:, mid_j_1025]

    # Full range from 0 to 105 so curves are not clipped at boundaries
    l_pe1, = ax2.plot(pe_cut_513, y513, color='#1f77b4', linestyle='-', linewidth=1.6, label=r'$Pe$ ($513 \times 513$)')
    l_pe2, = ax2.plot(pe_cut_1025, y1025, color='#1f77b4', linestyle='--', linewidth=1.6, label=r'$Pe$ ($1025 \times 1025$)')
    l_sw = ax2.axvline(2.0, color='red', linestyle=':', linewidth=1.3, label=r'Threshold ($Pe=2$)')
    ax2.set_xlabel(r'Cell P\'eclet Number $Pe_x(0.5, y)$', fontsize=11)
    ax2.set_ylabel(r'$y / L$', fontsize=11)
    ax2.set_xlim(0, 105)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle=':', alpha=0.5)

    ax2_twin = ax2.twiny()
    l_nu1, = ax2_twin.plot(nu_cut_513, y513, color='#8c564b', linestyle='-', linewidth=1.6, label=r'$\nu_{\mathrm{num}}/\nu$ ($513 \times 513$)')
    l_nu2, = ax2_twin.plot(nu_cut_1025, y1025, color='#8c564b', linestyle='--', linewidth=1.6, label=r'$\nu_{\mathrm{num}}/\nu$ ($1025 \times 1025$)')
    ax2_twin.set_xlabel(r'Dissipation Ratio $\nu_{\mathrm{num}} / \nu$', fontsize=11)
    ax2_twin.set_xlim(0, 60)

    # Combined clean legend below panel (c)
    handles = [l_pe1, l_pe2, l_sw, l_nu1, l_nu2]
    labels = [h.get_label() for h in handles]
    ax2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.20),
               ncol=2, frameon=True, fancybox=True, edgecolor='#cccccc', fontsize=8.5)
    ax2.set_title(r'Mid-Plane Profiles ($x = 0.5$)', fontsize=12, pad=10)

    # Match panel (c) vertical geometry to equal-aspect panels (a) and (b)
    fig.canvas.draw()
    pos0 = ax0.get_position()
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0, pos0.y0, pos2.width, pos0.height])
    ax2_twin.set_position([pos2.x0, pos0.y0, pos2.width, pos0.height])

    out_path = 'figures/lid_driven_peclet_dissipation_map_Re100000.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SAVED] Generated publication figure: {out_path}\n")


if __name__ == '__main__':
    generate_figures()
