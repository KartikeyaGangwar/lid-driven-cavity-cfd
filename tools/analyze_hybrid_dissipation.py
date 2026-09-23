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

# Apply clean publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'lines.linewidth': 1.5,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
})


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

    Pex = np.abs(u) * h * Re
    Pey = np.abs(v) * h * Re
    Pemax = np.maximum(Pex, Pey)

    # Effective added numerical viscosity under 1st-order upwinding:
    # Truncation error diffusion is |u|*h/2 in each direction.
    # The hybrid scheme adds artificial viscosity only when |Pe| > 2:
    nu_num_x = np.maximum(0.0, np.abs(u) * h / 2.0 - nu)
    nu_num_y = np.maximum(0.0, np.abs(v) * h / 2.0 - nu)
    nu_num_eff = 0.5 * (nu_num_x + nu_num_y)
    nu_ratio = nu_num_eff / nu

    central_fraction = np.mean(Pemax <= 2.0) * 100.0
    upwind_fraction = np.mean(Pemax > 2.0) * 100.0

    # Center value at (0.5, 0.5)
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

    # 3-Panel Publication Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), dpi=300)
    plt.subplots_adjust(wspace=0.35, bottom=0.24, top=0.88)

    X513, Y513 = np.meshgrid(stats_513['x'], stats_513['y'])

    # Panel (a): Maximum Directional Cell Peclet Number
    ax0 = axes[0]
    levels_pe = np.linspace(0, 100, 21)
    cf0 = ax0.contourf(X513, Y513, np.clip(stats_513['Pemax'], 0, 100), levels=levels_pe, cmap='viridis')
    cs0 = ax0.contour(X513, Y513, stats_513['Pemax'], levels=[2.0], colors='red', linewidths=1.8, linestyles='--')
    ax0.clabel(cs0, fmt={2.0: r'$Pe=2$ (Switch)'}, fontsize=9, colors='red')
    cb0 = fig.colorbar(cf0, ax=ax0, shrink=0.85)
    cb0.set_label(r'$\max(Pe_x, Pe_y)$', fontsize=11)
    ax0.set_title(r'(a) Cell P\'eclet Number $Pe_{h}$ ($513^2$)', fontsize=12, pad=8)
    ax0.set_xlabel(r'$x / L$', fontsize=11)
    ax0.set_ylabel(r'$y / L$', fontsize=11)
    ax0.set_aspect('equal')

    # Panel (b): Numerical-to-Molecular Viscosity Ratio nu_num / nu
    ax1 = axes[1]
    levels_nu = np.linspace(0, 50, 21)
    cf1 = ax1.contourf(X513, Y513, np.clip(stats_513['nu_ratio'], 0, 50), levels=levels_nu, cmap='inferno')
    cs1 = ax1.contour(X513, Y513, stats_513['nu_ratio'], levels=[1.0, 5.0, 20.0], colors='white', linewidths=1.0, alpha=0.8)
    ax1.clabel(cs1, fmt='%1.0fx', fontsize=8, colors='white')
    cb1 = fig.colorbar(cf1, ax=ax1, shrink=0.85)
    cb1.set_label(r'$\nu_{\mathrm{num}} / \nu$', fontsize=11)
    ax1.set_title(r'(b) Numerical Dissipation Ratio $\nu_{\mathrm{num}}/\nu$', fontsize=12, pad=8)
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

    ax2.plot(pe_cut_513, y513, 'b-', linewidth=1.5, label=r'$Pe$ ($513\times 513$)')
    ax2.plot(pe_cut_1025, y1025, 'b--', linewidth=1.5, label=r'$Pe$ ($1025\times 1025$)')
    ax2.axvline(2.0, color='red', linestyle=':', linewidth=1.3, label=r'Switch ($Pe=2$)')
    ax2.set_xlabel(r'Cell P\'eclet Number $Pe_x(0.5, y)$', color='blue', fontsize=11)
    ax2.tick_params(axis='x', labelcolor='blue')
    ax2.set_ylabel(r'Vertical Station $y / L$', fontsize=11)
    ax2.set_xlim(-1, 55)
    ax2.grid(True, linestyle=':', alpha=0.5)

    ax2_twin = ax2.twiny()
    ax2_twin.plot(nu_cut_513, y513, color='purple', linestyle='-', linewidth=1.5, label=r'$\nu_{\mathrm{num}}/\nu$ ($513^2$)')
    ax2_twin.plot(nu_cut_1025, y1025, color='purple', linestyle='--', linewidth=1.5, label=r'$\nu_{\mathrm{num}}/\nu$ ($1025^2$)')
    ax2_twin.set_xlabel(r'Dissipation Ratio $\nu_{\mathrm{num}} / \nu$', color='purple', fontsize=11)
    ax2_twin.tick_params(axis='x', labelcolor='purple')
    ax2_twin.set_xlim(-0.5, 25)

    # Combined legend strictly below x-axis
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.22),
               ncol=2, frameon=True, fancybox=True, edgecolor='#cccccc', fontsize=8.5)
    ax2.set_title(r'(c) Mid-Plane Profiles ($x = 0.5$)', fontsize=12, pad=8)

    out_path = 'figures/lid_driven_peclet_dissipation_map_Re100000.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SAVED] Generated publication figure: {out_path}\n")


if __name__ == '__main__':
    generate_figures()
