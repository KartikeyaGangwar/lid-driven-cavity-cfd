"""
Ultra-Resolution HPC Suite for 2D Lid-Driven Cavity Flow (N = 1025 & N = 2049)
Scalable from Re = 10,000 to Re = 100,000 (1.05 Million to 4.2 Million Collocation Nodes)

Features:
- Bicubic spline prolongation from converged N=513 flow fields to N=1025 / N=2049.
- Zero-dissipation central differencing (nu_num = 0) with spectral DST-I Poisson solver.
- Steady continuation ladder up to Re = 100,000.
- Rigorous validation against Erturk (acenumerics.com Table 3: omega at y=0.95).
- Physical time-accurate unsteady marching (Hopf bifurcations, Strouhal number, phase portraits).
- 60-frame synchronized publication-grade animated GIF generation.

Author: Kartikey Singh
Year: 2026
License: MIT
"""

import os
import sys
import io
import time
import argparse
import numpy as np
import scipy.fft as sfft
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
import matplotlib.pyplot as plt
from PIL import Image

# Import base classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lid_driven_cavity_fdm import LidDrivenCavitySolver, _apply_latex_style, GHIA_DATA


# Reference data from Dr. Ercan Erturk (acenumerics.com Table 3 corrected, omega at y=0.95)
ERTURK_TABLE3 = {
    'x': np.array([1.0000, 0.9900, 0.9800, 0.9700, 0.9688, 0.9609, 0.9531, 0.9453,
                   0.9063, 0.8594, 0.8047, 0.6800, 0.5000, 0.3500, 0.2344, 0.2266,
                   0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0400, 0.0300, 0.0200,
                   0.0100, 0.0000]),
    'Re1000': np.array([83.82035, 28.93863, -12.58720, -18.69105, -18.32549, -15.20862, -12.89386, -11.48036,
                        -8.64711, -7.40476, -6.66338, -5.61582, -4.16691, -2.77107, -1.79633, -1.73873,
                        -1.20497, -0.33969, 0.02007, 0.22391, 0.44502, 1.27700, 1.97216, 3.32707, 5.88274, 9.42476]),
    'Re5000': np.array([164.5439, -15.48533, -23.09050, -15.68747, -15.09871, -12.00901, -9.90230, -8.40222,
                        -4.89364, -3.19578, -2.01511, -0.87921, -0.52943, 0.48950, 3.36999, 3.62983,
                        4.80194, 3.96155, 3.69877, 3.57614, 3.46299, 3.19687, 3.02777, 2.44848, 0.53450, -3.61013]),
    'Re7500': np.array([190.0801, -30.47677, -22.38487, -14.31540, -13.64843, -10.23672, -8.04562, -6.56918,
                        -3.31374, -1.81584, -1.06392, -0.92833, -1.01443, 0.09164, 4.44518, 4.84786,
                        5.35532, 3.86723, 3.61692, 3.52100, 3.44896, 3.39112, 3.39402, 3.04789, 0.16030, -8.60647]),
    'Re10000': np.array([208.6591, -37.75954, -21.64013, -12.87952, -12.17661, -8.69272, -6.57244, -5.19755,
                         -2.26655, -1.11327, -0.84706, -1.22799, -1.39683, -0.31583, 5.48332, 6.00703,
                         5.12516, 3.41940, 3.24928, 3.20714, 3.19528, 3.35613, 3.50997, 3.45164, 0.32588, -12.67382]),
    'Re20000': np.array([256.1678, -43.74377, -17.91398, -8.40550, -7.74710, -4.75306, -3.14088, -2.17012,
                         -0.60060, -0.88994, -1.43970, -1.88387, -1.96576, -1.46658, 9.33041, 9.63018,
                         2.86309, 2.35047, 2.41788, 2.45689, 2.50803, 2.87939, 3.31107, 3.95028, 2.17407, -24.32426]),
    'Re25000': np.array([273.3864, -43.69902, -16.14544, -6.87469, -6.26765, -3.57529, -2.17635, -1.36717,
                         -0.47653, -1.15565, -1.68308, -1.95128, -1.98142, -1.75677, 10.74982, 10.38588,
                         2.28026, 2.43886, 2.46935, 2.47963, 2.49477, 2.72977, 3.14237, 3.93747, 3.01380, -28.75590]),
    'Re30000': np.array([288.7116, -43.27584, -14.55303, -5.65336, -5.09666, -2.67522, -1.46208, -0.80476,
                         -0.54036, -1.39968, -1.82947, -1.95677, -1.95692, -1.92107, 11.63114, 10.40221,
                         2.12431, 2.58025, 2.57846, 2.57037, 2.56095, 2.65700, 3.00126, 3.86737, 3.67829, -32.73478]),
}


def prolong_flow_field(source_path, N_target=1025, L=1.0):
    """
    Bicubic spline prolongation from a coarse/intermediate solution (e.g. N=513)
    to an ultra-fine mesh (N=1025 or N=2049) with rigorous boundary enforcement.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source solution file not found: {source_path}")
    
    print(f"[PROLONG] Loading source file: {source_path} ...", flush=True)
    src = np.load(source_path)
    x_src = src['x']
    y_src = src['y']
    N_src = len(x_src)
    print(f"[PROLONG] Resampling from {N_src}x{N_src} -> {N_target}x{N_target} ({N_target**2:,} nodes)...", flush=True)

    x_tgt = np.linspace(0.0, L, N_target)
    y_tgt = np.linspace(0.0, L, N_target)
    X_tgt, Y_tgt = np.meshgrid(x_tgt, y_tgt, indexing='xy')
    query_pts = np.stack([Y_tgt.ravel(), X_tgt.ravel()], axis=-1)

    fields_prolonged = {}
    for var in ['psi', 'omega', 'u', 'v']:
        if var in src:
            interp = RegularGridInterpolator((y_src, x_src), src[var], method='cubic',
                                             bounds_error=False, fill_value=None)
            fields_prolonged[var] = interp(query_pts).reshape((N_target, N_target))

    # Exact boundary condition enforcement
    fields_prolonged['psi'][0, :] = 0.0
    fields_prolonged['psi'][-1, :] = 0.0
    fields_prolonged['psi'][:, 0] = 0.0
    fields_prolonged['psi'][:, -1] = 0.0

    fields_prolonged['u'][0, :] = 0.0
    fields_prolonged['u'][-1, :] = 1.0  # Constant moving lid
    fields_prolonged['u'][:, 0] = 0.0
    fields_prolonged['u'][:, -1] = 0.0

    fields_prolonged['v'][0, :] = 0.0
    fields_prolonged['v'][-1, :] = 0.0
    fields_prolonged['v'][:, 0] = 0.0
    fields_prolonged['v'][:, -1] = 0.0

    print("[PROLONG] Bicubic prolongation completed successfully.", flush=True)
    return fields_prolonged, x_tgt, y_tgt


def solve_ultra_steady(Re, N=1025, source_file=None, max_iter=25000, tol=1e-6,
                       check_interval=200, out_dir="data", fig_dir="figures"):
    """
    Executes high-resolution steady continuation on N=1025 (1.05M nodes) or N=2049.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"flow_fields_Re{Re}_N{N}.npz")

    print("=" * 80)
    print(f"ULTRA-RESOLUTION STEADY CONTINUATION: Re = {Re:,} on Mesh {N}x{N} ({N*N:,} nodes)")
    print("=" * 80)

    # Initialize solver
    solver = LidDrivenCavitySolver(
        N=N, Re=Re, lid_velocity=1.0, L=1.0, lid_profile='constant',
        poisson_solver='dst', convection_scheme='central', wall_bc='thom'
    )

    # Warm-start prolongation
    if source_file and os.path.exists(source_file):
        prolonged, _, _ = prolong_flow_field(source_file, N_target=N)
        solver.psi = prolonged['psi']
        solver.omega = prolonged['omega']
        solver.u = prolonged['u']
        solver.v = prolonged['v']
        solver.u_c = solver.u[1:-1, 1:-1].copy()
        solver.v_c = solver.v[1:-1, 1:-1].copy()
        print(f"[WARM-START] Injected warm-start fields from: {source_file}")
    else:
        print("[INFO] Starting from quiescent initial condition.")

    # Time-stepping relaxation
    t_start = time.perf_counter()
    converged = False
    
    for it in range(1, max_iter + 1):
        omega_prev = solver.omega.copy()
        solver.step()

        if it % check_interval == 0:
            diff = np.max(np.abs(solver.omega - omega_prev))
            psi_min = np.min(solver.psi)
            elapsed = time.perf_counter() - t_start
            sps = it / elapsed if elapsed > 0 else 0
            print(f"  [Iter {it:6d}] Max dOmega: {diff:.3e} | psi_min: {psi_min:.5f} | Speed: {sps:.1f} steps/s", flush=True)

            if diff < tol:
                print(f"[CONVERGED] Reached tolerance {tol:.1e} at iteration {it} in {elapsed:.1f}s!", flush=True)
                converged = True
                break

    if not converged:
        print(f"[NOTICE] Finished maximum iterations ({max_iter}). Saving current state.")

    # Save flow field
    np.savez_compressed(
        out_file,
        x=solver.x, y=solver.y,
        psi=solver.psi, omega=solver.omega,
        u=solver.u, v=solver.v,
        Re=Re, N=N, lid_velocity=solver.U
    )
    print(f"[SAVED] Saved flow field archive to: {out_file}")

    # Generate showcase plot
    vortices = solver.get_vortex_centers()
    p = vortices['primary']
    psi_val = p.get('psi', p.get('psi_min', 0.0))
    print(f"[EDDY RESULTS] Primary Vortex Center: ({p['x']:.5f}, {p['y']:.5f}) | psi_min: {psi_val:.5f}")
    if 'BR1' in vortices:
        br = vortices['BR1']
        br_psi = br.get('psi', br.get('psi_max', 0.0))
        print(f"               BR1 Vortex Center:    ({br['x']:.5f}, {br['y']:.5f}) | psi_max: {br_psi:.3e}")
    if 'BL1' in vortices:
        bl = vortices['BL1']
        bl_psi = bl.get('psi', bl.get('psi_max', 0.0))
        print(f"               BL1 Vortex Center:    ({bl['x']:.5f}, {bl['y']:.5f}) | psi_max: {bl_psi:.3e}")
    if 'TL1' in vortices:
        tl = vortices['TL1']
        tl_psi = tl.get('psi', tl.get('psi_max', 0.0))
        print(f"               TL1 Vortex Center:    ({tl['x']:.5f}, {tl['y']:.5f}) | psi_max: {tl_psi:.3e}")

    # Verification against Erturk Table 3 if applicable
    verify_against_erturk_table3(out_file)

    return solver, out_file


def verify_against_erturk_table3(npz_file):
    """
    Evaluates computed vorticity profile at y = 0.95 against Erturk (2009, Table 3).
    """
    data = np.load(npz_file)
    Re_val = int(data['Re'])
    key = f"Re{Re_val}"
    if key not in ERTURK_TABLE3:
        print(f"[BENCHMARK] No Erturk Table 3 data available for Re = {Re_val}. Skipping.")
        return None

    x_ref = ERTURK_TABLE3['x']
    ref_omega = ERTURK_TABLE3[key]
    omega = data['omega']
    x_grid = data['x']
    y_grid = data['y']
    N = len(x_grid)

    spline = RectBivariateSpline(y_grid, x_grid, omega, kx=3, ky=3)
    comp_omega = np.array([float(spline(0.95, xp)[0, 0]) for xp in x_ref])

    diff = np.abs(comp_omega - ref_omega)
    interior_mask = (x_ref > 0.0) & (x_ref < 1.0)
    int_l2 = np.sqrt(np.mean(diff[interior_mask]**2))
    int_mae = np.mean(diff[interior_mask])

    print("\n" + "-" * 75)
    print(f"BENCHMARK AUDIT: Erturk (acenumerics Table 3) at y = 0.95 for Re = {Re_val:,} (N={N})")
    print("-" * 75)
    print(f"  Cavity Interior L2 Error (0 < x < 1): {int_l2:8.4f}")
    print(f"  Cavity Interior MAE:                 {int_mae:8.4f}")
    print(f"  Mid-Plane x = 0.5000: Computed = {comp_omega[12]:8.4f} | Ref = {ref_omega[12]:8.4f} | Delta = {diff[12]:.4f}")
    print(f"  Left Core x = 0.1563: Computed = {comp_omega[16]:8.4f} | Ref = {ref_omega[16]:8.4f} | Delta = {diff[16]:.4f}")
    print(f"  Right Wall x = 1.0000: Computed = {comp_omega[0]:8.4f} | Ref = {ref_omega[0]:8.4f} | Delta = {diff[0]:.4f}")
    print("-" * 75 + "\n")

    return {'int_l2': int_l2, 'int_mae': int_mae, 'comp_omega': comp_omega, 'diff': diff}


def run_ultra_unsteady(Re, N=1025, base_npz=None, dt=0.0005, total_steps=20000,
                       out_dir="data", fig_dir="figures", make_gif=True, gif_frames=60):
    """
    Physical time-accurate marching on N=1025 to capture Hopf shedding and attractors.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 80)
    print(f"ULTRA-RESOLUTION UNSTEADY MARCHING: Re = {Re:,} on Mesh {N}x{N} ({N*N:,} nodes)")
    print("=" * 80)

    # Initialize solver with robust wall relaxation (0.50 for high Re stability)
    solver = LidDrivenCavitySolver(
        N=N, Re=Re, lid_velocity=1.0, L=1.0, lid_profile='constant',
        poisson_solver='dst', convection_scheme='central', wall_bc='thom',
        wall_beta=0.50 if Re >= 5000 else 0.60
    )
    # Ensure dt strictly respects the viscous and convective CFL bounds
    if dt is not None and dt < solver.dt:
        solver.dt = dt
        solver.alpha_adi = (solver.nu * solver.dt) / (2.0 * solver.h**2)
        solver._init_adi_coefficients()
    dt = solver.dt

    # Load initial condition
    if base_npz and os.path.exists(base_npz):
        data = np.load(base_npz)
        if len(data['x']) == N:
            solver.psi = data['psi'].copy()
            solver.omega = data['omega'].copy()
            solver.u = data['u'].copy()
            solver.v = data['v'].copy()
        else:
            prolonged, _, _ = prolong_flow_field(base_npz, N_target=N)
            solver.psi = prolonged['psi']
            solver.omega = prolonged['omega']
            solver.u = prolonged['u']
            solver.v = prolonged['v']
    else:
        # Check if default steady file exists
        default_base = os.path.join(out_dir, f"flow_fields_Re{Re}_N{N}.npz")
        if os.path.exists(default_base):
            data = np.load(default_base)
            solver.psi = data['psi'].copy()
            solver.omega = data['omega'].copy()
            solver.u = data['u'].copy()
            solver.v = data['v'].copy()
        else:
            raise FileNotFoundError(f"Base steady state not found for Re={Re}, N={N}")

    # Seed gentle shear-layer perturbation only near critical bifurcation (Re <= 15,000)
    # At Re >= 25,000, flow is naturally intrinsically unstable and sheds dynamically
    if Re <= 15000:
        X, Y = solver.X, solver.Y
        perturb = 0.02 * np.exp(-((X - 0.8)**2 + (Y - 0.8)**2) / (0.05**2))
        solver.omega += perturb
        solver.solve_streamfunction()
        solver.calculate_velocities()

    # Setup telemetry probe at BL secondary detachment zone (x=0.08, y=0.15)
    j_probe = int(np.argmin(np.abs(solver.x - 0.08)))
    i_probe = int(np.argmin(np.abs(solver.y - 0.15)))

    t_arr = []
    u_bl_arr = []
    v_bl_arr = []

    print(f"[MARCH] Advancing {total_steps} physical time steps with dt = {dt:.4e} (Total t = {total_steps*dt:.2f}s)...")
    t0 = time.perf_counter()

    for step in range(1, total_steps + 1):
        solver.step()

        cur_t = step * dt
        t_arr.append(cur_t)
        u_bl_arr.append(solver.u[i_probe, j_probe])
        v_bl_arr.append(solver.v[i_probe, j_probe])

        if step % 500 == 0:
            elapsed = time.perf_counter() - t0
            sps = step / elapsed if elapsed > 0 else 0
            print(f"  [Step {step:6d}/{total_steps}] t = {cur_t:6.3f}s | u_BL = {u_bl_arr[-1]:+.5f} | Speed = {sps:.1f} steps/s", flush=True)

    t_arr = np.array(t_arr)
    u_bl_arr = np.array(u_bl_arr)
    v_bl_arr = np.array(v_bl_arr)

    # FFT Spectral Analysis
    t_settled_idx = int(len(t_arr) * 0.4)
    t_sig = t_arr[t_settled_idx:]
    v_sig = v_bl_arr[t_settled_idx:] - np.mean(v_bl_arr[t_settled_idx:])
    
    dt_eff = np.mean(np.diff(t_sig))
    freqs = np.fft.rfftfreq(len(v_sig), d=dt_eff)
    fft_amp = np.abs(np.fft.rfft(v_sig)) / len(v_sig)
    
    idx_peak = np.argmax(fft_amp[1:]) + 1
    f0 = freqs[idx_peak]
    St = f0 * solver.L / solver.U
    T_period = 1.0 / f0 if f0 > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"SPECTRAL ANALYSIS: Dominant Frequency f0 = {f0:.4f} Hz | St = {St:.4f} | Period T = {T_period:.4f} s")
    print("=" * 60 + "\n")

    # Save unsteady trajectory
    unsteady_out = os.path.join(out_dir, f"unsteady_Re{Re}_N{N}.npz")
    np.savez_compressed(
        unsteady_out,
        t=t_arr, u_bl=u_bl_arr, v_bl=v_bl_arr,
        freqs=freqs, fft_amp=fft_amp,
        f0=f0, St=St, T_period=T_period,
        psi=solver.psi, omega=solver.omega,
        Re=Re, N=N
    )
    print(f"[SAVED] Saved unsteady telemetry data to: {unsteady_out}")

    # Generate Animated GIF if requested
    if make_gif:
        eff_period = T_period if (f0 > 0 and 0.01 <= T_period <= 0.5) else 0.05
        gif_out = os.path.join(fig_dir, f"lid_driven_vortex_shedding_Re{Re}_N{N}.gif")
        generate_synchronized_gif(solver, t_arr, u_bl_arr, v_bl_arr, eff_period,
                                  n_frames=gif_frames, out_path=gif_out)

    return unsteady_out


def generate_synchronized_gif(solver, t_history, u_hist, v_hist, T_period,
                              n_frames=60, out_path="vortex_shedding.gif"):
    """
    Renders synchronized 60-frame publication GIF over one complete fundamental cycle.
    """
    print(f"[GIF] Generating {n_frames}-frame synchronized animation over T = {T_period:.3f}s ...", flush=True)
    steps_per_frame = max(1, min(35, int(round((T_period / solver.dt) / n_frames))))
    
    frames = []
    t_cur = t_history[-1]
    probe_t = list(t_history[-1200:])
    probe_v = list(v_hist[-1200:])
    
    # Downsample grid for fast rendering if N > 513
    stride = 2 if solver.N > 513 else 1
    X_sub = solver.X[::stride, ::stride]
    Y_sub = solver.Y[::stride, ::stride]

    for frame_idx in range(n_frames):
        for _ in range(steps_per_frame):
            solver.step()
            t_cur += solver.dt
            probe_t.append(t_cur)
            probe_v.append(solver.v[int(np.argmin(np.abs(solver.y - 0.15))),
                                    int(np.argmin(np.abs(solver.x - 0.08)))])

        # Render frame
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=140)
        
        # Panel 1: Vorticity & Streamlines with SymLogNorm and Dense Contours
        ax1 = axs[0]
        from matplotlib.colors import SymLogNorm
        omega_sub = solver.omega[::stride, ::stride]
        psi_sub = solver.psi[::stride, ::stride]
        psi_min = float(psi_sub.min())
        psi_max = float(psi_sub.max())
        
        v_disp = 120.0
        norm_v = SymLogNorm(linthresh=1.0, linscale=0.5, vmin=-v_disp, vmax=v_disp, base=10)
        levels_v = np.unique(np.concatenate([
            -np.logspace(np.log10(v_disp), np.log10(0.1), 32),
            np.linspace(-0.1, 0.1, 9),
            np.logspace(np.log10(0.1), np.log10(v_disp), 32)
        ]))
        
        im = ax1.contourf(X_sub, Y_sub, omega_sub, levels=levels_v, norm=norm_v,
                          cmap='RdBu_r', extend='both')
        
        # Dense streamlines (zero blank space)
        neg_p = np.linspace(0.04, 1.0, 35)**2.2
        levels_neg = np.sort(- neg_p * (-psi_min))
        ax1.contour(X_sub, Y_sub, psi_sub, levels=levels_neg, colors='black', linewidths=0.45, alpha=0.4)
        if psi_max > 1e-9:
            levels_pos = np.logspace(-9, np.log10(max(psi_max, 1e-6)), 20)
            ax1.contour(X_sub, Y_sub, psi_sub, levels=levels_pos, colors='darkred', linewidths=0.5, alpha=0.55)
        ax1.contour(X_sub, Y_sub, psi_sub, levels=[0.0], colors='white', linewidths=1.2, linestyles='--')

        ax1.plot(0.08, 0.15, 'go', markersize=8, label='Probe (0.08, 0.15)')
        ax1.set_title(rf"Vorticity $\omega$ (SymLog) \& Streamlines ($Re = {solver.Re:,}$)")
        ax1.set_xlabel("$x / L$")
        ax1.set_ylabel("$y / L$")
        ax1.set_aspect('equal')
        ax1.legend(loc='lower right', fontsize=8)

        # Panel 2: Telemetry
        ax2 = axs[1]
        t_window = np.array(probe_t[-400:])
        v_window = np.array(probe_v[-400:])
        ax2.plot(t_window, v_window, 'b-', linewidth=1.5, label=r'$v_{\mathrm{BL}}(t)$')
        ax2.plot(t_window[-1], v_window[-1], 'ro', markersize=8, label='Current Phase')
        ax2.set_title("Boundary Layer Probe Telemetry")
        ax2.set_xlabel("Time $t$ (s)")
        ax2.set_ylabel(r"Vertical Velocity $v_{\mathrm{BL}}$")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=8)

        plt.suptitle(rf"2D Lid-Driven Cavity Vortex Shedding ($t = {t_cur:.3f}$ s, Frame {frame_idx+1}/{n_frames})",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        # Save buffer to PIL
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=140)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    # Save GIF
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / 14),
        loop=0
    )
    print(f"[GIF COMPLETE] Saved animated GIF to: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    import io
    parser = argparse.ArgumentParser(description="Ultra-Resolution HPC Lid-Driven Cavity CFD Suite (N=1025/N=2049)")
    parser.add_argument("--mode", type=str, default="steady", choices=["steady", "unsteady", "prolong", "benchmark", "ladder"],
                        help="Operation mode: 'steady', 'unsteady', 'prolong', 'benchmark', 'ladder'")
    parser.add_argument("--re", type=int, default=10000, help="Target Reynolds number")
    parser.add_argument("--n", type=int, default=1025, help="Target grid resolution (default 1025)")
    parser.add_argument("--source", type=str, default=None, help="Path to source coarse/intermediate .npz solution")
    parser.add_argument("--max_iter", type=int, default=15000, help="Maximum steady iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance for steady solve")
    parser.add_argument("--unsteady_steps", type=int, default=15000, help="Number of unsteady time steps")
    parser.add_argument("--dt", type=float, default=0.0004, help="Physical time step size for unsteady solve")
    parser.add_argument("--gif", action="store_true", help="Compile synchronized animated GIF")
    args = parser.parse_args()

    if args.mode == "steady":
        src = args.source
        if src is None:
            # Try auto-detecting N=513 file for same Re
            candidate = f"data/flow_fields_Re{args.re}_N513.npz"
            if os.path.exists(candidate):
                src = candidate
        solve_ultra_steady(args.re, N=args.n, source_file=src, max_iter=args.max_iter, tol=args.tol)

    elif args.mode == "unsteady":
        run_ultra_unsteady(args.re, N=args.n, base_npz=args.source, dt=args.dt,
                           total_steps=args.unsteady_steps, make_gif=args.gif)

    elif args.mode == "benchmark":
        npz = args.source if args.source else f"data/flow_fields_Re{args.re}_N{args.n}.npz"
        verify_against_erturk_table3(npz)

    elif args.mode == "ladder":
        # Execute continuation ladder up to extreme Re
        ladder_re = [10000, 20000, 30000, 40000, 50000, 75000, 100000]
        curr_source = f"data/flow_fields_Re10000_N513.npz"
        for r in ladder_re:
            print(f"\n[LADDER] Advancing to Re = {r} on N={args.n}...")
            _, out_f = solve_ultra_steady(r, N=args.n, source_file=curr_source,
                                          max_iter=args.max_iter, tol=args.tol)
            curr_source = out_f
