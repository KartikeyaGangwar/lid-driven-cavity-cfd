"""
Dynamic Vortex Shedding Animation Generator for 2D Lid-Driven Cavity Flow.

Simulates time-accurate periodic vortex shedding post-Hopf bifurcation and
compiles publication-grade, synchronized multi-panel animated GIFs.

Usage:
    python animate_shedding.py --Re 10000 --N 257 --n_frames 50 --fps 14
"""

import os
import sys
import io
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Unicode terminal fix
if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    try:
        current_encoding = getattr(sys.stdout, 'encoding', None)
        if current_encoding and current_encoding.lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from lid_driven_cavity_fdm import LidDrivenCavitySolver, _apply_latex_style


def generate_animation(Re=10000, N=257, dt=0.001, warmup_steps=10000,
                       n_frames=50, steps_per_frame=70, fps=14,
                       input_file=None, output_file=None):
    """
    Marches the time-accurate solver and generates a synchronized 2-panel GIF animation:
    Left: Vorticity contours with superimposed streamlines.
    Right: Real-time velocity probe telemetry tracking limit-cycle oscillations.
    """
    if input_file is None:
        input_file = f"data/flow_fields_Re{Re}_N{N}.npz"
    if output_file is None:
        os.makedirs("figures", exist_ok=True)
        output_file = f"figures/lid_driven_vortex_shedding_Re{Re}.gif"

    print("=" * 72)
    print(f"LID-DRIVEN CAVITY: VORTEX SHEDDING ANIMATION GENERATOR (Re = {Re})")
    print("=" * 72)

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Base initial solution not found: {input_file}\n"
            f"Please run `python lid_driven_cavity_fdm.py --Re {Re} --N {N}` first."
        )

    print(f"[INIT] Loading initial solution from: {input_file}", flush=True)
    data = np.load(input_file)

    solver = LidDrivenCavitySolver(
        N=N, Re=Re, lid_velocity=1.0, L=1.0, lid_profile='constant',
        poisson_solver='dst', convection_scheme='central',
        wall_bc='thom', wall_beta=1.0
    )
    solver.psi = data['psi'].copy()
    solver.omega = data['omega'].copy()
    solver.calculate_velocities()

    solver.dt = dt
    solver.alpha_adi = (solver.nu * dt) / (2.0 * solver.h**2)
    solver._init_adi_coefficients()

    # Seed localized perturbation to initiate the unstable shedding mode
    Y, X = solver.Y, solver.X
    perturbation = 0.05 * np.exp(-((X - 0.8)**2 + (Y - 0.8)**2) / (0.05**2))
    solver.omega += perturbation
    solver.apply_boundary_conditions()
    solver.solve_streamfunction()
    solver.calculate_velocities()

    # Warmup phase to reach stable limit cycle
    if warmup_steps > 0:
        print(f"[MARCH] Marching {warmup_steps:,} steps ({warmup_steps * dt:.1f} s) to establish limit cycle...", flush=True)
        for s in range(warmup_steps):
            solver.omega = solver.solve_vorticity_transport_ADI()
            solver.apply_boundary_conditions()
            solver.solve_streamfunction()
            solver.calculate_velocities()
            if (s + 1) % 2500 == 0:
                print(f"  Warmup step {s+1:,}/{warmup_steps:,} complete...", flush=True)

    # Frame capture phase
    print(f"[CAPTURE] Recording {n_frames} frames across full shedding cycles...", flush=True)
    probe_idx = (int(round(0.15 * (N - 1))), int(round(0.08 * (N - 1))))  # BL eddy probe
    t_curr = warmup_steps * dt

    frames_data = []
    t_tracker, u_tracker, v_tracker = [], [], []

    for f in range(n_frames):
        for _ in range(steps_per_frame):
            solver.omega = solver.solve_vorticity_transport_ADI()
            solver.apply_boundary_conditions()
            solver.solve_streamfunction()
            solver.calculate_velocities()
            t_curr += dt

        t_tracker.append(t_curr)
        u_tracker.append(float(solver.u[probe_idx]))
        v_tracker.append(float(solver.v[probe_idx]))

        frames_data.append({
            't': t_curr,
            'omega': solver.omega.copy(),
            'psi': solver.psi.copy(),
            'frame_idx': f
        })
        if (f + 1) % 10 == 0 or f == n_frames - 1:
            print(f"  Captured frame {f+1}/{n_frames} (t = {t_curr:.3f} s)...", flush=True)

    # Rendering frames
    print("[RENDER] Rendering high-contrast LaTeX frames...", flush=True)
    _apply_latex_style()

    temp_dir = os.path.join("figures", "temp_animation_frames")
    os.makedirs(temp_dir, exist_ok=True)
    frame_paths = []

    fig, (ax_flow, ax_trace) = plt.subplots(1, 2, figsize=(13, 5.5))
    levels_omega = np.linspace(-15, 15, 61)
    levels_psi_neg = np.linspace(-0.13, 0.0, 18)
    levels_psi_pos = np.linspace(1e-4, 3.5e-3, 9)

    t_arr = np.array(t_tracker)
    u_arr = np.array(u_tracker)
    v_arr = np.array(v_tracker)

    ax_trace.plot(t_arr, u_arr, color='#1f77b4', linewidth=1.8, label=r'BL Eddy $u(t)$')
    ax_trace.plot(t_arr, v_arr, color='#d62728', linewidth=1.8, linestyle='--', label=r'BL Eddy $v(t)$')
    dot_u, = ax_trace.plot([], [], 'o', color='#1f77b4', markersize=8)
    dot_v, = ax_trace.plot([], [], 's', color='#d62728', markersize=8)

    ax_trace.set_xlabel(r'Physical Convective Time $t \cdot (U/L)$', fontsize=11)
    ax_trace.set_ylabel(r'Probe Velocity Component', fontsize=11)
    ax_trace.set_title(r'\textbf{Real-Time Probe Telemetry}', fontsize=12, pad=8)
    ax_trace.grid(True, linestyle='--', alpha=0.5)
    ax_trace.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    ncol=2, frameon=True, fancybox=True, edgecolor='#cccccc', fontsize=9.5)

    for idx, fd in enumerate(frames_data):
        ax_flow.clear()

        # Vorticity contour fill
        ax_flow.contourf(X, Y, fd['omega'], levels=levels_omega, cmap='coolwarm', extend='both', alpha=0.85)
        # Streamlines
        ax_flow.contour(X, Y, fd['psi'], levels=levels_psi_neg, colors='black', linewidths=0.6, alpha=0.7)
        ax_flow.contour(X, Y, fd['psi'], levels=levels_psi_pos, colors='red', linewidths=0.8, alpha=0.8)

        ax_flow.set_title(rf'\textbf{{Vorticity \& Streamlines}} ($t = {fd["t"]:.2f}\,$s)', fontsize=12, pad=8)
        ax_flow.set_xlabel(r'$x/L$', fontsize=11)
        ax_flow.set_ylabel(r'$y/L$', fontsize=11)
        ax_flow.set_aspect('equal')
        ax_flow.grid(True, linestyle=':', alpha=0.4)

        # Update trace markers
        dot_u.set_data([fd['t']], [u_tracker[idx]])
        dot_v.set_data([fd['t']], [v_tracker[idx]])

        fig.suptitle(rf'\textbf{{Real-Time Vortex Shedding Animation}} --- $Re = {Re}$ ($N = {N} \times {N}$)',
                     fontsize=13, y=0.98)

        plt.subplots_adjust(bottom=0.20, top=0.88, wspace=0.28)
        frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        frame_paths.append(frame_path)

    plt.close()

    # Compile into GIF
    print(f"[ASSEMBLE] Compiling {len(frame_paths)} frames into animated GIF @ {fps} fps...", flush=True)
    images = [Image.open(p) for p in frame_paths]
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True
    )

    # Clean up temporary frames
    for p in frame_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[SUCCESS] Compiled GIF saved to: {output_file} ({size_mb:.2f} MB)")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate animated GIF of unsteady vortex shedding in lid-driven cavity"
    )
    parser.add_argument("--Re", type=int, default=10000, help="Reynolds number (default: 10000)")
    parser.add_argument("--N", type=int, default=257, help="Grid points per axis (default: 257)")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step size (default: 0.001)")
    parser.add_argument("--warmup", type=int, default=10000, help="Warmup time steps (default: 10000)")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to capture (default: 50)")
    parser.add_argument("--steps_per_frame", type=int, default=70, help="Time steps per frame (default: 70)")
    parser.add_argument("--fps", type=int, default=14, help="Frames per second for output GIF (default: 14)")
    parser.add_argument("--input", type=str, default=None, help="Path to initial solution .npz")
    parser.add_argument("--output", type=str, default=None, help="Output .gif file path")

    args = parser.parse_args()
    generate_animation(
        Re=args.Re, N=args.N, dt=args.dt, warmup_steps=args.warmup,
        n_frames=args.frames, steps_per_frame=args.steps_per_frame, fps=args.fps,
        input_file=args.input, output_file=args.output
    )
