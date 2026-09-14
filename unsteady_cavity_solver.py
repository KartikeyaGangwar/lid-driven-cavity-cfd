"""
Time-Accurate Unsteady Lid-Driven Cavity Flow Solver (2D, Incompressible)
Capturing Hopf Bifurcation, Limit-Cycle Attractor, and Vortex Shedding Dynamics.

Features:
- Physical time-accurate ADI marching with exact tridiagonal boundary closures
- Fast 2D Discrete Sine Transform (DST) Poisson solver (exact to machine precision)
- Zero numerical dissipation (pure 2nd-order Central Differencing, nu_num = 0)
- Multi-point probe telemetry (velocities u(t), v(t), kinetic energy, enstrophy)
- Fast Fourier Transform (FFT) Power Spectral Density and Strouhal number (St = f*L/U)
- Phase-space attractor reconstruction (u vs v limit cycle)
- Dynamic 4-phase cycle snapshot extraction
- Publication-grade LaTeX typography (Computer Modern) with externalized legends

Author: Kartikey Singh
Year: 2026
License: MIT
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.fft as sfft
import matplotlib.pyplot as plt

# UNICODE FIX FOR TERMINALS
if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    try:
        current_encoding = getattr(sys.stdout, 'encoding', None)
        if current_encoding and current_encoding.lower() != 'utf-8':
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from lid_driven_cavity_fdm import LidDrivenCavitySolver, _apply_latex_style


class UnsteadyCavitySolver:
    """
    Time-accurate unsteady solver for 2D lid-driven cavity flow at high Reynolds numbers.
    Simulates vortex shedding, Hopf bifurcations, and limit-cycle oscillations.
    """

    def __init__(self, N=257, Re=10000, lid_velocity=1.0, L=1.0, dt=0.001,
                 lid_profile='constant', convection_scheme='central', wall_bc='thom'):
        self.N = N
        self.Re = Re
        self.U = lid_velocity
        self.L = L
        self.dt = dt
        self.lid_profile = lid_profile
        self.convection_scheme = convection_scheme
        self.wall_bc = wall_bc
        self.wall_beta = 0.85 if Re >= 50000 else 1.0

        # Base solver instance with appropriate wall_beta for high-Re stability
        self.solver = LidDrivenCavitySolver(
            N=N, Re=Re, lid_velocity=lid_velocity, L=L,
            lid_profile=lid_profile, poisson_solver='dst',
            convection_scheme=convection_scheme,
            wall_bc=wall_bc, wall_beta=self.wall_beta
        )

        # Set physical time step and update ADI coefficients
        self.solver.dt = dt
        self.solver.alpha_adi = (self.solver.nu * dt) / (2.0 * self.solver.h**2)
        self.solver._init_adi_coefficients()

        # Monitoring probes (phys_x, phys_y, label, key)
        self.probes = [
            {'name': 'BL Secondary Eddy', 'x_phys': 0.08, 'y_phys': 0.15, 'key': 'BL'},
            {'name': 'TR Impingement Shear', 'x_phys': 0.85, 'y_phys': 0.85, 'key': 'TR'},
            {'name': 'TL Secondary Eddy', 'x_phys': 0.08, 'y_phys': 0.90, 'key': 'TL'},
            {'name': 'Core Vortex Region', 'x_phys': 0.50, 'y_phys': 0.50, 'key': 'Core'},
        ]
        for p in self.probes:
            j = int(np.argmin(np.abs(self.solver.x - p['x_phys'])))
            i = int(np.argmin(np.abs(self.solver.y - p['y_phys'])))
            p['idx'] = (i, j)
            p['x_actual'] = float(self.solver.x[j])
            p['y_actual'] = float(self.solver.y[i])

    def initialize_from_npz(self, npz_path, add_seed_perturbation=True):
        """Initialize flow field from saved steady/continuation solution."""
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Initial field file not found: {npz_path}")
        print(f"[INIT] Loading initial solution from: {npz_path}", flush=True)
        data = np.load(npz_path)

        src_N = len(data['x'])
        if src_N == self.N:
            self.solver.psi = data['psi'].copy()
            self.solver.omega = data['omega'].copy()
        elif src_N == 2 * (self.N - 1) + 1:
            print(f"[INIT] Exact injection from {src_N}x{src_N} -> {self.N}x{self.N} [::2, ::2]")
            self.solver.psi = data['psi'][::2, ::2].copy()
            self.solver.omega = data['omega'][::2, ::2].copy()
        else:
            from lid_driven_cavity_fdm import prolong_fields
            print(f"[INIT] Prolonging from {src_N}x{src_N} -> {self.N}x{self.N}")
            self.solver.psi, self.solver.omega = prolong_fields(
                data['psi'], data['omega'], data['x'], data['y'], self.solver.x, self.solver.y
            )

        self.solver.calculate_velocities()

        if add_seed_perturbation:
            # Add tiny localized shear layer perturbation to seed the unstable Hopf mode
            Y, X = self.solver.Y, self.solver.X
            perturbation = 0.05 * np.exp(-((X - 0.8)**2 + (Y - 0.8)**2) / (0.05**2))
            self.solver.omega += perturbation
            self.solver.apply_boundary_conditions()
            self.solver.solve_streamfunction()
            self.solver.calculate_velocities()
            print("[INIT] Applied localized shear perturbation (amplitude = 0.05) to trigger Hopf eigenmode.", flush=True)

    def run_simulation(self, t_end=25.0, sample_interval=10, log_interval=2000):
        """
        Advance in physical time up to t_end.
        Samples probe data, kinetic energy, and enstrophy.
        """
        dt = self.dt
        total_steps = int(round(t_end / dt))
        print(f"\n{'='*75}")
        print(f"STARTING UNSTEADY MARCHING: Re = {self.Re} | Grid = {self.N}x{self.N}")
        print(f"Total Duration: {t_end:.2f} s ({total_steps} physical time steps) | dt = {dt:.4e} s")
        print(f"Sampling Interval: every {sample_interval} steps ({sample_interval*dt:.4e} s)")
        print(f"{'='*75}\n", flush=True)

        telemetry = {
            'time': [],
            'u_probes': {p['key']: [] for p in self.probes},
            'v_probes': {p['key']: [] for p in self.probes},
            'kinetic_energy': [],
            'enstrophy': [],
        }

        h2 = self.solver.h**2
        t_start = time.time()

        for step in range(1, total_steps + 1):
            # 1. ADI step for omega
            self.solver.omega = self.solver.solve_vorticity_transport_ADI()
            # 2. Update wall vorticity
            self.solver.apply_boundary_conditions()
            # 3. Streamfunction Poisson
            self.solver.solve_streamfunction()
            # 4. Velocities
            self.solver.calculate_velocities()

            current_time = step * dt

            # Sampling telemetry
            if step % sample_interval == 0:
                telemetry['time'].append(current_time)
                for p in self.probes:
                    k = p['key']
                    idx = p['idx']
                    telemetry['u_probes'][k].append(float(self.solver.u[idx]))
                    telemetry['v_probes'][k].append(float(self.solver.v[idx]))

                ke = 0.5 * np.sum(self.solver.u**2 + self.solver.v**2) * h2
                ens = 0.5 * np.sum(self.solver.omega**2) * h2
                telemetry['kinetic_energy'].append(float(ke))
                telemetry['enstrophy'].append(float(ens))

            # Logging
            if step % log_interval == 0 or step == total_steps:
                elapsed = time.time() - t_start
                rate = step / elapsed if elapsed > 0 else 0
                bl_u = self.solver.u[self.probes[0]['idx']]
                bl_v = self.solver.v[self.probes[0]['idx']]
                print(f"Step {step:6d}/{total_steps} | t = {current_time:6.2f}s ({step/total_steps*100:4.1f}%) | "
                      f"BL Eddy (u={bl_u:+8.4f}, v={bl_v:+8.4f}) | Rate: {rate:5.1f} steps/s", flush=True)

        total_elapsed = time.time() - t_start
        print(f"\n[COMPLETE] Simulated {t_end:.2f} s ({total_steps} steps) in {total_elapsed:.2f} s ({total_steps/total_elapsed:.1f} steps/s).\n")

        for k in telemetry['u_probes']:
            telemetry['u_probes'][k] = np.array(telemetry['u_probes'][k])
            telemetry['v_probes'][k] = np.array(telemetry['v_probes'][k])
        telemetry['time'] = np.array(telemetry['time'])
        telemetry['kinetic_energy'] = np.array(telemetry['kinetic_energy'])
        telemetry['enstrophy'] = np.array(telemetry['enstrophy'])

        self.telemetry = telemetry
        return telemetry

    def analyze_frequency_and_attractor(self, probe_key='BL', t_start_analysis=None):
        """
        Analyze the periodic limit cycle using FFT on the statistically stationary interval (t >= t_start_analysis).
        Extracts dominant frequency, Strouhal number St = f*L/U, and phase portrait data.
        """
        t = self.telemetry['time']
        if t_start_analysis is None or t_start_analysis >= t[-1]:
            t_start_analysis = t[0] + 0.35 * (t[-1] - t[0])

        mask = t >= t_start_analysis
        t_stat = t[mask]
        v_signal = self.telemetry['v_probes'][probe_key][mask]
        u_signal = self.telemetry['u_probes'][probe_key][mask]

        dt_sample = t_stat[1] - t_stat[0]
        N_samples = len(v_signal)

        # Apply Hann window to eliminate spectral leakage
        window = np.hanning(N_samples)
        v_detrend = v_signal - np.mean(v_signal)
        v_windowed = v_detrend * window

        # FFT
        fft_vals = sfft.rfft(v_windowed)
        freqs = sfft.rfftfreq(N_samples, d=dt_sample)
        psd = (np.abs(fft_vals)**2) / (N_samples * np.sum(window**2))

        # Ignore DC / very low frequency (< 0.05 Hz)
        valid_idx = freqs >= 0.05
        if np.any(valid_idx):
            f_valid = freqs[valid_idx]
            psd_valid = psd[valid_idx]
            dom_idx = np.argmax(psd_valid)
            f_dom = float(f_valid[dom_idx])
        else:
            f_dom = float(freqs[1]) if len(freqs) > 1 else 1.0

        strouhal = f_dom * self.L / self.U
        period = 1.0 / f_dom if f_dom > 0 else 1.0
        if np.isnan(period) or period <= 0:
            period = 1.0
            strouhal = 1.0

        print(f"[FFT ANALYSIS] Dominant Frequency: f_0 = {f_dom:.4f} Hz")
        print(f"[FFT ANALYSIS] Non-Dimensional Strouhal Number: St = {strouhal:.4f}")
        print(f"[FFT ANALYSIS] Fundamental Oscillation Period: T = {period:.4f} s")

        return {
            'freqs': freqs,
            'psd': psd,
            'f_dom': f_dom,
            'strouhal': strouhal,
            'period': period,
            't_stat': t_stat,
            'u_stat': u_signal,
            'v_stat': v_signal
        }

    def capture_cycle_snapshots(self, period, n_phases=4):
        """
        March for one additional period T to capture n_phases equidistant snapshots of the flow field.
        """
        print(f"\n[SNAPSHOTS] Capturing {n_phases} flow field phases over one period T = {period:.4f} s...")
        dt = self.dt
        steps_per_phase = max(1, int(round((period / n_phases) / dt)))
        snapshots = []

        for phase_idx in range(n_phases):
            phase_deg = phase_idx * (360.0 / n_phases)
            print(f"  Capturing Phase {phase_idx+1}/{n_phases} (Phase Angle: {phase_deg:.0f}°)...", flush=True)
            snapshots.append({
                'phase_deg': phase_deg,
                'time_in_period': phase_idx * (period / n_phases),
                'psi': self.solver.psi.copy(),
                'omega': self.solver.omega.copy(),
                'u': self.solver.u.copy(),
                'v': self.solver.v.copy(),
            })

            # Advance by steps_per_phase
            for _ in range(steps_per_phase):
                self.solver.omega = self.solver.solve_vorticity_transport_ADI()
                self.solver.apply_boundary_conditions()
                self.solver.solve_streamfunction()
                self.solver.calculate_velocities()

        return snapshots

    def generate_shedding_gif(self, period, n_frames=36, fps=12, out_path=None):
        """
        Marches across one complete fundamental cycle of the established attractor
        and generates a synchronized 2-panel publication animated GIF.
        """
        if out_path is None:
            suffix = f"_N{self.N}" if self.N != 257 else ""
            out_path = f"figures/lid_driven_vortex_shedding_Re{self.Re}{suffix}.gif"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        print(f"\n[GIF] Generating synchronized {n_frames}-frame animation over T = {period:.4f} s...")
        dt = self.dt
        total_steps_cycle = int(round(period / dt))
        steps_per_frame = max(1, total_steps_cycle // n_frames)
        actual_frames = total_steps_cycle // steps_per_frame

        probe_idx = (int(round(0.15 * (self.N - 1))), int(round(0.08 * (self.N - 1))))
        frames_data = []
        t_tracker, u_tracker, v_tracker = [], [], []
        t_curr = 0.0

        for f in range(actual_frames):
            for _ in range(steps_per_frame):
                self.solver.omega = self.solver.solve_vorticity_transport_ADI()
                self.solver.apply_boundary_conditions()
                self.solver.solve_streamfunction()
                self.solver.calculate_velocities()
                t_curr += dt

            t_tracker.append(t_curr)
            u_tracker.append(float(self.solver.u[probe_idx]))
            v_tracker.append(float(self.solver.v[probe_idx]))
            frames_data.append({
                't': t_curr,
                'omega': self.solver.omega.copy(),
                'psi': self.solver.psi.copy(),
                'phase_deg': (f / actual_frames) * 360.0
            })

        _apply_latex_style()
        temp_dir = f"figures/temp_frames_Re{self.Re}_N{self.N}"
        os.makedirs(temp_dir, exist_ok=True)
        frame_paths = []

        fig, (ax_flow, ax_trace) = plt.subplots(1, 2, figsize=(13, 5.5))
        levels_omega = np.linspace(-20, 20, 69)
        levels_psi_neg = np.linspace(-0.13, 0.0, 18)
        levels_psi_pos = np.linspace(1e-4, 3.5e-3, 9)

        t_norm = np.array(t_tracker) / period
        u_arr = np.array(u_tracker)
        v_arr = np.array(v_tracker)

        ax_trace.plot(t_norm, u_arr, color='#1f77b4', linewidth=2.0, label=r'BL Eddy $u(t)$')
        ax_trace.plot(t_norm, v_arr, color='#d62728', linewidth=2.0, linestyle='--', label=r'BL Eddy $v(t)$')
        dot_u, = ax_trace.plot([], [], 'o', color='#1f77b4', markersize=9)
        dot_v, = ax_trace.plot([], [], 's', color='#d62728', markersize=9)

        ax_trace.set_xlabel(r'Normalized Cycle Time $t / T$', fontsize=11)
        ax_trace.set_ylabel(r'Velocity Component', fontsize=11)
        ax_trace.set_xlim(0.0, 1.0)
        ax_trace.set_title(rf'\textbf{{Real-Time Probe Telemetry}} ({self.N} $\times$ {self.N})', fontsize=12, pad=8)
        ax_trace.grid(True, linestyle='--', alpha=0.5)
        ax_trace.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                        ncol=2, frameon=True, fancybox=True, edgecolor='#cccccc', fontsize=9.5)

        X, Y = self.solver.X, self.solver.Y

        for idx, fd in enumerate(frames_data):
            ax_flow.clear()
            ax_flow.contourf(X, Y, fd['omega'], levels=levels_omega, cmap='coolwarm', extend='both', alpha=0.85)
            ax_flow.contour(X, Y, fd['psi'], levels=levels_psi_neg, colors='black', linewidths=0.6, alpha=0.7)
            ax_flow.contour(X, Y, fd['psi'], levels=levels_psi_pos, colors='red', linewidths=0.8, alpha=0.8)

            ax_flow.set_title(rf'\textbf{{Vorticity \& Streamlines}} ($\theta = {fd["phase_deg"]:.0f}^\circ$)', fontsize=12, pad=8)
            ax_flow.set_xlabel(r'$x/L$', fontsize=11)
            ax_flow.set_ylabel(r'$y/L$', fontsize=11)
            ax_flow.set_aspect('equal')
            ax_flow.grid(True, linestyle=':', alpha=0.4)

            dot_u.set_data([t_norm[idx]], [u_tracker[idx]])
            dot_v.set_data([t_norm[idx]], [v_tracker[idx]])

            fig.suptitle(rf'\textbf{{Dynamic Vortex Shedding}} --- $Re = {self.Re}$, \textbf{{Ultra-Fine}} ${self.N} \times {self.N}$',
                         fontsize=13, y=0.98)
            plt.subplots_adjust(bottom=0.20, top=0.88, wspace=0.28)
            frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
            plt.savefig(frame_path, dpi=120, bbox_inches='tight')
            frame_paths.append(frame_path)

        plt.close(fig)

        from PIL import Image
        images = [Image.open(p) for p in frame_paths]
        images[0].save(
            out_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True
        )

        for p in frame_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"[SUCCESS] Saved dynamic animated GIF to: {out_path} ({size_mb:.2f} MB)")
        return out_path


def plot_unsteady_timeseries(telemetry, Re, save_path='figures/lid_driven_unsteady_timeseries_Re10000.png'):
    """Plot multi-probe velocity time histories and total kinetic energy/enstrophy."""
    _apply_latex_style()
    t = telemetry['time']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    # Panel 1: Probe velocities
    ax1.plot(t, telemetry['u_probes']['BL'], color='#1f77b4', linewidth=1.5,
             label=r'BL Eddy $u(t)$ ($x=0.08, y=0.15$)')
    ax1.plot(t, telemetry['v_probes']['BL'], color='#aec7e8', linewidth=1.5, linestyle='--',
             label=r'BL Eddy $v(t)$ ($x=0.08, y=0.15$)')
    ax1.plot(t, telemetry['u_probes']['TR'], color='#d62728', linewidth=1.5,
             label=r'TR Shear $u(t)$ ($x=0.85, y=0.85$)')
    ax1.plot(t, telemetry['v_probes']['TR'], color='#ff9896', linewidth=1.5, linestyle='--',
             label=r'TR Shear $v(t)$ ($x=0.85, y=0.85$)')

    ax1.set_ylabel(r'Velocity Component ($u/U, v/U$)', fontsize=12)
    ax1.set_title(rf'\textbf{{Unsteady Vortex Telemetry}} --- $Re = {Re}$ (Hopf Limit Cycle Onset)', fontsize=13, pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Total Kinetic Energy & Enstrophy
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(t, telemetry['kinetic_energy'], color='#2ca02c', linewidth=1.6, label=r'Total Kinetic Energy $E_k(t) = \frac{1}{2}\int (u^2 + v^2) \, dA$')
    l2 = ax2_twin.plot(t, telemetry['enstrophy'], color='#9467bd', linewidth=1.6, linestyle='-.', label=r'Total Enstrophy $\mathcal{E}(t) = \frac{1}{2}\int \omega^2 \, dA$')

    ax2.set_xlabel(r'Physical Convective Time $t \cdot (U/L)$', fontsize=12)
    ax2.set_ylabel(r'Kinetic Energy $E_k$', color='#2ca02c', fontsize=12)
    ax2_twin.set_ylabel(r'Enstrophy $\mathcal{E}$', color='#9467bd', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Combined legends strictly below plots
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, -0.08),
               ncol=2, frameon=True, fancybox=True, edgecolor='#cccccc', fontsize=9.5)

    handles2 = l1 + l2
    labels2 = [h.get_label() for h in handles2]
    ax2.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, -0.22),
               ncol=2, frameon=True, fancybox=True, edgecolor='#cccccc', fontsize=9.5)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18, hspace=0.35)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Time-series figure saved to: {save_path}", flush=True)


def plot_phase_portrait_and_psd(fft_data, Re, save_path='figures/lid_driven_unsteady_phase_portrait_Re10000.png'):
    """Plot phase portrait (u vs v limit cycle) and FFT Power Spectral Density."""
    _apply_latex_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: Phase Space Orbit (u vs v)
    u = fft_data['u_stat']
    v = fft_data['v_stat']
    # Color code by time progression
    n_pts = len(u)
    colors = np.linspace(0, 1, n_pts)
    ax1.plot(u, v, color='#1f77b4', linewidth=1.5, alpha=0.85, label=r'Phase Orbit $(u(t), v(t))$')
    # Mark start and end points
    ax1.scatter(u[0], v[0], color='#2ca02c', s=55, zorder=5, label=r'Cycle Entry ($t=10.0$)')
    ax1.scatter(u[-1], v[-1], color='#d62728', marker='s', s=55, zorder=5, label=r'Cycle Terminal Point')

    ax1.set_xlabel(r'Horizontal Velocity $u/U$', fontsize=12)
    ax1.set_ylabel(r'Vertical Velocity $v/U$', fontsize=12)
    ax1.set_title(rf'\textbf{{Phase-Space Limit-Cycle Attractor}} ($Re={Re}$)', fontsize=13, pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16),
               ncol=3, frameon=True, fancybox=True, edgecolor='#cccccc', fontsize=9.5)

    # Panel 2: FFT Power Spectral Density
    freqs = fft_data['freqs']
    psd = fft_data['psd']
    f_dom = fft_data['f_dom']
    strouhal = fft_data['strouhal']

    # Display up to fundamental + harmonics
    mask = (freqs >= 0.05) & (freqs <= max(2.5, f_dom * 2.5))
    if not np.any(mask):
        mask = np.ones_like(freqs, dtype=bool)

    ax2.plot(freqs[mask], psd[mask], color='#d62728', linewidth=1.8, label=r'Power Spectral Density $|V(f)|^2$')
    ax2.axvline(f_dom, color='#1f77b4', linestyle='--', linewidth=1.4,
                label=rf'Dominant $f_0 = {f_dom:.3f}$ ($St = {strouhal:.3f}$)')

    # Annotate peak
    max_psd = np.max(psd[mask]) if np.any(mask) else (np.max(psd) if len(psd) > 0 else 1.0)
    ax2.annotate(rf'\textbf{{Peak:}} $St = {strouhal:.3f}$ ($f_0 = {f_dom:.3f}$ Hz)',
                 xy=(f_dom, max_psd), xytext=(f_dom + 0.35, max_psd * 0.85),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=10.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', edgecolor='#cccc99'))

    ax2.set_xlabel(r'Non-Dimensional Frequency $f \cdot (L/U)$ / Strouhal $St$', fontsize=12)
    ax2.set_ylabel(r'Power Spectral Density (PSD)', fontsize=12)
    ax2.set_title(rf'\textbf{{FFT Power Spectrum}} (Hopf Fundamental Frequency)', fontsize=13, pad=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16),
               ncol=2, frameon=True, fancybox=True, edgecolor='#cccccc', fontsize=9.5)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Phase portrait & PSD saved to: {save_path}", flush=True)


def plot_cycle_snapshots(snapshots, solver, Re, save_path='figures/lid_driven_unsteady_cycle_snapshots_Re10000.png'):
    """Plot 4 equidistant phase snapshots of vorticity field & streamlines over one complete shedding period."""
    _apply_latex_style()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.0), sharey=True)
    X, Y = solver.X, solver.Y

    for idx, snap in enumerate(snapshots):
        ax = axes[idx]
        deg = snap['phase_deg']
        omega = snap['omega']
        psi = snap['psi']

        # Vorticity contour fill
        levels_omega = np.linspace(-15, 15, 61)
        cf = ax.contourf(X, Y, omega, levels=levels_omega, cmap='coolwarm', extend='both', alpha=0.85)

        # Streamline contours
        # Primary negative contours
        ax.contour(X, Y, psi, levels=np.linspace(-0.13, 0.0, 18), colors='black', linewidths=0.6, alpha=0.7)
        # Secondary positive contours
        ax.contour(X, Y, psi, levels=np.linspace(1e-4, 3.5e-3, 9), colors='red', linewidths=0.8, alpha=0.8)

        ax.set_title(rf'\textbf{{Phase}} $\theta = {deg:.0f}^\circ$', fontsize=12, pad=8)
        ax.set_xlabel(r'$x/L$', fontsize=11)
        if idx == 0:
            ax.set_ylabel(r'$y/L$', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.4)

    fig.suptitle(rf'\textbf{{Dynamic Vortex Shedding Over One Complete Cycle}} --- $Re = {Re}$',
                 fontsize=14, y=0.98)

    # Colorbar below all subplots
    cbar_ax = fig.add_axes([0.25, 0.08, 0.50, 0.035])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'Vorticity Field $\omega \cdot (L/U)$', fontsize=11)

    plt.subplots_adjust(bottom=0.22, top=0.86, wspace=0.15)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Cycle snapshots saved to: {save_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Time-Accurate Unsteady Cavity Flow Solver')
    parser.add_argument('--Re', type=int, default=10000, help='Reynolds number')
    parser.add_argument('--N', type=int, default=257, help='Grid points along each axis')
    parser.add_argument('--dt', type=float, default=None, help='Physical time step (default auto-tuned by Re)')
    parser.add_argument('--t_end', type=float, default=None, help='Total physical time (default auto-tuned by Re)')
    parser.add_argument('--sample_interval', type=int, default=10, help='Telemetry sampling frequency (in steps)')
    parser.add_argument('--init_npz', type=str, default=None, help='Initial solution path')
    parser.add_argument('--gif', action='store_true', default=True, help='Generate publication animated GIF')
    parser.add_argument('--no_gif', dest='gif', action='store_false')
    parser.add_argument('--gif_frames', type=int, default=36, help='Number of frames in animated GIF')
    parser.add_argument('--fps', type=int, default=12, help='Frames per second for output GIF')
    args = parser.parse_args()

    dt = args.dt
    if dt is None:
        dt = 0.0001 if args.Re >= 50000 else 0.001

    t_end = args.t_end
    if t_end is None:
        t_end = 2.0 if args.Re >= 100000 else (5.0 if args.Re >= 50000 else 25.0)

    init_path = args.init_npz
    if init_path is None:
        # Check standard N path first, then fallback to N=1025
        cand1 = f'data/flow_fields_Re{args.Re}_N{args.N}.npz'
        cand2 = f'data/flow_fields_Re{args.Re}_N1025.npz'
        cand3 = f'data/flow_fields_Re{args.Re}_N513.npz'
        if os.path.exists(cand1):
            init_path = cand1
        elif os.path.exists(cand2):
            init_path = cand2
        elif os.path.exists(cand3):
            init_path = cand3
        else:
            init_path = cand1

    unsteady_sim = UnsteadyCavitySolver(
        N=args.N, Re=args.Re, dt=dt,
        convection_scheme='central', wall_bc='thom'
    )

    unsteady_sim.initialize_from_npz(init_path, add_seed_perturbation=True)

    # 1. Run simulation
    telemetry = unsteady_sim.run_simulation(t_end=t_end, sample_interval=args.sample_interval)

    # 2. Analyze Limit Cycle & FFT
    fft_data = unsteady_sim.analyze_frequency_and_attractor(probe_key='BL', t_start_analysis=None)

    # 3. Capture 4-phase cycle snapshots
    snapshots = unsteady_sim.capture_cycle_snapshots(period=fft_data['period'], n_phases=4)

    # 4. Save npz data
    out_npz = f'data/unsteady_Re{args.Re}_N{args.N}.npz'
    np.savez_compressed(
        out_npz,
        time=telemetry['time'],
        u_BL=telemetry['u_probes']['BL'],
        v_BL=telemetry['v_probes']['BL'],
        u_TR=telemetry['u_probes']['TR'],
        v_TR=telemetry['v_probes']['TR'],
        kinetic_energy=telemetry['kinetic_energy'],
        enstrophy=telemetry['enstrophy'],
        freqs=fft_data['freqs'],
        psd=fft_data['psd'],
        f_dom=fft_data['f_dom'],
        strouhal=fft_data['strouhal'],
        period=fft_data['period'],
        snapshot_0_omega=snapshots[0]['omega'],
        snapshot_0_psi=snapshots[0]['psi'],
        snapshot_90_omega=snapshots[1]['omega'],
        snapshot_90_psi=snapshots[1]['psi'],
        snapshot_180_omega=snapshots[2]['omega'],
        snapshot_180_psi=snapshots[2]['psi'],
        snapshot_270_omega=snapshots[3]['omega'],
        snapshot_270_psi=snapshots[3]['psi'],
        Re=args.Re,
        N=args.N
    )
    print(f"[SAVED] Telemetry data and cycle snapshots saved to {out_npz}")

    # 5. Generate Figures
    suffix = f"_N{args.N}" if args.N != 257 else ""
    plot_unsteady_timeseries(telemetry, args.Re, f'figures/lid_driven_unsteady_timeseries_Re{args.Re}{suffix}.png')
    plot_phase_portrait_and_psd(fft_data, args.Re, f'figures/lid_driven_unsteady_phase_portrait_Re{args.Re}{suffix}.png')
    plot_cycle_snapshots(snapshots, unsteady_sim.solver, args.Re, f'figures/lid_driven_unsteady_cycle_snapshots_Re{args.Re}{suffix}.png')

    # 6. Generate Synchronized Publication Animated GIF
    if args.gif:
        unsteady_sim.generate_shedding_gif(
            period=fft_data['period'],
            n_frames=args.gif_frames,
            fps=args.fps
        )

    print("\n[SUCCESS] Time-Accurate Unsteady Simulation, Visuals & Animated GIF Complete!\n")


if __name__ == '__main__':
    main()
