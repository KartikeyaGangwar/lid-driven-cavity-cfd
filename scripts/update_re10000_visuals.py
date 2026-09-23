"""
Update all Re=10,000 unsteady visuals (Timeseries, Phase Portrait + PSD, 4-Phase Snapshots, Animated GIF)
from the fully converged extended simulation dataset (t = 22.0, N = 513).
Ensures exact LaTeX styling, legends below x-axes, and matching visual aesthetics across all figures.
"""

import os
import sys
import shutil
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lid_driven_cavity_fdm import _apply_latex_style
from unsteady_cavity_solver import (
    UnsteadyCavitySolver,
    plot_unsteady_timeseries,
    plot_phase_portrait_and_psd,
    plot_cycle_snapshots,
)

def update_all_re10000_visuals():
    _apply_latex_style()

    npz_path = 'data/unsteady_Re10000_N513.npz'
    print(f"[LOAD] Loading extended Re=10,000 dataset from {npz_path}...")
    d = np.load(npz_path)

    t = d['time']
    u_bl = d['u_BL']
    v_bl = d['v_BL']
    u_tr = d['u_TR']
    v_tr = d['v_TR']
    ke = d['kinetic_energy']
    ens = d['enstrophy']

    # 1. Telemetry dict for plot_unsteady_timeseries
    telemetry = {
        'time': t,
        'u_probes': {'BL': u_bl, 'TR': u_tr},
        'v_probes': {'BL': v_bl, 'TR': v_tr},
        'kinetic_energy': ke,
        'enstrophy': ens
    }

    # 2. Extract stationary interval (t >= 10.0) for FFT and Phase Portrait
    mask = t >= 10.0
    t_stat = t[mask]
    u_stat = u_bl[mask]
    v_stat = v_bl[mask]

    dt_sample = t_stat[1] - t_stat[0]
    n_samples = len(v_stat)
    window = np.hanning(n_samples)
    v_detrend = v_stat - np.mean(v_stat)
    v_windowed = v_detrend * window

    fft_vals = sfft.rfft(v_windowed)
    freqs = sfft.rfftfreq(n_samples, d=dt_sample)
    psd = (np.abs(fft_vals)**2) / (n_samples * np.sum(window**2))

    valid = freqs >= 0.05
    dom_idx = np.argmax(psd[valid])
    f_dom = float(freqs[valid][dom_idx])
    strouhal = f_dom
    period = 1.0 / f_dom

    print(f"[FFT] Dominant Strouhal St = {strouhal:.4f}, Period T = {period:.4f}")

    fft_data = {
        'freqs': freqs,
        'psd': psd,
        'f_dom': f_dom,
        'strouhal': strouhal,
        'period': period,
        't_stat': t_stat,
        'u_stat': u_stat,
        'v_stat': v_stat
    }

    # 3. Generate Timeseries Figures (both N513 and base name)
    print("\n[PLOT 1] Generating Publication Timeseries...")
    ts_path_n513 = 'figures/lid_driven_unsteady_timeseries_Re10000_N513.png'
    ts_path_base = 'figures/lid_driven_unsteady_timeseries_Re10000.png'
    plot_unsteady_timeseries(telemetry, 10000, save_path=ts_path_n513)
    shutil.copy2(ts_path_n513, ts_path_base)
    print(f"  -> Saved {ts_path_n513} and {ts_path_base}")

    # 4. Generate Phase Portrait + PSD (2-Panel with legends below x-axis)
    print("\n[PLOT 2] Generating 2-Panel Phase Portrait & FFT Power Spectrum...")
    pp_path_n513 = 'figures/lid_driven_unsteady_phase_portrait_Re10000_N513.png'
    pp_path_base = 'figures/lid_driven_unsteady_phase_portrait_Re10000.png'
    plot_phase_portrait_and_psd(fft_data, 10000, save_path=pp_path_n513)
    shutil.copy2(pp_path_n513, pp_path_base)
    print(f"  -> Saved {pp_path_n513} and {pp_path_base}")

    # 5. Initialize solver to capture 4-phase cycle snapshots and generate GIF
    print("\n[SOLVER] Initializing solver at t = 22.0 to capture shedding cycle...")
    unsteady_sim = UnsteadyCavitySolver(
        N=513, Re=10000, dt=5.0e-4,
        convection_scheme='central', wall_bc='thom', wall_beta=1.0
    )
    unsteady_sim.solver.omega = d['final_omega'].copy()
    unsteady_sim.solver.psi = d['final_psi'].copy()
    unsteady_sim.solver.calculate_velocities()

    print(f"[SNAPSHOTS] Capturing 4-phase cycle snapshots over T = {period:.4f}...")
    snapshots = unsteady_sim.capture_cycle_snapshots(period=period, n_phases=4)

    snap_path_n513 = 'figures/lid_driven_unsteady_cycle_snapshots_Re10000_N513.png'
    snap_path_base = 'figures/lid_driven_unsteady_cycle_snapshots_Re10000.png'
    plot_cycle_snapshots(snapshots, unsteady_sim.solver, 10000, save_path=snap_path_n513)
    shutil.copy2(snap_path_n513, snap_path_base)
    print(f"  -> Saved {snap_path_n513} and {snap_path_base}")

    # 6. Generate synchronized animated GIF
    print(f"\n[GIF] Generating 36-frame synchronized animated GIF over T = {period:.4f}...")
    gif_path_n513 = 'figures/lid_driven_vortex_shedding_Re10000_N513.gif'
    gif_path_base = 'figures/lid_driven_vortex_shedding_Re10000.gif'
    unsteady_sim.generate_shedding_gif(period=period, n_frames=36, fps=12, out_path=gif_path_n513)
    shutil.copy2(gif_path_n513, gif_path_base)
    print(f"  -> Saved {gif_path_n513} and {gif_path_base}")

    # 7. Update npz file with true snapshot fields
    data_dict = dict(d)
    data_dict['snapshot_0_omega'] = snapshots[0]['omega']
    data_dict['snapshot_0_psi'] = snapshots[0]['psi']
    data_dict['snapshot_90_omega'] = snapshots[1]['omega']
    data_dict['snapshot_90_psi'] = snapshots[1]['psi']
    data_dict['snapshot_180_omega'] = snapshots[2]['omega']
    data_dict['snapshot_180_psi'] = snapshots[2]['psi']
    data_dict['snapshot_270_omega'] = snapshots[3]['omega']
    data_dict['snapshot_270_psi'] = snapshots[3]['psi']
    data_dict['final_omega'] = unsteady_sim.solver.omega
    data_dict['final_psi'] = unsteady_sim.solver.psi

    np.savez_compressed(npz_path, **data_dict)
    print(f"\n[SUCCESS] Updated {npz_path} with 4-phase snapshot fields.")
    print("[ALL DONE] All Re=10,000 visuals, animated GIF, and datasets synchronized!\n")


if __name__ == '__main__':
    update_all_re10000_visuals()
