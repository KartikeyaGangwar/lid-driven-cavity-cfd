"""
Batch Extended Unsteady Marching Runner for Higher Reynolds Numbers
(Re = 15,000, 25,000, 50,000, 100,000) on 513 x 513 Mesh.

Executes Option B:
- Advances physical time marching from existing checkpoints
- Eliminates Fourier window-length artifacts
- Evaluates stationary window FFT power spectra
- Extracts 4-phase field cycle snapshots
- Generates publication figures (Timeseries, Phase Portrait + PSD, Snapshots)
- Generates synchronized animated GIFs
- Backs up existing datasets before writing
"""

import os
import sys
import time
import shutil
import argparse
import numpy as np

# Ensure root directory is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from lid_driven_cavity_fdm import _apply_latex_style
from unsteady_cavity_solver import (
    UnsteadyCavitySolver,
    plot_unsteady_timeseries,
    plot_phase_portrait_and_psd,
    plot_cycle_snapshots
)

CASES = [
    {
        'Re': 15000,
        'N': 513,
        'dt': 4.0e-4,
        't_target': 20.0,
        't_transient': 2.0,
        'convection_scheme': 'central',
        'wall_beta': 1.0,
        'sample_interval': 10,
        'log_interval': 2500,
    },
    {
        'Re': 25000,
        'N': 513,
        'dt': 3.0e-4,
        't_target': 22.0,
        't_transient': 4.2,
        'convection_scheme': 'central',
        'wall_beta': 1.0,
        'sample_interval': 10,
        'log_interval': 2500,
    },
    {
        'Re': 50000,
        'N': 513,
        'dt': 2.0e-4,
        't_target': 18.0,
        't_transient': 1.33,
        'convection_scheme': 'central',
        'wall_beta': 0.85,
        'sample_interval': 10,
        'log_interval': 2500,
    },
    {
        'Re': 100000,
        'N': 513,
        'dt': 2.0e-4,
        't_target': 16.0,
        't_transient': 2.8,
        'convection_scheme': 'hybrid',
        'wall_beta': 0.75,
        'sample_interval': 10,
        'log_interval': 2500,
    },
]


def run_single_case(cfg):
    _apply_latex_style()
    re_val = cfg['Re']
    n_val = cfg['N']
    dt_val = cfg['dt']
    t_target = cfg['t_target']
    t_transient = cfg['t_transient']
    conv_scheme = cfg['convection_scheme']
    wall_beta = cfg['wall_beta']
    sample_int = cfg['sample_interval']
    log_int = cfg['log_interval']

    npz_path = os.path.join(ROOT_DIR, 'data', f'unsteady_Re{re_val}_N{n_val}.npz')
    backup_dir = os.path.join(ROOT_DIR, 'data', 'backups')
    os.makedirs(backup_dir, exist_ok=True)

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Source dataset not found: {npz_path}")

    # Backup prior dataset
    backup_path = os.path.join(backup_dir, f'unsteady_Re{re_val}_N{n_val}_prior.npz')
    shutil.copy2(npz_path, backup_path)
    print(f"\n[BACKUP] Copied current state to {backup_path}", flush=True)

    print(f"\n{'='*80}")
    print(f"EXTENDED TIME MARCHING: Re = {re_val} | Grid = {n_val}x{n_val} | Scheme = {conv_scheme} (beta = {wall_beta})")
    print(f"Target Physical Time: t = {t_target:.2f} s | dt = {dt_val:.4e} s")
    print(f"{'='*80}\n", flush=True)

    # Initialize solver
    solver = UnsteadyCavitySolver(
        N=n_val, Re=re_val, dt=dt_val,
        convection_scheme=conv_scheme, wall_bc='thom', wall_beta=wall_beta
    )

    # Resume flow state and past telemetry
    t_start, initial_telemetry = solver.resume_from_unsteady_npz(npz_path)
    print(f"[RESUME] Resuming from t = {t_start:.4f} s to t_target = {t_target:.4f} s...", flush=True)

    if t_target <= t_start:
        print(f"[INFO] Target time {t_target:.2f} s already achieved (t_start = {t_start:.2f} s). Skipping march.")
    else:
        # March forward in physical time
        telemetry = solver.run_simulation(
            t_end=t_target,
            sample_interval=sample_int,
            log_interval=log_int,
            t_start=t_start,
            initial_telemetry=initial_telemetry,
            checkpoint_path=npz_path,
            checkpoint_interval=2500
        )

    for k in solver.telemetry['u_probes']:
        solver.telemetry['u_probes'][k] = np.asarray(solver.telemetry['u_probes'][k])
        solver.telemetry['v_probes'][k] = np.asarray(solver.telemetry['v_probes'][k])
    solver.telemetry['time'] = np.asarray(solver.telemetry['time'])
    solver.telemetry['kinetic_energy'] = np.asarray(solver.telemetry['kinetic_energy'])
    solver.telemetry['enstrophy'] = np.asarray(solver.telemetry['enstrophy'])

    # Verify numerical stability
    max_u = np.max(np.abs(solver.solver.u))
    max_v = np.max(np.abs(solver.solver.v))
    print(f"\n[STABILITY CHECK] Max velocity magnitudes: |u|_max = {max_u:.4f}, |v|_max = {max_v:.4f}")
    if max_u > 5.0 or max_v > 5.0 or np.isnan(max_u) or np.isnan(max_v):
        raise RuntimeError(f"Numerical divergence detected at Re = {re_val}: |u|_max = {max_u}")

    # Perform FFT spectral analysis on stationary interval
    print(f"\n[SPECTRAL] Analyzing stationary interval t >= {t_transient:.2f} s...")
    fft_data = solver.analyze_frequency_and_attractor(probe_key='BL', t_start_analysis=t_transient)

    f_dom = fft_data['f_dom']
    strouhal = fft_data['strouhal']
    period = fft_data['period']
    stat_len = solver.telemetry['time'][-1] - t_transient
    n_cycles = stat_len / period if period > 0 else 0.0

    print(f"  Dominant Strouhal: St = {strouhal:.4f}")
    print(f"  Fundamental Period: T = {period:.4f} s")
    print(f"  Stationary Duration: Delta_t = {stat_len:.2f} s")
    print(f"  Fully Resolved Cycles Captured: {n_cycles:.2f} cycles")

    # Capture 4-phase field cycle snapshots
    snap_period = min(period, 4.0)
    snapshots = solver.capture_cycle_snapshots(period=snap_period, n_phases=4)

    # Save updated compressed .npz
    print(f"\n[SAVE] Serializing full extended dataset to {npz_path}...")
    np.savez_compressed(
        npz_path,
        time=solver.telemetry['time'],
        u_BL=solver.telemetry['u_probes']['BL'],
        v_BL=solver.telemetry['v_probes']['BL'],
        u_TR=solver.telemetry['u_probes']['TR'],
        v_TR=solver.telemetry['v_probes']['TR'],
        kinetic_energy=solver.telemetry['kinetic_energy'],
        enstrophy=solver.telemetry['enstrophy'],
        freqs=fft_data['freqs'],
        psd=fft_data['psd'],
        f_dom=fft_data['f_dom'],
        strouhal=fft_data['strouhal'],
        period=fft_data['period'],
        final_omega=solver.solver.omega,
        final_psi=solver.solver.psi,
        snapshot_0_omega=snapshots[0]['omega'],
        snapshot_0_psi=snapshots[0]['psi'],
        snapshot_90_omega=snapshots[1]['omega'],
        snapshot_90_psi=snapshots[1]['psi'],
        snapshot_180_omega=snapshots[2]['omega'],
        snapshot_180_psi=snapshots[2]['psi'],
        snapshot_270_omega=snapshots[3]['omega'],
        snapshot_270_psi=snapshots[3]['psi'],
        Re=re_val,
        N=n_val
    )
    print(f"[SAVED] Dataset successfully updated: {npz_path}")

    # Generate Publication Figures
    figures_dir = os.path.join(ROOT_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print("\n[VISUALS] Generating publication figures...")
    ts_n513 = os.path.join(figures_dir, f'lid_driven_unsteady_timeseries_Re{re_val}_N513.png')
    ts_base = os.path.join(figures_dir, f'lid_driven_unsteady_timeseries_Re{re_val}.png')
    plot_unsteady_timeseries(solver.telemetry, re_val, ts_n513)
    shutil.copy2(ts_n513, ts_base)
    print(f"  -> Saved {ts_n513} and {ts_base}")

    pp_n513 = os.path.join(figures_dir, f'lid_driven_unsteady_phase_portrait_Re{re_val}_N513.png')
    pp_base = os.path.join(figures_dir, f'lid_driven_unsteady_phase_portrait_Re{re_val}.png')
    plot_phase_portrait_and_psd(fft_data, re_val, pp_n513)
    shutil.copy2(pp_n513, pp_base)
    print(f"  -> Saved {pp_n513} and {pp_base}")

    sn_n513 = os.path.join(figures_dir, f'lid_driven_unsteady_cycle_snapshots_Re{re_val}_N513.png')
    sn_base = os.path.join(figures_dir, f'lid_driven_unsteady_cycle_snapshots_Re{re_val}.png')
    plot_cycle_snapshots(snapshots, solver.solver, re_val, sn_n513)
    shutil.copy2(sn_n513, sn_base)
    print(f"  -> Saved {sn_n513} and {sn_base}")

    # Generate synchronized animated GIF
    print("\n[GIF] Generating synchronized vortex shedding animation...")
    gif_n513 = os.path.join(figures_dir, f'lid_driven_vortex_shedding_Re{re_val}_N513.gif')
    gif_base = os.path.join(figures_dir, f'lid_driven_vortex_shedding_Re{re_val}.gif')
    gif_period = min(period, 3.5)
    solver.generate_shedding_gif(period=gif_period, n_frames=36, fps=12, out_path=gif_n513)
    shutil.copy2(gif_n513, gif_base)
    print(f"  -> Saved {gif_n513} and {gif_base}")

    summary = {
        'Re': re_val,
        'N': n_val,
        'dt': dt_val,
        't_start': t_start,
        't_end': t_target,
        'St': strouhal,
        'T': period,
        'cycles': n_cycles
    }
    print(f"\n[DONE] Case Re = {re_val} finished successfully!\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Batch Extended Unsteady Marching Runner')
    parser.add_argument('--re', type=int, default=None, help='Specific Re to run (default: run all 15k, 25k, 50k, 100k)')
    parser.add_argument('--from-re', type=int, default=None, help='Start running from this Re onward')
    args = parser.parse_args()

    t_suite_start = time.time()
    if args.re is not None:
        cases_to_run = [c for c in CASES if c['Re'] == args.re]
    elif getattr(args, 'from_re', None) is not None:
        cases_to_run = [c for c in CASES if c['Re'] >= args.from_re]
    else:
        cases_to_run = list(CASES)

    print(f"\n{'#'*80}")
    print(f"LAUNCHING EXTENDED UNSTEADY SUITE ({len(cases_to_run)} cases queued)")
    print(f"{'#'*80}\n")

    results = []
    for idx, cfg in enumerate(cases_to_run):
        print(f"\n>>> Running Case {idx+1}/{len(cases_to_run)}: Re = {cfg['Re']} ...")
        t_case_start = time.time()
        res = run_single_case(cfg)
        elapsed = time.time() - t_case_start
        res['elapsed_min'] = elapsed / 60.0
        results.append(res)
        print(f">>> Case Re = {cfg['Re']} completed in {res['elapsed_min']:.2f} minutes.")

    total_elapsed_min = (time.time() - t_suite_start) / 60.0

    print(f"\n{'='*80}")
    print(f"SUITE COMPLETED IN {total_elapsed_min:.2f} MINUTES")
    print(f"{'='*80}")
    print(f"{'Re':>8} | {'Grid':>9} | {'t_start':>7} -> {'t_end':>7} | {'St':>8} | {'Period T':>9} | {'Cycles':>8} | {'Time (min)':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['Re']:8d} | {r['N']:4d}x{r['N']:4d} | {r['t_start']:7.2f} -> {r['t_end']:7.2f} | {r['St']:8.4f} | {r['T']:9.4f} | {r['cycles']:8.2f} | {r['elapsed_min']:10.2f}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
