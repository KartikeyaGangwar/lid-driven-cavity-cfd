"""
Script to recompute exact spectral metrics using the updated 3-point parabolic interpolation,
update all 5 unsteady .npz archives, and re-render publication phase portraits and time-series figures.
"""

import os
import sys
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lid_driven_cavity_fdm import _apply_latex_style
from unsteady_cavity_solver import (
    UnsteadyCavitySolver,
    plot_phase_portrait_and_psd,
    plot_unsteady_timeseries
)

def recompute_and_update_all():
    _apply_latex_style()

    configs = [
        {'Re': 10000, 'N': 513, 'dt': 5.0e-4, 't_transient': 10.0},
        {'Re': 15000, 'N': 513, 'dt': 4.0e-4, 't_transient': 2.0},
        {'Re': 25000, 'N': 513, 'dt': 3.0e-4, 't_transient': 4.2},
        {'Re': 50000, 'N': 513, 'dt': 2.0e-4, 't_transient': 1.33},
        {'Re': 100000, 'N': 513, 'dt': 2.0e-4, 't_transient': 2.8},
    ]

    for cfg in configs:
        re_val = cfg['Re']
        n_val = cfg['N']
        dt_val = cfg['dt']
        t_trans = cfg['t_transient']
        npz_path = f'data/unsteady_Re{re_val}_N{n_val}.npz'

        if not os.path.exists(npz_path):
            print(f"[SKIP] {npz_path} does not exist.")
            continue

        print(f"\n=======================================================")
        print(f"PROCESSING Re = {re_val} (Grid {n_val}x{n_val})")
        print(f"=======================================================")
        d = dict(np.load(npz_path))
        t = d['time']
        u_bl = d['u_BL']
        v_bl = d['v_BL']
        u_tr = d.get('u_TR', None)
        v_tr = d.get('v_TR', None)
        ke = d.get('kinetic_energy', None)
        ens = d.get('enstrophy', None)

        # Create solver instance and populate telemetry
        sim = UnsteadyCavitySolver(N=n_val, Re=re_val, dt=dt_val)
        sim.telemetry = {
            'time': t,
            'u_probes': {'BL': u_bl, 'TR': u_tr if u_tr is not None else u_bl},
            'v_probes': {'BL': v_bl, 'TR': v_tr if v_tr is not None else v_bl},
            'kinetic_energy': ke,
            'enstrophy': ens
        }

        # Run updated frequency analysis
        fft_data = sim.analyze_frequency_and_attractor(probe_key='BL', t_start_analysis=t_trans)

        # Update dictionary with newly computed fields
        d['f_dom'] = np.array(fft_data['f_dom'])
        d['f_bin'] = np.array(fft_data['f_bin'])
        d['delta_f'] = np.array(fft_data['delta_f'])
        d['strouhal'] = np.array(fft_data['strouhal'])
        d['strouhal_uncertainty'] = np.array(fft_data['strouhal_uncertainty'])
        d['period'] = np.array(fft_data['period'])
        d['period_time_domain'] = np.array(fft_data['period_time_domain'])
        d['period_td_std'] = np.array(fft_data['period_td_std'])
        d['n_cycles'] = np.array(fft_data['n_cycles'])
        d['freqs'] = fft_data['freqs']
        d['psd'] = fft_data['psd']

        # Save back to .npz
        np.savez_compressed(npz_path, **d)
        print(f"[SAVED] Updated archive: {npz_path}")

        # Re-render Phase Portrait + PSD
        pp_n513 = f'figures/lid_driven_unsteady_phase_portrait_Re{re_val}_N513.png'
        pp_base = f'figures/lid_driven_unsteady_phase_portrait_Re{re_val}.png'
        plot_phase_portrait_and_psd(fft_data, re_val, save_path=pp_n513)
        shutil.copy2(pp_n513, pp_base)
        print(f"[SAVED] Phase portraits: {pp_n513} and {pp_base}")

        # Re-render Timeseries
        if u_tr is not None and v_tr is not None and ke is not None and ens is not None:
            telemetry = {
                'time': t,
                'u_probes': {'BL': u_bl, 'TR': u_tr},
                'v_probes': {'BL': v_bl, 'TR': v_tr},
                'kinetic_energy': ke,
                'enstrophy': ens
            }
            ts_n513 = f'figures/lid_driven_unsteady_timeseries_Re{re_val}_N513.png'
            ts_base = f'figures/lid_driven_unsteady_timeseries_Re{re_val}.png'
            plot_unsteady_timeseries(telemetry, re_val, save_path=ts_n513)
            shutil.copy2(ts_n513, ts_base)
            print(f"[SAVED] Timeseries: {ts_n513} and {ts_base}")

    print("\n[ALL COMPLETE] All 5 unsteady cases recomputed, archives updated, and figures refreshed.")

if __name__ == '__main__':
    recompute_and_update_all()
