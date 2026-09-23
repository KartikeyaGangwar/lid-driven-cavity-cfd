"""
Re-render all phase portraits and FFT power spectra for Re in [15000, 25000, 50000, 100000]
to strictly enforce:
1. Exact LaTeX Computer Modern typography (via _apply_latex_style).
2. Non-dimensional annotation box: Peak: $St = ...$ ($T = ...$), removing all 'Hz'.
3. Legends positioned cleanly below the horizontal axis.
"""

import os
import sys
import shutil
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lid_driven_cavity_fdm import _apply_latex_style
from unsteady_cavity_solver import plot_phase_portrait_and_psd, plot_unsteady_timeseries

def replot_all():
    _apply_latex_style()

    re_configs = [
        {'Re': 15000, 't_transient': 2.0, 'title': 'Modulated Quasi-Periodic Orbit'},
        {'Re': 25000, 't_transient': 4.2, 'title': 'Nonlinear Folded Attractor'},
        {'Re': 50000, 't_transient': 1.33, 'title': 'Multi-Loop Phase Trajectory'},
        {'Re': 100000, 't_transient': 2.8, 'title': 'High-Reynolds Dynamic Trace'},
    ]

    for cfg in re_configs:
        re_val = cfg['Re']
        t_trans = cfg['t_transient']
        npz_path = f'data/unsteady_Re{re_val}_N513.npz'
        if not os.path.exists(npz_path):
            print(f"[SKIP] {npz_path} does not exist.")
            continue

        print(f"[LOAD] Loading Re={re_val} dataset from {npz_path}...")
        d = np.load(npz_path)
        t = d['time']
        u_bl = d['u_BL']
        v_bl = d['v_BL']
        u_tr = d.get('u_TR', None)
        v_tr = d.get('v_TR', None)
        ke = d.get('kinetic_energy', None)
        ens = d.get('enstrophy', None)

        mask = t >= t_trans
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

        print(f"  Re = {re_val}: f_dom = {f_dom:.4f}, St = {strouhal:.4f}, T = {period:.4f}, cycles = {len(t_stat)*dt_sample / period:.2f}")

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

        pp_path_n513 = f'figures/lid_driven_unsteady_phase_portrait_Re{re_val}_N513.png'
        pp_path_base = f'figures/lid_driven_unsteady_phase_portrait_Re{re_val}.png'
        plot_phase_portrait_and_psd(fft_data, re_val, save_path=pp_path_n513)
        shutil.copy2(pp_path_n513, pp_path_base)
        print(f"  -> Updated {pp_path_n513} and {pp_path_base}")

if __name__ == '__main__':
    replot_all()
