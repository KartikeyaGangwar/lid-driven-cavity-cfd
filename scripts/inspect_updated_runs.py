import numpy as np

for re_val in [25000, 50000, 100000]:
    f = f'data/unsteady_Re{re_val}_N513.npz'
    d = np.load(f)
    t = d['time']
    f_dom = float(d['f_dom'])
    st = float(d['strouhal'])
    period = float(d['period'])
    t_span = float(t[-1] - t[0])
    dt = float(t[1] - t[0])
    n_cycles = t_span / period if period > 0 else 0
    u_bl = d['u_BL']
    v_bl = d['v_BL']
    print(f"=== Re = {re_val:,} (N = {int(d['N'])}) ===")
    print(f"  Time Window: t_start = {t[0]:.3f}s -> t_end = {t[-1]:.3f}s (span = {t_span:.3f}s, samples = {len(t)}, dt = {dt:.5f}s)")
    print(f"  Spectral: f_0 = {f_dom:.4f} Hz | St = {st:.4f} | Period T = {period:.4f} s")
    print(f"  Cycle Resolution: {n_cycles:.2f} complete shedding periods captured!")
    print(f"  BL Eddy u: [{u_bl.min():+.4f}, {u_bl.max():+.4f}] | mean = {u_bl.mean():+.4f} | std = {u_bl.std():.4f}")
    print(f"  BL Eddy v: [{v_bl.min():+.4f}, {v_bl.max():+.4f}] | mean = {v_bl.mean():+.4f} | std = {v_bl.std():.4f}")
    print(f"  Final state saved: {'final_omega' in d and 'final_psi' in d}")
    print()
