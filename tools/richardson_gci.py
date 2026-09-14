"""
Richardson Extrapolation & ASME PTC 19.1 / Roache Grid Convergence Index (GCI)
Rigorous numerical verification tool for 2D Lid-Driven Cavity CFD solutions.

Computes:
- Observed order of convergence (p)
- Richardson extrapolated continuum values (h -> 0)
- Approximate relative error (e_a) and extrapolated relative error (e_ext)
- Grid Convergence Index (GCI_fine and GCI_medium) with ASME safety factor Fs = 1.25
- Asymptotic range of convergence check: GCI_coarse / (r^p * GCI_fine) ~ 1.00
- 17-point Ghia centerline velocity benchmark tables

Author: Kartikey Singh
Year: 2026
License: MIT
"""

import os
import glob
import numpy as np
import pandas as pd

# Standard 17 Ghia et al. (1982) measurement locations
GHIA_Y = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
    0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000
])

GHIA_X = np.array([
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
    0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000
])


def compute_gci(f1, f2, f3, r=2.0, Fs=1.25):
    """
    Compute ASME GCI for a 3-grid triplet (f1=fine, f2=medium, f3=coarse).
    Refinement ratio r = h2/h1 = h3/h2 (default r = 2.0).
    Safety factor Fs = 1.25 for triplets.
    """
    eps21 = f2 - f1
    eps32 = f3 - f2

    # Check for monotonic vs oscillatory convergence
    if eps32 == 0 or eps21 == 0:
        return {
            'p': 2.0,
            'f_ext': f1,
            'e_a': 0.0,
            'e_ext': 0.0,
            'gci_fine': 0.0,
            'gci_medium': 0.0,
            'asymptotic_ratio': 1.0,
            'monotonic': True
        }

    ratio = eps32 / eps21
    if ratio <= 0:
        # Non-monotonic or oscillatory convergence
        p = 2.0 # fallback theoretical 2nd-order
        monotonic = False
    else:
        p = np.log(np.abs(ratio)) / np.log(r)
        monotonic = True

    # Bound observed order for stability
    p_eff = max(0.5, min(4.0, p))

    # Richardson extrapolation (h -> 0)
    denom = (r**p_eff) - 1.0
    f_ext = f1 + (f1 - f2) / denom

    # Relative errors
    denom_f1 = f1 if abs(f1) > 1e-12 else 1e-12
    denom_f2 = f2 if abs(f2) > 1e-12 else 1e-12
    denom_ext = f_ext if abs(f_ext) > 1e-12 else 1e-12

    e_a_21 = abs((f1 - f2) / denom_f1)
    e_a_32 = abs((f2 - f3) / denom_f2)
    e_ext_21 = abs((f_ext - f1) / denom_ext)

    # GCI
    gci_fine = (Fs * e_a_21) / denom
    gci_med = (Fs * e_a_32) / denom

    # Asymptotic ratio check
    asymp_ratio = gci_med / ((r**p_eff) * gci_fine) if gci_fine > 0 else 1.0

    return {
        'p': p,
        'p_eff': p_eff,
        'f_ext': f_ext,
        'e_a': e_a_21 * 100.0,        # percentage
        'e_ext': e_ext_21 * 100.0,    # percentage
        'gci_fine': gci_fine * 100.0, # percentage
        'gci_med': gci_med * 100.0,   # percentage
        'asymptotic_ratio': asymp_ratio,
        'monotonic': monotonic
    }


def analyze_triplet(re_val, n_triplet=(513, 257, 129)):
    """
    Perform full GCI analysis for a given Reynolds number across (fine, medium, coarse).
    """
    n1, n2, n3 = n_triplet
    f1_path = f"data/flow_fields_Re{re_val}_N{n1}.npz"
    f2_path = f"data/flow_fields_Re{re_val}_N{n2}.npz"
    f3_path = f"data/flow_fields_Re{re_val}_N{n3}.npz"

    for p in [f1_path, f2_path, f3_path]:
        if not os.path.exists(p):
            return None

    d1 = np.load(f1_path)
    d2 = np.load(f2_path)
    d3 = np.load(f3_path)

    # Minimum streamfunction
    psi_min1 = float(np.min(d1['psi']))
    psi_min2 = float(np.min(d2['psi']))
    psi_min3 = float(np.min(d3['psi']))

    # Primary vortex center
    idx1 = np.unravel_index(np.argmin(d1['psi']), d1['psi'].shape)
    idx2 = np.unravel_index(np.argmin(d2['psi']), d2['psi'].shape)
    idx3 = np.unravel_index(np.argmin(d3['psi']), d3['psi'].shape)

    xc1, yc1 = float(d1['x'][idx1[1]]), float(d1['y'][idx1[0]])
    xc2, yc2 = float(d2['x'][idx2[1]]), float(d2['y'][idx2[0]])
    xc3, yc3 = float(d3['x'][idx3[1]]), float(d3['y'][idx3[0]])

    # Centerline velocity u(0.5, y=0.5)
    u_mid1 = float(d1['u'][d1['N']//2, d1['N']//2]) if 'N' in d1 else float(d1['u'][len(d1['y'])//2, len(d1['x'])//2])
    u_mid2 = float(d2['u'][d2['N']//2, d2['N']//2]) if 'N' in d2 else float(d2['u'][len(d2['y'])//2, len(d2['x'])//2])
    u_mid3 = float(d3['u'][d3['N']//2, d3['N']//2]) if 'N' in d3 else float(d3['u'][len(d3['y'])//2, len(d3['x'])//2])

    res_psi = compute_gci(psi_min1, psi_min2, psi_min3)
    res_xc  = compute_gci(xc1, xc2, xc3)
    res_yc  = compute_gci(yc1, yc2, yc3)
    res_u   = compute_gci(u_mid1, u_mid2, u_mid3)

    return {
        'Re': re_val,
        'grids': (n1, n2, n3),
        'psi_min': {'f1': psi_min1, 'f2': psi_min2, 'f3': psi_min3, 'gci': res_psi},
        'xc': {'f1': xc1, 'f2': xc2, 'f3': xc3, 'gci': res_xc},
        'yc': {'f1': yc1, 'f2': yc2, 'f3': yc3, 'gci': res_yc},
        'u_mid': {'f1': u_mid1, 'f2': u_mid2, 'f3': u_mid3, 'gci': res_u},
    }


def extract_ghia_centerlines(re_list=[1000, 5000, 10000, 25000, 50000, 100000]):
    """
    Extract exact u(y) at x=0.5 and v(x) at y=0.5 along the 17 standard Ghia stations.
    """
    u_records = {'y': GHIA_Y}
    v_records = {'x': GHIA_X}

    for re_val in re_list:
        # Find finest grid available
        cand = [
            f"data/flow_fields_Re{re_val}_N1025.npz",
            f"data/flow_fields_Re{re_val}_N513.npz",
            f"data/flow_fields_Re{re_val}_N257.npz",
            f"data/flow_fields_Re{re_val}_N129.npz",
        ]
        chosen = next((c for c in cand if os.path.exists(c)), None)
        if chosen is None:
            continue

        d = np.load(chosen)
        x, y = d['x'], d['y']
        u, v = d['u'], d['v']

        # Interpolate u at x = 0.5 across GHIA_Y
        x_mid_idx = np.argmin(np.abs(x - 0.5))
        u_mid_profile = u[:, x_mid_idx]
        u_interp = np.interp(GHIA_Y, y, u_mid_profile)

        # Interpolate v at y = 0.5 across GHIA_X
        y_mid_idx = np.argmin(np.abs(y - 0.5))
        v_mid_profile = v[y_mid_idx, :]
        v_interp = np.interp(GHIA_X, x, v_mid_profile)

        u_records[f"Re={re_val}"] = np.round(u_interp, 5)
        v_records[f"Re={re_val}"] = np.round(v_interp, 5)

    df_u = pd.DataFrame(u_records)
    df_v = pd.DataFrame(v_records)
    return df_u, df_v


if __name__ == "__main__":
    print("=" * 80)
    print("ASME PTC 19.1 / ROACHE GRID CONVERGENCE INDEX (GCI) VERIFICATION")
    print("=" * 80)

    # Analyze triplets
    triplets = [
        (10000, (513, 257, 129)),
        (100000, (1025, 513, 257)),
    ]

    for re_val, grids in triplets:
        res = analyze_triplet(re_val, grids)
        if res is None:
            print(f"[SKIP] Grids {grids} for Re={re_val} not all available.")
            continue
        g = res['psi_min']['gci']
        print(f"\nRe = {re_val} | Grids: {grids[0]} -> {grids[1]} -> {grids[2]}")
        print(f"  psi_min: fine={res['psi_min']['f1']:.6f}, med={res['psi_min']['f2']:.6f}, coarse={res['psi_min']['f3']:.6f}")
        print(f"  Observed order p: {g['p']:.2f} (effective p: {g['p_eff']:.2f})")
        print(f"  Extrapolated value (h -> 0): {g['f_ext']:.6f}")
        print(f"  Approx Error e_a: {g['e_a']:.4f}% | Extrapolated Error e_ext: {g['e_ext']:.4f}%")
        print(f"  GCI fine: {g['gci_fine']:.4f}% | GCI medium: {g['gci_med']:.4f}%")
        print(f"  Asymptotic Range Ratio: {g['asymptotic_ratio']:.4f} (~1.00)")

    print("\n" + "=" * 80)
    print("EXTRACTING 17-POINT GHIA CENTERLINE VELOCITY PROFILES")
    print("=" * 80)
    df_u, df_v = extract_ghia_centerlines()
    os.makedirs("data/benchmark_tables", exist_ok=True)
    df_u.to_csv("data/benchmark_tables/ghia_centerline_u_y.csv", index=False)
    df_v.to_csv("data/benchmark_tables/ghia_centerline_v_x.csv", index=False)
    print("\n[SAVED] Saved data/benchmark_tables/ghia_centerline_u_y.csv")
    print(df_u.to_string(index=False))
    print("\n[SAVED] Saved data/benchmark_tables/ghia_centerline_v_x.csv")
    print(df_v.to_string(index=False))
