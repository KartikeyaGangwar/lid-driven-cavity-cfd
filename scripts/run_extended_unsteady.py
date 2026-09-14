"""
Batch Runner for Extended Unsteady Cavity Simulations
Runs or continues unsteady simulations to capture multi-cycle limit cycles (3 to 5 full periods).

Can be executed directly on Google Colab or local workstation:
    python scripts/run_extended_unsteady.py --re 100000 --add_time 5.0
    python scripts/run_extended_unsteady.py --all
"""

import os
import sys
import argparse
import subprocess

TARGET_RUNS = [
    {
        'Re': 100000,
        'N': 513,
        'dt': 2.0e-4,
        'additional_time': 5.0,     # from t = 3.0s to t = 8.0s (~4.1 periods)
        'convection': 'hybrid',
        'wall_beta': 0.75,
        'sample_interval': 10,
    },
    {
        'Re': 50000,
        'N': 513,
        'dt': 2.0e-4,
        'additional_time': 5.0,     # from t = 5.0s to t = 10.0s (~3.3 periods)
        'convection': 'central',
        'wall_beta': 0.85,
        'sample_interval': 10,
    },
    {
        'Re': 25000,
        'N': 513,
        'dt': 3.0e-4,
        'additional_time': 6.0,     # from t = 6.0s to t = 12.0s (~3.4 periods)
        'convection': 'central',
        'wall_beta': 1.0,
        'sample_interval': 10,
    },
]

def run_single(run_cfg):
    re = run_cfg['Re']
    n = run_cfg['N']
    add_time = run_cfg['additional_time']
    dt = run_cfg['dt']
    conv = run_cfg['convection']
    beta = run_cfg['wall_beta']
    sample = run_cfg['sample_interval']
    
    npz_path = f"data/unsteady_Re{re}_N{n}.npz"
    cmd = [
        sys.executable, "unsteady_cavity_solver.py",
        "--Re", str(re),
        "--N", str(n),
        "--dt", str(dt),
        "--convection", conv,
        "--wall_beta", str(beta),
        "--sample_interval", str(sample),
        "--resume",
        "--additional_time", str(add_time),
        "--gif",
        "--gif_frames", "48",
        "--fps", "12"
    ]
    print(f"\n{'#'*80}")
    print(f"EXECUTING EXTENDED MARCHING: Re = {re} | N = {n} | +{add_time:.1f} s")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'#'*80}\n", flush=True)
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run extended unsteady simulations")
    parser.add_argument("--re", type=int, default=None, choices=[25000, 50000, 100000],
                        help="Specific Re to run (25000, 50000, 100000)")
    parser.add_argument("--add_time", type=float, default=None, help="Override additional physical time (s)")
    parser.add_argument("--all", action="store_true", help="Run all high-Re cases sequentially")
    args = parser.parse_args()

    if args.re is not None:
        cfg = next((c for c in TARGET_RUNS if c['Re'] == args.re), None)
        if cfg is None:
            print(f"Error: Reynolds number {args.re} not in target list.")
            sys.exit(1)
        if args.add_time is not None:
            cfg['additional_time'] = args.add_time
        run_single(cfg)
    elif args.all:
        for cfg in TARGET_RUNS:
            if args.add_time is not None:
                cfg['additional_time'] = args.add_time
            run_single(cfg)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
