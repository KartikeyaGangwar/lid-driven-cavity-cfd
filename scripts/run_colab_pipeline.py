"""
Colab Master Execution Pipeline
Automated batch execution for extended unsteady limit-cycle simulations on Google Colab.

Runs:
1. Re = 100,000 (+5.0s -> t = 8.0s, ~4.1 shedding periods)
2. Re = 50,000  (+5.0s -> t = 10.0s, ~3.3 shedding periods)
3. Re = 25,000  (+6.0s -> t = 12.0s, ~3.4 shedding periods)

Generates:
- High-resolution timeseries and phase portraits
- Closed multi-cycle limit-cycle attractors
- Dynamic 4-phase cycle snapshots
- 48-frame synchronized animated GIFs
- Automatically zips all results to 'colab_unsteady_results.zip' for 1-click download

Usage in Google Colab:
    !python scripts/run_colab_pipeline.py
"""

import os
import sys
import time
import zipfile
import subprocess

PIPELINE_TASKS = [
    {
        'Re': 100000,
        'N': 513,
        'dt': 2.0e-4,
        'add_time': 5.0,
        'convection': 'hybrid',
        'wall_beta': 0.75,
        'sample': 10,
    },
    {
        'Re': 50000,
        'N': 513,
        'dt': 2.0e-4,
        'add_time': 5.0,
        'convection': 'central',
        'wall_beta': 0.85,
        'sample': 10,
    },
    {
        'Re': 25000,
        'N': 513,
        'dt': 3.0e-4,
        'add_time': 6.0,
        'convection': 'central',
        'wall_beta': 1.0,
        'sample': 10,
    },
]

def main():
    print("=" * 80)
    print("GOOGLE COLAB HIGH-RESOLUTION EXTENDED UNSTEADY PIPELINE")
    print(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")
    print("=" * 80)

    start_total = time.time()
    generated_files = []

    for idx, task in enumerate(PIPELINE_TASKS, 1):
        re_val = task['Re']
        n_val = task['N']
        add_time = task['add_time']
        dt_val = task['dt']
        conv = task['convection']
        beta = task['wall_beta']
        sample = task['sample']

        print(f"\n[{idx}/{len(PIPELINE_TASKS)}] STARTING EXTENDED RUN: Re = {re_val} | N = {n_val} | +{add_time:.1f} s")
        cmd = [
            sys.executable, "unsteady_cavity_solver.py",
            "--Re", str(re_val),
            "--N", str(n_val),
            "--dt", str(dt_val),
            "--convection", conv,
            "--wall_beta", str(beta),
            "--sample_interval", str(sample),
            "--resume",
            "--additional_time", str(add_time),
            "--gif",
            "--gif_frames", "48",
            "--fps", "12"
        ]
        
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - t0
            print(f"[{idx}/{len(PIPELINE_TASKS)}] SUCCESS: Re = {re_val} completed in {elapsed/60.0:.2f} minutes.")
        except Exception as e:
            print(f"[{idx}/{len(PIPELINE_TASKS)}] ERROR in Re = {re_val}: {e}")

        # Track outputs
        npz = f"data/unsteady_Re{re_val}_N{n_val}.npz"
        ts_fig = f"figures/lid_driven_unsteady_timeseries_Re{re_val}_N{n_val}.png"
        pp_fig = f"figures/lid_driven_unsteady_phase_portrait_Re{re_val}_N{n_val}.png"
        cs_fig = f"figures/lid_driven_unsteady_cycle_snapshots_Re{re_val}_N{n_val}.png"
        gif_fig = f"figures/lid_driven_vortex_shedding_Re{re_val}_N{n_val}.gif"

        for f in [npz, ts_fig, pp_fig, cs_fig, gif_fig]:
            if os.path.exists(f):
                generated_files.append(f)

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED IN {total_time/60.0:.2f} MINUTES!")
    print("=" * 80)

    # Create Zip Archive for easy Colab download
    zip_name = "colab_unsteady_results.zip"
    print(f"\n[PACKAGING] Compressing {len(generated_files)} result files into '{zip_name}'...")
    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in set(generated_files):
            zf.write(f)
            print(f"  Added: {f} ({os.path.getsize(f) / 1024:.1f} KB)")

    print(f"\n[DOWNLOAD READY] '{zip_name}' created successfully! Size: {os.path.getsize(zip_name) / (1024*1024):.2f} MB")
    print("You can download this zip directly from Colab files or via files.download('colab_unsteady_results.zip').")

if __name__ == "__main__":
    main()
