"""
Automated Ingestion and Manuscript Update Script
Extracts 'colab_unsteady_results.zip', updates telemetry and figures,
and recompiles the 17-page manuscript.

Usage:
    python scripts/ingest_colab_results.py [optional_path_to_zip]
"""

import os
import sys
import glob
import zipfile
import subprocess
import numpy as np

def find_zip(custom_path=None):
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    candidates = [
        "colab_unsteady_results.zip",
        os.path.expanduser("~/Downloads/colab_unsteady_results.zip"),
        os.path.expanduser("~/Downloads/colab_unsteady_results (1).zip"),
        os.path.join(os.environ.get("USERPROFILE", ""), "Downloads", "colab_unsteady_results.zip"),
        os.path.join(os.environ.get("USERPROFILE", ""), "Downloads", "colab_unsteady_results (1).zip"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def main():
    custom_zip = sys.argv[1] if len(sys.argv) > 1 else None
    zip_path = find_zip(custom_zip)

    if not zip_path:
        print("[WAIT] 'colab_unsteady_results.zip' not found in project root or Downloads folder.")
        print("Please place the downloaded zip file in this directory and re-run:")
        print("    python scripts/ingest_colab_results.py")
        sys.exit(1)

    print(f"[EXTRACTING] Found archive: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(".")
        print(f"Successfully extracted {len(zf.namelist())} files.")

    # Inspect updated unsteady datasets
    print("\n" + "=" * 75)
    print("EXTENDED UNSTEADY RUNS INSPECTION SUMMARY")
    print("=" * 75)
    
    for re_val in [25000, 50000, 100000]:
        npz_file = f"data/unsteady_Re{re_val}_N513.npz"
        if os.path.exists(npz_file):
            d = np.load(npz_file)
            t_end = float(d['time'][-1])
            period = float(d['period'])
            f_dom = float(d['f_dom'])
            st = float(d['strouhal'])
            n_cycles = t_end / period if period > 0 else 0
            print(f"Re = {re_val:6d} | t_end = {t_end:5.2f} s | T = {period:5.4f} s | f_0 = {f_dom:6.4f} Hz | St = {st:6.4f} | Cycles = {n_cycles:4.1f}")

    print("\n" + "=" * 75)
    print("RECOMPILING MANUSCRIPT PDF WITH UPDATED DATA...")
    print("=" * 75)
    
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"], cwd="paper", check=True)
        print("\n[SUCCESS] 'paper/manuscript.pdf' successfully recompiled with latest results!")
    except Exception as e:
        print(f"[WARN] LaTeX compilation notice: {e}")

if __name__ == "__main__":
    main()
