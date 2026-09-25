# Release Notes — v2.0.0: Reference Benchmark Suite & Research Preprint

**Tag:** `v2.0.0`  
**Title:** Reference CFD Solver, Verified Benchmark Datasets ($Re = 100 \to 100,000$) & Paper Preprint  
**Author:** Kartikey Singh (Department of Mathematics, University of Delhi, Delhi, 110007, India)  
**Permanent Zenodo DOI:** [10.5281/zenodo.18312938](https://doi.org/10.5281/zenodo.18312938)  
**License:** [MIT License](LICENSE)  

---

## 📌 Release Overview

This release provides the complete open-source scientific software suite, 30 verified high-resolution benchmark datasets, and the full research manuscript preprint accompanying the submission to *Computers & Fluids* (Elsevier):

> **"A High-Resolution Finite-Difference Framework for Steady Continuation and Finite-Window Unsteady Dynamics up to $Re = 100{,}000$ in 2D Lid-Driven Cavity Flows"**  
> *Author:* Kartikey Singh  
> *Preprint PDF:* [`paper/manuscript.pdf`](paper/manuscript.pdf) (25 pages, 32 verified references, 4 Appendices)  

---

## 📦 What is Included in this Release?

### 1. Research Manuscript & Submission Package
* **`paper/manuscript.pdf`**: The full compiled 25-page research article adhering to Elsevier *Computers & Fluids* standards.
* **`paper/manuscript.tex` & `paper/references.bib`**: Complete LaTeX sources and 32 audited bibliographic entries.
* **`submission_package/`**: Complete ready-to-submit archive including Cover Letter (`01_Cover_Letter.pdf`), Highlights (`02_Highlights.pdf`), Manuscript (`03_Manuscript.pdf`), Declaration of Competing Interest (`04_Declaration_of_Competing_Interest.pdf`), and Metadata (`05_Metadata_and_Reviewer_Details.txt`).

### 2. 30 Verified Numerical Flow Field Datasets (`data/`)
Serialized as standard NumPy binary archives (`.npz`), containing 2D coordinate grids `(X, Y)`, streamfunction $\psi$, scalar vorticity $\omega$, velocity components `(u, v)`, dynamic pressure $p$, and cell Péclet number distributions:
* **23 Stationary Continuation Solutions ($Re = 100 \to 100,000$):**
  * $Re = 100, 1{,}000$ on $129 \times 129$
  * $Re = 3{,}200, 5{,}000, 7{,}500, 10{,}000, 15{,}000, 20{,}000, 25{,}000, 30{,}000$ on $257 \times 257$
  * $Re = 10{,}000, 15{,}000, 20{,}000, 25{,}000, 30{,}000, 35{,}000, 40{,}000, 45{,}000, 50{,}000, 100{,}000$ on $513 \times 513$
  * $Re = 100{,}000$ ultra-resolution benchmark on $1025 \times 1025$ ($1{,}050{,}625$ nodes)
* **7 Finite-Window Unsteady Telemetry Records:**
  * $Re = 10{,}000, 15{,}000$ on $257 \times 257$ ($t = 20.0\text{--}22.0$)
  * $Re = 10{,}000, 15{,}000, 25{,}000, 50{,}000, 100{,}000$ on $513 \times 513$ ($t = 16.0\text{--}22.0$)
* All datasets include cryptographic SHA-256 checksums documented in Appendix A (`tab:dataset_manifest`).

### 3. High-Performance CFD Algorithms
* **Fast Discrete Sine Transform (DST-I) Poisson Solver:** Solves $\nabla_h^2 \psi = -\omega$ in $\mathcal{O}(N^2 \log N)$ complexity ($< 4\,\text{ms}$ on $513^2$, $< 24\,\text{ms}$ on $1025^2$).
* **Vectorized Alternating Direction Implicit (ADI) Marching:** Directional Thomas algorithm tridiagonal solves across entire grid slices simultaneously in NumPy.
* **Spatial Monotonicity & Hybrid Regularization:** Pure second-order central differencing ($\nu_{\text{art}} = 0$) preserved for all $Re \le 50{,}000$; directional hybrid differencing at $Re = 100{,}000$ with full dissipation tensor mapping.
* **ASME V&V 20 Uncertainty Verification:** Rigorous Grid Convergence Index (GCI) and Richardson extrapolation.

### 4. Interactive Tools & Animations
* **`cavity_hpc_colab.ipynb`**: 1-click cloud execution on Google Colab with GPU/CPU acceleration.
* **`plot_from_npz.py`**: Standalone tool to inspect flow fields and reproduce publication figures from `.npz` files.
* **`animate_shedding.py`**: Pipeline to generate high-resolution synchronized vortex shedding GIFs with live probe telemetry.

---

## 💻 Quick Start: How to Load Datasets

```python
import numpy as np
import matplotlib.pyplot as plt

# Load high-resolution stationary benchmark at Re = 100,000 on 1025 x 1025 mesh
data = np.load("data/flow_fields_Re100000_N1025.npz")

X = data["X"]          # 2D X coordinates (1025 x 1025)
Y = data["Y"]          # 2D Y coordinates (1025 x 1025)
psi = data["psi"]      # Streamfunction field
omega = data["omega"]  # Vorticity field
u = data["u"]          # Horizontal velocity
v = data["v"]          # Vertical velocity

print(f"Loaded Re = {data['Re']}, Grid = {data['N']}x{data['N']}")
print(f"Minimum Streamfunction psi_min = {psi.min():.7f}")
```

---

## 📜 How to Cite

```bibtex
@article{singh2026lid_driven_cavity,
  title={A High-Resolution Finite-Difference Framework for Steady Continuation and Finite-Window Unsteady Dynamics up to $Re = 100{,}000$ in 2D Lid-Driven Cavity Flows},
  author={Singh, Kartikey},
  journal={arXiv preprint / Submitted to Computers \& Fluids},
  year={2026}
}

@misc{singh2026solver,
  author       = {Singh, Kartikey},
  title        = {[dataset] Reference Finite-Difference Solutions and Verification Datasets for {2D} Lid-Driven Cavity Flows up to Extreme {Reynolds} Numbers},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18312938},
  url          = {https://doi.org/10.5281/zenodo.18312938}
}
```
