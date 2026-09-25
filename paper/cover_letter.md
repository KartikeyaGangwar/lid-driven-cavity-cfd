# Cover Letter for Journal Submission

**To:** Editor-in-Chief, *Computers & Fluids* (Elsevier)  
**Date:** September 2026  
**Subject:** Submission of Original Research Manuscript  

**Manuscript Title:**  
A High-Resolution Finite-Difference Framework for Steady Continuation and Finite-Window Unsteady Dynamics up to $Re = 100{,}000$ in 2D Lid-Driven Cavity Flows  

**Author:** Kartikey Singh  
Department of Mathematics, University of Delhi, Delhi, 110007, India  
Email: kartikeysingh525@protonmail.com | ORCID: https://orcid.org/0009-0009-1973-7532  

---

Dear Editor-in-Chief and Editorial Board,

We are pleased to submit our original research manuscript entitled **"A High-Resolution Finite-Difference Framework for Steady Continuation and Finite-Window Unsteady Dynamics up to $Re = 100{,}000$ in 2D Lid-Driven Cavity Flows"** for consideration as a regular research article in *Computers \& Fluids*.

### Significance and Key Contributions:
The incompressible 2D lid-driven cavity flow remains the canonical proving ground for computational fluid dynamics. However, established benchmarks stagnate at $Re \le 10,000$ (Ghia et al., 1982) or $Re \le 25,000$ (Erturk et al., 2005). In this manuscript, we push the numerical and scientific frontiers through several key contributions:
1. **Exact Spectral Poisson Acceleration:** We integrate an exact Discrete Sine Transform (DST-I) Poisson solver operating at optimal $\mathcal{O}(N^2 \log N)$ complexity, eliminating iterative residual drift, pressure-velocity decoupling, and numerical dissipation ($\nu_{\text{art}} = 0$).
2. **Asymptotic Core Vorticity Homogenization ($Re \to 100,000$):** We compute steady continuation solutions up to $Re = 100,000$ on ultra-fine grids ($513 \times 513$) and a high-resolution grid ($1025 \times 1025$, $1,050,625$ nodes). We observe core behavior qualitatively consistent with Batchelor's (1956) asymptotic theorem ($\omega_0 \approx -2.54$) alongside mid-plane symmetry ($x_c \to 0.50000$).
3. **Interior Hybrid Differencing & Dissipation Mapping:** At $Re = 100,000$ ($Pe = 195.3$), an interior hybrid differencing scheme regularizes corner singularity dispersion with added numerical viscosity tensor $\boldsymbol{\nu}_{\text{art}} = \operatorname{diag}(\nu_{\text{art},x}, \nu_{\text{art},y})$, yielding directional mean ratio $\bar{\nu}_{\text{art}}/\nu = 1.485$ at the geometric cavity center $(0.5, 0.5)$ and $48.93$ near the lid on $513 \times 513$ (dropping to $0.74$ and $24.46$ on $1025 \times 1025$), with pure central differencing strictly preserved everywhere for all $Re \le 50,000$.
4. **Finite-Window Unsteady Marching & Spectral Dynamics:** Through physical time-accurate marching on $513 \times 513$ meshes ($t = 16.0$--$22.0$ c.t.u., $St \equiv fL/U$), we systematically resolve the dominant periodic shedding mode ($Re = 10,000$, $St = 0.6249 \pm 0.0416$), modulated multi-frequency dynamics ($Re = 15,000$, $St = 0.5364 \pm 0.0278$), folded multi-frequency orbits ($Re = 25,000$, $St = 0.0953 \pm 0.0281$), filtered-boundary shedding ($Re = 50,000$, $St = 0.2581 \pm 0.0300$), and hybrid filtered-boundary shedding at $Re = 100,000$ ($St = 0.1908 \pm 0.0379$), reporting discrete Fourier bin half-width resolution ($\pm \Delta f / 2$).
5. **Scientific Reproducibility & Open Benchmarks:** We openly release thirty (30) verified benchmark full 2D coordinate/flow field datasets (23 steady continuation and 7 unsteady records), phase portrait scripts, and automated visualization tools under the MIT License with permanent Zenodo DOIs.

### Suggested Potential Reviewers:
1. **Prof. Ercan Erturk** (Kocaeli University / Istanbul Technical University, Turkey) — `erturke@itu.edu.tr`
2. **Prof. Olivier Botella** (Université de Lorraine, LEMTA, France) — `olivier.botella@univ-lorraine.fr`
3. **Prof. Chang Shu** (National University of Singapore, Singapore) — `mpeshuc@nus.edu.sg`
4. **Prof. Charles-Henri Bruneau** (Université de Bordeaux, France) — `charles-henri.bruneau@u-bordeaux.fr`

### Declarations:
- This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere.
- All data and tools are openly available.
- There are no competing financial or non-financial interests to declare.

Sincerely yours,  
**Kartikey Singh**  
Department of Mathematics, University of Delhi, Delhi, 110007, India  
Email: kartikeysingh525@protonmail.com | ORCID: https://orcid.org/0009-0009-1973-7532  
