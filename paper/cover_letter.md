# Cover Letter for Journal Submission

**To:** Editor-in-Chief, *Computers & Fluids* (Elsevier)  
**Date:** September 14, 2026  
**Subject:** Submission of Original Research Article  

**Manuscript Title:**  
A High-Resolution Finite-Difference Benchmark for Steady Continuation and Hopf Bifurcation Dynamics up to Extreme Reynolds Numbers ($Re = 100,000$) in 2D Lid-Driven Cavity Flows  

**Author:** Kartikeya Gangwar  
Department of Mechanical Engineering, Indian Institute of Technology / Research Affiliate  
Email: kartikeyagangwar@alumni.iit.ac.in  

---

Dear Editor-in-Chief and Editorial Board,

We are pleased to submit our original research manuscript entitled **"A High-Resolution Finite-Difference Benchmark for Steady Continuation and Hopf Bifurcation Dynamics up to Extreme Reynolds Numbers ($Re = 100,000$) in 2D Lid-Driven Cavity Flows"** for consideration as a regular research paper in *Computers \& Fluids*.

### Significance and Key Contributions:
The incompressible 2D lid-driven cavity flow remains the quintessential proving ground for computational fluid dynamics. However, established benchmarks have largely stagnated at $Re \le 10,000$ (e.g., Ghia et al., 1982) or $Re \le 25,000$ (Erturk et al., 2005) due to numerical divergence stemming from the lid corner singularities and escalating cell Péclet numbers ($Pe \gg 2$). 

In this manuscript, we push the resolution and Reynolds frontiers of the standard finite-difference streamfunction-vorticity $(\psi\text{--}\omega)$ formulation through several key technical advances:
1. **Exact Spectral Poisson Acceleration:** We integrate an exact Discrete Sine Transform (DST-I) Poisson solver operating at optimal $\mathcal{O}(N^2 \log N)$ complexity, eliminating iterative convergence errors, pressure-velocity decoupling, and artificial numerical dissipation ($\nu_{\text{num}} = 0$).
2. **Asymptotic Batchelor Core Verification at $Re = 100,000$:** We compute steady continuation solutions up to an unprecedented $Re = 100,000$ on ultra-fine grids ($513 \times 513$) and a mega-mesh ($1025 \times 1025$, $1,050,625$ nodes). We quantitatively demonstrate convergence to Batchelor's (1956) classical asymptotic core theorem: the primary vortex center migrates exactly to the geometric mid-plane ($x_c \to 0.50000$), while core vorticity stabilizes to an invariant plateau value of $\omega_0 = -2.5418$.
3. **Localized Singular Shock Regularization:** At $Re = 100,000$ ($Pe = 195.3 \gg 2$), we formulate a localized Spalding–Patankar hybrid differencing scheme that prevents spurious numerical oscillations at the lid corner singularity without polluting the interior recirculating core.
4. **Unsteady Bifurcation Mapping on $513 \times 513$ Meshes:** Through time-accurate physical marching, we systematically resolve the primary supercritical Hopf bifurcation ($Re = 10,000$, $St = 0.6661$), the secondary Hopf 2-torus quasi-periodic attractor ($Re = 15,000$, $St = 0.6662$), multi-harmonic folded attractors ($Re = 25,000$ and $Re = 50,000$), and extreme frontier limit cycles at $Re = 100,000$ ($St = 0.1923$, $T = 5.20\,\text{s}$).
5. **Commitment to Open Science and Reproducibility:** We openly release 28 full 2D coordinate/flow field datasets, phase portrait scripts, and automated visualization tools under the MIT License with permanent Zenodo DOIs.

### Suggested Potential Reviewers:
1. **Prof. Ercan Erturk** (Kocaeli University / ITU) — Leading authority on high-Reynolds lid-driven cavity benchmarking and steady Navier–Stokes solutions up to $Re = 25,000$.
2. **Prof. Olivier Botella** (Université de Lorraine, LEMTA) — Renowned specialist in high-order compact finite difference schemes and incompressible cavity benchmarks.
3. **Prof. Chang Shu** (National University of Singapore) — Pioneer of differential quadrature and lattice Boltzmann / FDM formulations for high-Reynolds cavity flows.
4. **Prof. Jie Shen** (Purdue University) — Expert in spectral methods, projection schemes, and Navier–Stokes stability.

### Declarations:
- This manuscript is an original work, has not been published previously, and is not under simultaneous consideration for publication elsewhere.
- All authors have read and approved the final manuscript.
- There are no competing financial or non-financial interests to declare.

Sincerely yours,  
**Kartikeya Gangwar**  
Department of Mechanical Engineering  
Indian Institute of Technology / Research Affiliate  
Email: kartikeyagangwar@alumni.iit.ac.in  
