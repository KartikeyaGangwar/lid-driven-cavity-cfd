# Cover Letter for Journal Submission

**To:** Editor-in-Chief, *Computers & Fluids* (Elsevier)  
**Date:** September 2026  
**Subject:** Submission of Original Research Manuscript  

**Manuscript Title:**  
A High-Resolution Finite-Difference Benchmark for Steady Continuation and Hopf Bifurcation Dynamics up to Extreme Reynolds Numbers ($Re = 100,000$) in 2D Lid-Driven Cavity Flows  

**Author:** Kartikey Singh  
Department of Mathematics, University of Delhi, Delhi, 110007, India  
Email: kartikeysingh525@protonmail.com | ORCID: https://orcid.org/0009-0009-1973-7532  

---

Dear Editor-in-Chief and Editorial Board,

We are pleased to submit our original research manuscript entitled **"A High-Resolution Finite-Difference Benchmark for Steady Continuation and Hopf Bifurcation Dynamics up to Extreme Reynolds Numbers ($Re = 100,000$) in 2D Lid-Driven Cavity Flows"** for consideration as a regular research article in *Computers \& Fluids*.

### Significance and Key Contributions:
The incompressible 2D lid-driven cavity flow remains the canonical proving ground for computational fluid dynamics. However, established benchmarks stagnate at $Re \le 10,000$ (Ghia et al., 1982) or $Re \le 25,000$ (Erturk et al., 2005). In this manuscript, we push the numerical and scientific frontiers through several key contributions:
1. **Exact Spectral Poisson Acceleration:** We integrate an exact Discrete Sine Transform (DST-I) Poisson solver operating at optimal $\mathcal{O}(N^2 \log N)$ complexity, eliminating iterative residual drift, pressure-velocity decoupling, and numerical dissipation.
2. **Asymptotic Core Vorticity Homogenization ($Re \to 100,000$):** We compute steady continuation solutions up to $Re = 100,000$ on ultra-fine grids ($513 \times 513$) and a high-resolution grid ($1025 \times 1025$, $1,050,625$ nodes). We verify Batchelor's (1956) asymptotic recirculating core theorem through the plateau $\omega_0 = -2.5418$ alongside symmetric mid-plane alignment ($x_c \to 0.50000$).
3. **Localized Hybrid Differencing & Dissipation Mapping:** We incorporate a localized Spalding–Patankar hybrid differencing scheme at $Re = 100,000$ ($Pe = 195.3 \gg 2$) that regularizes corner singularity shocks while providing a rigorous 2D spatial audit of numerical viscosity ($\nu_{\text{num}}/\nu \le 0.99$ in the core on $513 \times 513$ and $0.24$ on $1025 \times 1025$), with pure central differencing ($\nu_{\text{num}} = 0$) strictly preserved everywhere for all $Re \le 50,000$.
4. **Extended Unsteady Marching & Bifurcation Cascades:** Through time-accurate physical marching on $513 \times 513$ meshes, we establish the primary supercritical Hopf limit cycle over extended convective time ($t = 22.0$) with cycle recurrence error $\epsilon_{\text{rec}} < 0.08\%$ ($St = 0.6661$), capture secondary Hopf 2-torus quasi-periodicity ($St = 0.6662$), and trace nonlinear unsteady dynamics across $Re = 25,000$ ($St = 0.1282$), $Re = 50,000$ ($St = 0.4615$), and $Re = 100,000$ ($St = 0.1923$).
5. **Scientific Reproducibility & Open Benchmarks:** We openly release 28 full 2D coordinate/flow field datasets, phase portrait scripts, and automated visualization tools under the MIT License with permanent Zenodo DOIs.

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
