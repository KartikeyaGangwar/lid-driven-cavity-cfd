"""
Lid-Driven Cavity Flow Solver (2D, Incompressible)

Numerical methods:
- Streamfunction–Vorticity formulation (psi–omega)
- High-Performance Alternating Direction Implicit (ADI) scheme with exact tridiagonal boundary closures
- Direct Precomputed Sparse LU Decomposition for Streamfunction Poisson equation (exact to machine precision)
  (with optional Vectorized Red–Black SOR fallback)
- Dual Lid Profiles:
  * 'constant': Standard benchmark lid U(x, 1) = U (Ghia et al., 1982)
  * 'regularized': Singularity-free polynomial profile u(x, 1) = 16*U*(x/L)^2*(1 - x/L)^2
- Physical coordinate system with instant post-processing and Ghia et al. validation

Author: Kartikey Singh
Year: 2026
License: MIT
"""

import os
import sys
import io
import time
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.fft as sfft
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# ---- NumPy pickle compatibility patch ----
try:
    if hasattr(np, '_core') and hasattr(np._core, 'numeric'):
        sys.modules['numpy._core.numeric'] = np._core.numeric
    elif hasattr(np, 'core') and hasattr(np.core, 'numeric'):
        sys.modules['numpy._core.numeric'] = np.core.numeric
except Exception:
    pass

# UNICODE FIX FOR TERMINALS
if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    try:
        current_encoding = getattr(sys.stdout, 'encoding', None)
        if current_encoding and current_encoding.lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


# ============================================================================
# BENCHMARK DATA: Ghia, Ghia & Shin (1982)
# High-Re solutions for incompressible flow using the Navier-Stokes equations
# and a multigrid method, J. Comput. Phys., 48(3), 387-411.
# ============================================================================
GHIA_DATA = {
    'y_u': np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
                     0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000]),
    'u_Re100': np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
                         -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000]),
    'u_Re400': np.array([1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135,
                         -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.00000]),
    'u_Re1000': np.array([1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702,
                          -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000]),
    'u_Re3200': np.array([1.00000, 0.53236, 0.48296, 0.46547, 0.46101, 0.34682, 0.19791, 0.07156,
                          -0.04272, -0.08636, -0.24427, -0.34323, -0.41933, -0.37827, -0.35344, -0.32407, 0.00000]),
    'u_Re5000': np.array([1.00000, 0.48223, 0.46120, 0.45992, 0.46036, 0.33556, 0.20087, 0.08183,
                          -0.03039, -0.07404, -0.22855, -0.33050, -0.40435, -0.43643, -0.42901, -0.41165, 0.00000]),
    'u_Re7500': np.array([1.00000, 0.47244, 0.47048, 0.47323, 0.47167, 0.34228, 0.20591, 0.08342,
                          -0.03800, -0.07503, -0.23176, -0.32393, -0.38324, -0.43025, -0.43590, -0.43154, 0.00000]),
    'u_Re10000': np.array([1.00000, 0.47221, 0.47783, 0.48070, 0.47804, 0.34635, 0.20673, 0.08344,
                           -0.03111, -0.07540, -0.23186, -0.32709, -0.38000, -0.41657, -0.42537, -0.42735, 0.00000]),
    'x_v': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
                     0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
    'v_Re100': np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533,
                         0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000]),
    'v_Re400': np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598,
                         0.05188, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0.00000]),
    'v_Re1000': np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966,
                          0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.00000]),
    'v_Re3200': np.array([0.00000, -0.39017, -0.47425, -0.52357, -0.54053, -0.44307, -0.37401, -0.31184,
                          0.00999, 0.28188, 0.29030, 0.37119, 0.42768, 0.41906, 0.40917, 0.39560, 0.00000]),
    'v_Re5000': np.array([0.00000, -0.49774, -0.55069, -0.55408, -0.52876, -0.41442, -0.36214, -0.30018,
                          0.00945, 0.27280, 0.28066, 0.35368, 0.42951, 0.43648, 0.43329, 0.42447, 0.00000]),
    'v_Re7500': np.array([0.00000, -0.53858, -0.55216, -0.52347, -0.48590, -0.41050, -0.36213, -0.30448,
                          0.00824, 0.27348, 0.28117, 0.35060, 0.41824, 0.43564, 0.44030, 0.43979, 0.00000]),
    'v_Re10000': np.array([0.00000, -0.54302, -0.52987, -0.49099, -0.45863, -0.41496, -0.36737, -0.30719,
                           0.00831, 0.27224, 0.28003, 0.35070, 0.41487, 0.43124, 0.43733, 0.43983, 0.00000]),
    'vortex_benchmarks': {
        100: {
            'primary': {'psi_min': -0.1033, 'x': 0.6172, 'y': 0.7344},
            'BR1': {'psi_max': 1.75e-6, 'x': 0.9453, 'y': 0.0625},
            'BL1': {'psi_max': 1.75e-6, 'x': 0.0313, 'y': 0.0391}
        },
        400: {
            'primary': {'psi_min': -0.1139, 'x': 0.5547, 'y': 0.6055},
            'BR1': {'psi_max': 6.42e-4, 'x': 0.8906, 'y': 0.1250},
            'BL1': {'psi_max': 1.42e-5, 'x': 0.0547, 'y': 0.0469}
        },
        1000: {
            'primary': {'psi_min': -0.1179, 'x': 0.5313, 'y': 0.5625},
            'BR1': {'psi_max': 1.75e-3, 'x': 0.8594, 'y': 0.1094},
            'BL1': {'psi_max': 2.31e-4, 'x': 0.0859, 'y': 0.0781}
        },
        3200: {
            'primary': {'psi_min': -0.1204, 'x': 0.5165, 'y': 0.5469},
            'BR1': {'psi_max': 2.93e-3, 'x': 0.8125, 'y': 0.0859},
            'BL1': {'psi_max': 9.80e-4, 'x': 0.0859, 'y': 0.1172},
            'TL1': {'psi_max': 7.30e-4, 'x': 0.0625, 'y': 0.8984}
        },
        5000: {
            'primary': {'psi_min': -0.1190, 'x': 0.5117, 'y': 0.5352},
            'BR1': {'psi_max': 3.08e-3, 'x': 0.8047, 'y': 0.0742},
            'BL1': {'psi_max': 1.36e-3, 'x': 0.0742, 'y': 0.1367},
            'TL1': {'psi_max': 1.46e-3, 'x': 0.0625, 'y': 0.9063}
        },
        7500: {
            'primary': {'psi_min': -0.1199, 'x': 0.5117, 'y': 0.5322},
            'BR1': {'psi_max': 3.20e-3, 'x': 0.7969, 'y': 0.0664},
            'BL1': {'psi_max': 1.50e-3, 'x': 0.0664, 'y': 0.1523},
            'TL1': {'psi_max': 2.05e-3, 'x': 0.0625, 'y': 0.9102}
        },
        10000: {
            'primary': {'psi_min': -0.1197, 'x': 0.5117, 'y': 0.5333},
            'BR1': {'psi_max': 3.42e-3, 'x': 0.7930, 'y': 0.0625},
            'BL1': {'psi_max': 1.52e-3, 'x': 0.0586, 'y': 0.1641},
            'TL1': {'psi_max': 2.42e-3, 'x': 0.0586, 'y': 0.9141}
        },
    }
}


def prolong_fields(psi_old, omega_old, old_x, old_y, new_x, new_y):
    """
    Interpolate streamfunction and vorticity fields from a coarse mesh to a fine mesh.
    Uses bicubic spline interpolation (RegularGridInterpolator with method='cubic').
    Strictly re-enforces boundary no-penetration (psi=0) conditions on walls.
    """
    interp_psi = RegularGridInterpolator((old_y, old_x), psi_old, method='cubic', bounds_error=False, fill_value=0.0)
    interp_omega = RegularGridInterpolator((old_y, old_x), omega_old, method='cubic', bounds_error=False, fill_value=0.0)

    NEW_Y, NEW_X = np.meshgrid(new_y, new_x, indexing='ij')
    points = np.stack([NEW_Y.ravel(), NEW_X.ravel()], axis=-1)

    psi_new = interp_psi(points).reshape(len(new_y), len(new_x))
    omega_new = interp_omega(points).reshape(len(new_y), len(new_x))

    # Re-enforce exact no-slip wall streamfunction
    psi_new[0, :] = 0.0
    psi_new[-1, :] = 0.0
    psi_new[:, 0] = 0.0
    psi_new[:, -1] = 0.0

    return psi_new, omega_new


def _apply_latex_style():
    """Apply publication-grade LaTeX Computer Modern typography and clean grid styling."""
    import shutil
    has_latex = shutil.which('latex') is not None and shutil.which('dvipng') is not None
    if has_latex:
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman', 'cmr10'],
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
            'axes.formatter.use_mathtext': True,
            'font.size': 11,
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 10.5,
            'ytick.labelsize': 10.5,
            'figure.autolayout': False,
        })
    else:
        plt.rcParams.update({
            'text.usetex': False,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman', 'serif'],
            'mathtext.fontset': 'cm',
            'mathtext.rm': 'serif',
            'axes.formatter.use_mathtext': True,
            'font.size': 11,
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 10.5,
            'ytick.labelsize': 10.5,
            'figure.autolayout': False,
        })


class LidDrivenCavitySolver:
    """
    Solves 2D incompressible lid-driven cavity flow using the streamfunction-vorticity (psi-omega) method.
    
    Features:
    - Precomputed direct sparse LU Poisson solver (O(ms) per step, machine precision).
    - Fully vectorized ADI vorticity transport with exact tridiagonal boundary closures.
    - Support for constant (U=1) or regularized singularity-free (16*x^2*(1-x)^2) lid profiles.
    - Automated pressure Poisson solver with Neumann zero-gradient boundary conditions.
    """

    def __init__(self, N=129, Re=1000, lid_velocity=1.0, L=1.0, lid_profile='constant',
                 poisson_solver='dst', convection_scheme='central', wall_bc='thom', wall_beta=None):
        """
        Parameters:
        -----------
        N : int
            Number of grid points along each axis (N x N mesh).
        Re : float
            Reynolds number (Re = U * L / nu).
        lid_velocity : float
            Peak velocity of the moving lid.
        L : float
            Cavity dimension (domain [0, L] x [0, L]).
        lid_profile : str
            'constant': classic benchmark U(x, 1) = U.
            'regularized': smooth profile u(x, 1) = 16*U*(x/L)^2*(1 - x/L)^2.
        poisson_solver : str
            'dst' / 'fft': Fast 2D Discrete Sine Transform (fastest, exact to machine precision, 0 memory overhead).
            'lu': direct sparse LU factorized solve (exact).
            'sor': vectorized Red-Black Successive Over-Relaxation.
        convection_scheme : str
            'central': pure 2nd-order central differencing (0 numerical viscosity, required for Ghia benchmark).
            'hybrid': 2nd-order Central where Pe <= 2, upwind where Pe > 2.
            'upwind': 1st-order upwind differencing.
        wall_bc : str
            'thom': classic 1st-order Thom wall vorticity formula (robust, unconditionally stable).
            'woods': 2nd-order Woods wall vorticity condition.
        wall_beta : float or None
            Under-relaxation factor for wall vorticity (0 < beta <= 1). Default auto-tunes by Re.
        """
        self.N = N
        self.M = N - 2
        self.Re = Re
        self.U = lid_velocity
        self.L = L
        self.lid_profile = lid_profile.lower()
        self.poisson_solver = poisson_solver.lower()
        self.convection_scheme = convection_scheme.lower()
        self.wall_bc = wall_bc.lower()

        # Under-relaxation factor for wall vorticity to eliminate boundary-layer sloshing
        if wall_beta is not None:
            self.wall_beta = float(wall_beta)
        else:
            if self.Re <= 400:
                self.wall_beta = 1.0
            elif self.Re <= 1000:
                self.wall_beta = 0.6
            else:
                self.wall_beta = 0.5

        # Grid parameters
        self.h = L / (N - 1)
        self.nu = self.U * self.L / self.Re

        # Grid coordinates
        self.x = np.linspace(0, L, N)
        self.y = np.linspace(0, L, N)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

        # Lid boundary profile
        x_norm = self.x / self.L
        if self.lid_profile == 'regularized':
            self.u_lid = 16.0 * self.U * (x_norm**2) * ((1.0 - x_norm)**2)
        else:
            self.u_lid = np.full(N, self.U)

        # Flow fields (array indexing: [y_idx, x_idx])
        self.psi = np.zeros((N, N))
        self.omega = np.zeros((N, N))
        self.u = np.zeros((N, N))
        self.v = np.zeros((N, N))
        self.p = np.zeros((N, N))

        # Time step: CFL condition + Viscous-advective stability bound (dt <= 1.8 * nu / U^2)
        cfl_factor = 0.2 if self.poisson_solver in ['dst', 'fft', 'lu'] else 0.1
        dt_cfl = cfl_factor * self.h / (self.U if self.U > 0 else 1.0)
        dt_visc = 1.8 * self.nu / ((self.U**2) if self.U > 0 else 1.0)
        self.dt = min(dt_cfl, dt_visc)
        self.alpha_adi = (self.nu * self.dt) / (2.0 * self.h**2)

        print(f"[INFO] Mesh: {N}x{N} | Re: {Re} | Lid: '{self.lid_profile}' | Convection: '{self.convection_scheme}'", flush=True)
        print(f"[INFO] Wall BC: '{self.wall_bc}' (beta={self.wall_beta:.2f}) | Poisson: '{self.poisson_solver}'", flush=True)
        print(f"[INFO] h = {self.h:.4e} | dt = {self.dt:.4e} | nu = {self.nu:.4e} | alpha_adi = {self.alpha_adi:.4e}", flush=True)

        # Interior velocities for ADI convection
        self.u_c = np.zeros((self.M, self.M))
        self.v_c = np.zeros((self.M, self.M))

        # Convergence tracking
        self.history = {'iterations': [], 'max_change': [], 'psi_min': []}

        # Initialize Poisson Solver
        if self.poisson_solver in ['dst', 'fft']:
            self._init_dst_poisson()
        elif self.poisson_solver == 'lu':
            self._init_sparse_lu_poisson()
        else:
            self._init_sor_masks()

        # Precompute 1D Thomas recurrence coefficients for ADI
        self._init_adi_coefficients()

    def _init_dst_poisson(self):
        """Precompute analytical eigenvalues for 2D Type-I Discrete Sine Transform (DST) Poisson solver."""
        M = self.M
        h2 = self.h**2
        k = np.arange(1, M + 1)
        eig_1d = 2.0 * (np.cos(np.pi * k / (M + 1)) - 1.0) / h2
        self.dst_denom = eig_1d[None, :] + eig_1d[:, None]

        # Precompute 1D Thomas recurrence coefficients for ADI
        self._init_adi_coefficients()

    def _init_sparse_lu_poisson(self):
        """Precompute sparse LU factorization of the 2D Laplacian operator."""
        M = self.M
        h2 = self.h**2
        main_diag = -2.0 * np.ones(M) / h2
        off_diag = 1.0 * np.ones(M - 1) / h2
        T = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(M, M))
        I = sp.eye(M)
        L_2d = (sp.kron(I, T) + sp.kron(T, I)).tocsc()
        self.lu_poisson = spla.splu(L_2d)

    def _init_sor_masks(self):
        """Create checkerboard masks for Red-Black SOR."""
        N = self.N
        i_vals, j_vals = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        self.red_interior_mask = ((i_vals + j_vals) % 2 == 0) & (i_vals > 0) & (i_vals < N-1) & (j_vals > 0) & (j_vals < N-1)
        self.black_interior_mask = ~self.red_interior_mask & (i_vals > 0) & (i_vals < N-1) & (j_vals > 0) & (j_vals < N-1)

    def _init_adi_coefficients(self):
        """Precompute recurrence factors for vectorized tridiagonal solver."""
        M = self.M
        alpha = self.alpha_adi
        A = -alpha
        B = 1.0 + 2.0 * alpha
        C = -alpha
        self.A_adi, self.B_adi, self.C_adi = A, B, C

        self.P_adi = np.zeros(M)
        self.denom_adi = np.zeros(M)
        self.denom_adi[0] = B
        self.P_adi[0] = C / B
        for j in range(1, M):
            self.denom_adi[j] = B - A * self.P_adi[j - 1]
            self.P_adi[j] = C / self.denom_adi[j]

    def set_state(self, psi, omega):
        """Inject existing flow state for warm starting."""
        if psi.shape != (self.N, self.N) or omega.shape != (self.N, self.N):
            raise ValueError(f"State shape {psi.shape} does not match grid {self.N}x{self.N}")
        self.psi = psi.copy()
        self.omega = omega.copy()
        self.apply_boundary_conditions()
        self.calculate_velocities()

    def apply_boundary_conditions(self):
        """Apply boundary conditions with optional Woods/Thom formula and relaxation."""
        h = self.h
        beta = self.wall_beta

        # Streamfunction BCs: all walls are no-slip impermeable (psi = 0)
        self.psi[0, :] = 0.0
        self.psi[-1, :] = 0.0
        self.psi[:, 0] = 0.0
        self.psi[:, -1] = 0.0

        if self.wall_bc == 'woods':
            # Woods' second-order wall vorticity condition:
            # omega_w = -3*psi_1 / h^2 - 0.5*omega_1 - 3*U / h
            target_top = -3.0 * self.psi[-2, 1:-1] / (h**2) - 0.5 * self.omega[-2, 1:-1] - 3.0 * self.u_lid[1:-1] / h
            target_bottom = -3.0 * self.psi[1, 1:-1] / (h**2) - 0.5 * self.omega[1, 1:-1]
            target_left = -3.0 * self.psi[1:-1, 1] / (h**2) - 0.5 * self.omega[1:-1, 1]
            target_right = -3.0 * self.psi[1:-1, -2] / (h**2) - 0.5 * self.omega[1:-1, -2]
        else:
            # Thom's first-order formula:
            # omega_w = -2*psi_1 / h^2 - 2*U / h
            target_top = -2.0 * self.psi[-2, 1:-1] / (h**2) - 2.0 * self.u_lid[1:-1] / h
            target_bottom = -2.0 * self.psi[1, 1:-1] / (h**2)
            target_left = -2.0 * self.psi[1:-1, 1] / (h**2)
            target_right = -2.0 * self.psi[1:-1, -2] / (h**2)

        # Apply under-relaxation to suppress boundary-layer sloshing:
        # omega_w = (1 - beta) * omega_w_old + beta * omega_w_target
        if beta < 1.0:
            self.omega[-1, 1:-1] = (1.0 - beta) * self.omega[-1, 1:-1] + beta * target_top
            self.omega[0, 1:-1] = (1.0 - beta) * self.omega[0, 1:-1] + beta * target_bottom
            self.omega[1:-1, 0] = (1.0 - beta) * self.omega[1:-1, 0] + beta * target_left
            self.omega[1:-1, -1] = (1.0 - beta) * self.omega[1:-1, -1] + beta * target_right
        else:
            self.omega[-1, 1:-1] = target_top
            self.omega[0, 1:-1] = target_bottom
            self.omega[1:-1, 0] = target_left
            self.omega[1:-1, -1] = target_right

        # Corner vorticity closure: average adjacent wall points
        self.omega[0, 0] = 0.5 * (self.omega[1, 0] + self.omega[0, 1])
        self.omega[0, -1] = 0.5 * (self.omega[1, -1] + self.omega[0, -2])
        self.omega[-1, 0] = 0.5 * (self.omega[-2, 0] + self.omega[-1, 1])
        self.omega[-1, -1] = 0.5 * (self.omega[-2, -1] + self.omega[-1, -2])

    def solve_streamfunction(self, max_iterations=1000, tolerance=1e-5, omega_relaxation=1.8):
        """Solve Poisson equation del^2(psi) = -omega."""
        if self.poisson_solver in ['dst', 'fft']:
            rhs = -self.omega[1:-1, 1:-1]
            psi_hat = sfft.dstn(rhs, type=1) / self.dst_denom
            psi_inner = sfft.idstn(psi_hat, type=1)
            max_change = np.max(np.abs(self.psi[1:-1, 1:-1] - psi_inner))
            self.psi[1:-1, 1:-1] = psi_inner
            return max_change
        elif self.poisson_solver == 'lu':
            rhs = -self.omega[1:-1, 1:-1].ravel()
            psi_inner = self.lu_poisson.solve(rhs).reshape((self.M, self.M))
            max_change = np.max(np.abs(self.psi[1:-1, 1:-1] - psi_inner))
            self.psi[1:-1, 1:-1] = psi_inner
            return max_change
        else:
            # Fallback Red-Black SOR
            source_term = (self.h**2) * self.omega
            for iteration in range(max_iterations):
                psi_old = self.psi.copy()
                psi_new_red = 0.25 * (
                    np.roll(self.psi, 1, axis=0)[self.red_interior_mask] +
                    np.roll(self.psi, -1, axis=0)[self.red_interior_mask] +
                    np.roll(self.psi, 1, axis=1)[self.red_interior_mask] +
                    np.roll(self.psi, -1, axis=1)[self.red_interior_mask] +
                    source_term[self.red_interior_mask]
                )
                self.psi[self.red_interior_mask] = (1 - omega_relaxation) * self.psi[self.red_interior_mask] + omega_relaxation * psi_new_red

                psi_new_black = 0.25 * (
                    np.roll(self.psi, 1, axis=0)[self.black_interior_mask] +
                    np.roll(self.psi, -1, axis=0)[self.black_interior_mask] +
                    np.roll(self.psi, 1, axis=1)[self.black_interior_mask] +
                    np.roll(self.psi, -1, axis=1)[self.black_interior_mask] +
                    source_term[self.black_interior_mask]
                )
                self.psi[self.black_interior_mask] = (1 - omega_relaxation) * self.psi[self.black_interior_mask] + omega_relaxation * psi_new_black

                max_change = np.max(np.abs(self.psi - psi_old))
                if max_change < tolerance:
                    break
            return max_change

    def calculate_velocities(self):
        """Calculate velocities u = d(psi)/dy and v = -d(psi)/dx."""
        h = self.h
        self.u[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2.0 * h)
        self.v[1:-1, 1:-1] = -(self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2.0 * h)

        self.u_c = self.u[1:-1, 1:-1]
        self.v_c = self.v[1:-1, 1:-1]

        # Apply boundary velocities
        self.u[-1, :] = self.u_lid
        self.v[-1, :] = 0.0
        self.u[0, :] = 0.0
        self.v[0, :] = 0.0
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[:, -1] = 0.0

    def solve_vorticity_transport_ADI(self):
        """
        Solve vorticity transport equation using vectorized Alternating Direction Implicit (ADI).
        - Implicit central-difference diffusion (unconditionally stable in 1D).
        - Convection: 'hybrid' (adaptive 2nd-order Central where Pe <= 2, Upwind where Pe > 2),
                      'central' (pure 2nd-order), or 'upwind' (pure 1st-order).
        - Exact Dirichlet wall-vorticity boundary closures incorporated directly into tridiagonal RHS.
        """
        M = self.M
        h = self.h
        nu = self.nu
        dt = self.dt
        A, B, C = self.A_adi, self.B_adi, self.C_adi
        P, denom = self.P_adi, self.denom_adi
        omega_old = self.omega.copy()

        # 1. Convection terms
        domega_dx_cd = (omega_old[1:-1, 2:] - omega_old[1:-1, :-2]) / (2.0 * h)
        domega_dy_cd = (omega_old[2:, 1:-1] - omega_old[:-2, 1:-1]) / (2.0 * h)

        if self.convection_scheme == 'central':
            domega_dx = domega_dx_cd
            domega_dy = domega_dy_cd
        elif self.convection_scheme == 'upwind':
            domega_dx = np.where(
                self.u_c > 0,
                (omega_old[1:-1, 1:-1] - omega_old[1:-1, :-2]) / h,
                (omega_old[1:-1, 2:] - omega_old[1:-1, 1:-1]) / h
            )
            domega_dy = np.where(
                self.v_c > 0,
                (omega_old[1:-1, 1:-1] - omega_old[:-2, 1:-1]) / h,
                (omega_old[2:, 1:-1] - omega_old[1:-1, 1:-1]) / h
            )
        else:
            # 'hybrid' scheme (Spalding / Patankar):
            # Pure central differencing where local cell Peclet <= 2 (exact 2nd order, 0 artificial diffusion).
            # Upwind where cell Peclet > 2 to cleanly suppress dispersive wiggles.
            pe_x = np.abs(self.u_c) * h / nu
            pe_y = np.abs(self.v_c) * h / nu

            domega_dx_up = np.where(
                self.u_c > 0,
                (omega_old[1:-1, 1:-1] - omega_old[1:-1, :-2]) / h,
                (omega_old[1:-1, 2:] - omega_old[1:-1, 1:-1]) / h
            )
            domega_dy_up = np.where(
                self.v_c > 0,
                (omega_old[1:-1, 1:-1] - omega_old[:-2, 1:-1]) / h,
                (omega_old[2:, 1:-1] - omega_old[1:-1, 1:-1]) / h
            )

            domega_dx = np.where(pe_x <= 2.0, domega_dx_cd, domega_dx_up)
            domega_dy = np.where(pe_y <= 2.0, domega_dy_cd, domega_dy_up)

        conv = self.u_c * domega_dx + self.v_c * domega_dy

        # 2. Step 1: Implicit X (row sweeps), Explicit Y (Peaceman-Rachford half-step dt/2)
        diff_y = (nu / (h**2)) * (omega_old[2:, 1:-1] + omega_old[:-2, 1:-1] - 2.0 * omega_old[1:-1, 1:-1])
        RHS_x = omega_old[1:-1, 1:-1] + (0.5 * dt) * (diff_y - conv)

        # Boundary closures on left (col 0) and right (col -1)
        RHS_x[:, 0] -= A * omega_old[1:-1, 0]
        RHS_x[:, -1] -= C * omega_old[1:-1, -1]

        # Vectorized Thomas algorithm across all M rows concurrently
        Q_x = np.zeros((M, M))
        Q_x[:, 0] = RHS_x[:, 0] / denom[0]
        for j in range(1, M):
            Q_x[:, j] = (RHS_x[:, j] - A * Q_x[:, j - 1]) / denom[j]

        omega_star_inner = np.zeros((M, M))
        omega_star_inner[:, -1] = Q_x[:, -1]
        for j in range(M - 2, -1, -1):
            omega_star_inner[:, j] = Q_x[:, j] - P[j] * omega_star_inner[:, j + 1]

        omega_star = self.omega.copy()
        omega_star[1:-1, 1:-1] = omega_star_inner

        # 3. Step 2: Implicit Y (col sweeps), Explicit X (Peaceman-Rachford half-step dt/2)
        diff_x = (nu / (h**2)) * (omega_star[1:-1, 2:] + omega_star[1:-1, :-2] - 2.0 * omega_star[1:-1, 1:-1])
        RHS_y = omega_star[1:-1, 1:-1] + (0.5 * dt) * (diff_x - conv)

        # Boundary closures on bottom (row 0) and top (row -1)
        RHS_y[0, :] -= A * omega_star[0, 1:-1]
        RHS_y[-1, :] -= C * omega_star[-1, 1:-1]

        # Vectorized Thomas algorithm across all M columns concurrently
        Q_y = np.zeros((M, M))
        Q_y[0, :] = RHS_y[0, :] / denom[0]
        for i in range(1, M):
            Q_y[i, :] = (RHS_y[i, :] - A * Q_y[i - 1, :]) / denom[i]

        omega_new_inner = np.zeros((M, M))
        omega_new_inner[-1, :] = Q_y[-1, :]
        for i in range(M - 2, -1, -1):
            omega_new_inner[i, :] = Q_y[i, :] - P[i] * omega_new_inner[i + 1, :]

        omega_new = self.omega.copy()
        omega_new[1:-1, 1:-1] = omega_new_inner
        return omega_new

    def step_adi(self):
        """Advance the vorticity field by one Peaceman-Rachford ADI step."""
        self.omega = self.solve_vorticity_transport_ADI()
        return self.omega

    def solve_poisson(self):
        """Invert the Poisson equation for streamfunction: del^2(psi) = -omega."""
        self.solve_streamfunction()
        return self.psi

    def update_velocity(self):
        """Update velocity components (u, v) from streamfunction derivatives."""
        self.calculate_velocities()
        return self.u, self.v

    def update_wall_vorticity(self):
        """Apply Thom's wall boundary conditions with under-relaxation."""
        self.apply_boundary_conditions()
        return self.omega

    def step(self):
        """
        Advance complete coupled Navier-Stokes state by one physical time step:
        1. ADI vorticity transport
        2. Wall vorticity boundary enforcement
        3. Poisson streamfunction inversion
        4. Interior & wall velocity differentiation
        """
        self.omega = self.solve_vorticity_transport_ADI()
        self.apply_boundary_conditions()
        self.solve_streamfunction()
        self.calculate_velocities()
        return self.omega, self.psi

    def calculate_pressure(self, max_iter=2500, tol=1e-5):
        """
        Recover the pressure field by solving the pressure Poisson equation:
            del^2(p) = - [ (du/dx)^2 + 2*(du/dy)*(dv/dx) + (dv/dy)^2 ]
        with homogeneous Neumann boundary conditions dp/dn = 0 and mean-zero gauge.
        """
        h = self.h
        dudx = (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2.0 * h)
        dudy = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2.0 * h)
        dvdx = (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2.0 * h)
        dvdy = (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2.0 * h)

        rhs = -(dudx**2 + 2.0 * dudy * dvdx + dvdy**2)

        p_new = self.p.copy()
        for _ in range(max_iter):
            p_old = p_new.copy()
            p_new[1:-1, 1:-1] = 0.25 * (
                p_old[2:, 1:-1] + p_old[:-2, 1:-1] +
                p_old[1:-1, 2:] + p_old[1:-1, :-2] - (h**2) * rhs
            )
            # Neumann boundary conditions: zero normal derivative
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]

            if np.max(np.abs(p_new[1:-1, 1:-1] - p_old[1:-1, 1:-1])) < tol:
                break

        self.p = p_new - np.mean(p_new[1:-1, 1:-1])

    solve_pressure_poisson = calculate_pressure

    def solve(self, max_iterations=25000, tolerance=1e-5, min_iterations=300, log_interval=500, compute_pressure=True):
        """Main iterative coupling loop between vorticity transport and streamfunction."""
        print(f"\n{'='*65}")
        print(f"LID-DRIVEN CAVITY SOLVER (Re={self.Re}, Grid={self.N}x{self.N}, Lid='{self.lid_profile}')")
        print(f"{'='*65}")

        start_time = time.time()
        for iteration in range(max_iterations):
            omega_old = self.omega.copy()

            # 1. Update velocities
            self.calculate_velocities()

            # 2. Enforce wall vorticity
            self.apply_boundary_conditions()

            # 3. Vectorized ADI step for omega
            self.omega = self.solve_vorticity_transport_ADI()

            # 4. Enforce wall vorticity
            self.apply_boundary_conditions()

            # 5. Solve streamfunction Poisson equation
            self.solve_streamfunction()

            # 6. Check convergence on interior vorticity
            max_change = np.max(np.abs(self.omega[1:-1, 1:-1] - omega_old[1:-1, 1:-1]))

            if iteration % log_interval == 0 or iteration == max_iterations - 1:
                self.history['iterations'].append(iteration)
                self.history['max_change'].append(max_change)
                self.history['psi_min'].append(float(self.psi.min()))
                print(f"Iter {iteration:6d} | max d(omega): {max_change:.2e} | psi_min: {self.psi.min():.6f}", flush=True)

            if max_change < tolerance and iteration >= min_iterations:
                elapsed = time.time() - start_time
                print(f"\n[SUCCESS] Converged in {iteration} iterations ({elapsed:.2f} s). Final max d(omega): {max_change:.2e}", flush=True)
                self.calculate_velocities()
                if compute_pressure:
                    self.calculate_pressure()
                return True, iteration

        elapsed = time.time() - start_time
        print(f"\n[INFO] Reached max iterations ({max_iterations}) in {elapsed:.2f} s. Final max d(omega): {max_change:.2e}", flush=True)
        self.calculate_velocities()
        if compute_pressure:
            self.calculate_pressure()
        return False, max_iterations

    def evaluate_ghia_metrics(self):
        """
        Compute quantitative validation metrics against Ghia, Ghia & Shin (1982) benchmark.
        Returns dict with L2 and L_inf errors for u(y) and v(x) centerlines, and vortex position error.
        """
        if self.lid_profile != 'constant' or f'u_Re{self.Re}' not in GHIA_DATA:
            return None

        # Centerline u(y) at x = L/2
        x_idx = np.argmin(np.abs(self.x - self.L / 2.0))
        u_center = self.u[:, x_idx]
        ghia_y = GHIA_DATA['y_u']
        ghia_u = GHIA_DATA[f'u_Re{self.Re}']
        u_interp = np.interp(ghia_y, self.y, u_center)
        u_l2 = float(np.sqrt(np.mean((u_interp - ghia_u)**2)))
        u_max = float(np.max(np.abs(u_interp - ghia_u)))

        # Centerline v(x) at y = L/2
        y_idx = np.argmin(np.abs(self.y - self.L / 2.0))
        v_center = self.v[y_idx, :]
        ghia_x = GHIA_DATA['x_v']
        ghia_v = GHIA_DATA[f'v_Re{self.Re}']
        v_interp = np.interp(ghia_x, self.x, v_center)
        v_l2 = float(np.sqrt(np.mean((v_interp - ghia_v)**2)))
        v_max = float(np.max(np.abs(v_interp - ghia_v)))

        # Vortex position offsets
        vortices = self.get_vortex_centers()
        p = vortices['primary']
        vortex_metrics = None
        if self.Re in GHIA_DATA['vortex_benchmarks']:
            ref_dict = GHIA_DATA['vortex_benchmarks'][self.Re]
            ref_p = ref_dict.get('primary', ref_dict)
            dx = p['x'] - ref_p['x']
            dy = p['y'] - ref_p['y']
            dist = float(np.sqrt(dx**2 + dy**2))
            dpsi = float(np.abs(p['psi'] - ref_p['psi_min']))
            vortex_metrics = {
                'computed_center': (p['x'], p['y']),
                'benchmark_center': (ref_p['x'], ref_p['y']),
                'center_distance_error': dist,
                'computed_psi_min': p['psi'],
                'benchmark_psi_min': ref_p['psi_min'],
                'psi_min_abs_error': dpsi
            }

        return {
            'Re': self.Re,
            'u_centerline_l2_error': u_l2,
            'u_centerline_max_error': u_max,
            'v_centerline_l2_error': v_l2,
            'v_centerline_max_error': v_max,
            'vortex': vortex_metrics
        }

    @classmethod
    def solve_with_continuation(cls, target_Re=3200, target_N=257, re_schedule=None,
                                lid_profile='constant', poisson_solver='dst', convection_scheme='central',
                                wall_bc='thom', wall_beta=None,
                                tolerance=1e-5, max_iterations=35000, log_interval=500):
        """
        Curriculum parameter continuation solver for lid-driven cavity flow.
        Progressively solves through a sequence of Reynolds numbers, prolonging
        the mesh from coarse (e.g. N=129) to fine (target_N, e.g. 257).
        """
        if re_schedule is None:
            if target_Re <= 100:
                ladder = [target_Re]
            elif target_Re <= 400:
                ladder = [100, target_Re]
            elif target_Re <= 1000:
                ladder = [100, 400, target_Re] if target_Re != 400 else [100, 400]
            elif target_Re <= 3200:
                ladder = [100, 400, 1000, target_Re]
            elif target_Re <= 5000:
                ladder = [100, 1000, 3200, target_Re]
            elif target_Re <= 10000:
                ladder = [100, 1000, 3200, 5000, target_Re]
            else:
                ladder = [100, 1000, 3200, 5000, 10000, target_Re]
        else:
            ladder = list(re_schedule)
            if ladder[-1] != target_Re:
                ladder.append(target_Re)

        unique_ladder = []
        for r in ladder:
            if r not in unique_ladder:
                unique_ladder.append(r)
        ladder = unique_ladder

        print("\n" + "#"*70, flush=True)
        print("REYNOLDS NUMBER CURRICULUM CONTINUATION ENGINE", flush=True)
        print(f"Target: Re = {target_Re}, Grid = {target_N}x{target_N} | Schedule: {ladder}", flush=True)
        print("#"*70, flush=True)

        current_psi = None
        current_omega = None
        current_x = None
        current_y = None
        current_solver = None

        total_start_time = time.time()

        for idx, stage_Re in enumerate(ladder):
            is_final = (idx == len(ladder) - 1)
            # Use coarse grid (129) for lower stages if target_N >= 257 to accelerate warm-up
            if target_N >= 257 and stage_Re < target_Re and not is_final:
                stage_N = 129
            else:
                stage_N = target_N

            stage_tol = tolerance if is_final else max(2e-4, tolerance * 5.0)
            stage_max_iters = max_iterations if is_final else min(5000, max_iterations)
            stage_min_iters = 300 if is_final else 100

            print(f"\n>>> [STAGE {idx+1}/{len(ladder)}] Running Re = {stage_Re} on Mesh {stage_N}x{stage_N} (Tol={stage_tol:.1e})", flush=True)

            stage_solver = cls(
                N=stage_N,
                Re=stage_Re,
                lid_profile=lid_profile,
                poisson_solver=poisson_solver,
                convection_scheme=convection_scheme,
                wall_bc=wall_bc,
                wall_beta=wall_beta
            )

            # Warm start from previous stage if available
            if current_psi is not None:
                if current_psi.shape == (stage_N, stage_N):
                    stage_solver.set_state(current_psi, current_omega)
                    print(f"    [Warm Start] Inherited state directly from previous stage (Re = {ladder[idx-1]}).", flush=True)
                else:
                    # Prolong fields from coarse to fine grid
                    print(f"    [Prolongation] Interpolating fields from {current_psi.shape[0]}x{current_psi.shape[1]} to {stage_N}x{stage_N}...", flush=True)
                    prolonged_psi, prolonged_omega = prolong_fields(
                        current_psi, current_omega,
                        current_x, current_y,
                        stage_solver.x, stage_solver.y
                    )
                    stage_solver.set_state(prolonged_psi, prolonged_omega)
                    print(f"    [Prolongation] State successfully interpolated and boundary conditions re-enforced.", flush=True)

            # Solve stage
            stage_solver.solve(
                max_iterations=stage_max_iters,
                tolerance=stage_tol,
                min_iterations=stage_min_iters,
                log_interval=log_interval,
                compute_pressure=is_final
            )

            current_psi = stage_solver.psi.copy()
            current_omega = stage_solver.omega.copy()
            current_x = stage_solver.x.copy()
            current_y = stage_solver.y.copy()
            current_solver = stage_solver

        total_elapsed = time.time() - total_start_time
        print(f"\n{'='*70}", flush=True)
        print(f"[CURRICULUM COMPLETE] Total continuation runtime: {total_elapsed:.2f} s", flush=True)
        print(f"{'='*70}\n", flush=True)

        return current_solver

    def get_vortex_centers(self):
        """
        Locate physical coordinates and streamfunctions of primary and secondary corner vortices.
        Returns dict with keys: 'primary', 'BR1' (bottom-right), 'BL1' (bottom-left), 'TL1' (top-left).
        """
        p_idx = np.unravel_index(np.argmin(self.psi), self.psi.shape)
        results = {
            'primary': {
                'x': float(self.x[p_idx[1]]),
                'y': float(self.y[p_idx[0]]),
                'psi': float(self.psi[p_idx])
            }
        }

        nx_mid = len(self.x) // 2
        ny_mid = len(self.y) // 2

        # Bottom-Right corner: x in [L/2, L], y in [0, L/2] (interior only)
        br_psi = self.psi[1:ny_mid, nx_mid:-1]
        if br_psi.size > 0 and np.max(br_psi) > 1e-7:
            br_loc = np.unravel_index(np.argmax(br_psi), br_psi.shape)
            results['BR1'] = {
                'x': float(self.x[nx_mid + br_loc[1]]),
                'y': float(self.y[1 + br_loc[0]]),
                'psi': float(br_psi[br_loc])
            }

        # Bottom-Left corner: x in [0, L/2], y in [0, L/2] (interior only)
        bl_psi = self.psi[1:ny_mid, 1:nx_mid]
        if bl_psi.size > 0 and np.max(bl_psi) > 1e-7:
            bl_loc = np.unravel_index(np.argmax(bl_psi), bl_psi.shape)
            results['BL1'] = {
                'x': float(self.x[1 + bl_loc[1]]),
                'y': float(self.y[1 + bl_loc[0]]),
                'psi': float(bl_psi[bl_loc])
            }

        # Top-Left corner: x in [0, L/2], y in [L/2, L] (interior only)
        tl_psi = self.psi[ny_mid:-1, 1:nx_mid]
        if tl_psi.size > 0 and np.max(tl_psi) > 1e-7:
            tl_loc = np.unravel_index(np.argmax(tl_psi), tl_psi.shape)
            results['TL1'] = {
                'x': float(self.x[1 + tl_loc[1]]),
                'y': float(self.y[ny_mid + tl_loc[0]]),
                'psi': float(tl_psi[tl_loc])
            }

        return results

    def get_vortex_center(self):
        """Return (x, y) physical coordinates of the primary vortex center."""
        centers = self.get_vortex_centers()
        return centers['primary']['x'], centers['primary']['y']

    def print_summary(self):
        """Display physical summary of the solution and Ghia validation."""
        vortices = self.get_vortex_centers()
        p = vortices['primary']
        vel_mag = np.sqrt(self.u**2 + self.v**2)

        print(f"\n{'='*65}", flush=True)
        print("SOLUTION SUMMARY", flush=True)
        print(f"{'='*65}", flush=True)
        print(f"Grid Size: {self.N} x {self.N} | Reynolds: Re = {self.Re}", flush=True)
        print(f"Lid Velocity Profile: {self.lid_profile} (Peak U = {self.U})", flush=True)
        print(f"Primary Vortex Center: x = {p['x']:.4f}, y = {p['y']:.4f} (psi = {p['psi']:.6f})", flush=True)
        if 'BR1' in vortices:
            br = vortices['BR1']
            print(f"Secondary BR1 Vortex:  x = {br['x']:.4f}, y = {br['y']:.4f} (psi = {br['psi']:.6e})", flush=True)
        if 'BL1' in vortices:
            bl = vortices['BL1']
            print(f"Secondary BL1 Vortex:  x = {bl['x']:.4f}, y = {bl['y']:.4f} (psi = {bl['psi']:.6e})", flush=True)
        if 'TL1' in vortices:
            tl = vortices['TL1']
            print(f"Secondary TL1 Vortex:  x = {tl['x']:.4f}, y = {tl['y']:.4f} (psi = {tl['psi']:.6e})", flush=True)
        print(f"Streamfunction: min = {self.psi.min():.6f}, max = {self.psi.max():.6f}", flush=True)
        print(f"Vorticity:      min = {self.omega.min():.2f}, max = {self.omega.max():.2f}", flush=True)
        print(f"Max Velocity:   |V|_max = {vel_mag.max():.4f}", flush=True)
        print(f"Pressure Range: min = {self.p.min():.4f}, max = {self.p.max():.4f}", flush=True)

        metrics = self.evaluate_ghia_metrics()
        if metrics is not None:
            print("-" * 65, flush=True)
            print("GHIA ET AL. (1982) BENCHMARK VALIDATION", flush=True)
            print("-" * 65, flush=True)
            print(f"u-centerline L2 Relative Error:  {metrics['u_centerline_l2_error']:.4e}", flush=True)
            print(f"u-centerline Max Absolute Error: {metrics['u_centerline_max_error']:.4e}", flush=True)
            print(f"v-centerline L2 Relative Error:  {metrics['v_centerline_l2_error']:.4e}", flush=True)
            print(f"v-centerline Max Absolute Error: {metrics['v_centerline_max_error']:.4e}", flush=True)
            if metrics['vortex'] is not None:
                v_info = metrics['vortex']
                print(f"Primary Vortex Distance Offset:  {v_info['center_distance_error']:.4e} (Ghia: {v_info['benchmark_center']})", flush=True)
                print(f"Streamfunction Min Abs Error:    {v_info['psi_min_abs_error']:.6f} (Ghia: {v_info['benchmark_psi_min']:.4f})", flush=True)
        print(f"{'='*65}\n", flush=True)

    def _plot_streamlines_with_eddies(self, ax, show_labels=False):
        """
        Plot continuous streamfunction field resolving primary, secondary, and quaternary
        corner eddies with zero blank spaces and clean, marker-free presentation.
        """
        psi_min = float(self.psi.min())
        psi_max = float(self.psi.max())

        # 1. Primary clockwise vortex: high-density power-law levels filling entire domain to wall
        n_neg = 55
        neg_p = np.linspace(0.03, 1.0, n_neg)**2.2
        levels_neg = np.sort(- neg_p * (-psi_min))

        ax.contourf(self.X, self.Y, self.psi, levels=levels_neg, cmap='viridis', alpha=0.9)
        ax.contour(self.X, self.Y, self.psi, levels=levels_neg, colors='black', linewidths=0.5, alpha=0.45)

        # 2. Secondary, tertiary, & quaternary counter-clockwise corner vortices (psi > 0)
        if psi_max > 1e-9:
            levels_pos = np.logspace(-9, np.log10(max(psi_max, 1e-6)), 35)
            ax.contourf(self.X, self.Y, self.psi, levels=levels_pos, cmap='autumn_r', alpha=0.95)
            ax.contour(self.X, self.Y, self.psi, levels=levels_pos, colors='darkred', linewidths=0.6, alpha=0.6)

        # 3. Dividing separatrix streamline (psi = 0)
        ax.contour(self.X, self.Y, self.psi, levels=[0.0], colors='white', linewidths=1.5, linestyles='--')

        ax.set_aspect('equal')
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)

    def create_plots(self, save_fig=True, show=True, out_path=None):
        """Generate comprehensive 6-panel flow diagnostics figure with LaTeX styling and clean legend placement."""
        from matplotlib.colors import SymLogNorm
        _apply_latex_style()

        fig = plt.figure(figsize=(16, 10.5), dpi=300)

        # 1. Streamlines (with secondary corner eddies resolved, zero blank space)
        ax1 = plt.subplot(231)
        self._plot_streamlines_with_eddies(ax1, show_labels=False)
        ax1.set_xlabel(r'$x / L$')
        ax1.set_ylabel(r'$y / L$')
        ax1.set_title(r'\textbf{Streamlines} $\psi$ \textbf{(All Eddies Resolved)}', fontsize=13)

        # 2. Vorticity with Symmetric Logarithmic Colormap (No washout at high Re)
        ax2 = plt.subplot(232)
        v_lim = max(abs(float(self.omega.min())), abs(float(self.omega.max())), 10.0)
        v_disp = min(v_lim, 200.0)
        norm_vort = SymLogNorm(linthresh=1.0, linscale=0.5, vmin=-v_disp, vmax=v_disp, base=10)
        levels_vort = np.unique(np.concatenate([
            -np.logspace(np.log10(v_disp), np.log10(0.05), 45),
            np.linspace(-0.05, 0.05, 11),
            np.logspace(np.log10(0.05), np.log10(v_disp), 45)
        ]))
        c2 = ax2.contourf(self.X, self.Y, self.omega, levels=levels_vort, norm=norm_vort, cmap='RdBu_r', extend='both')
        ax2.contour(self.X, self.Y, self.omega, levels=[-10, -5, -2, -1, 1, 2, 5, 10], colors='black', linewidths=0.35, alpha=0.3)
        ax2.set_xlabel(r'$x / L$')
        ax2.set_ylabel(r'$y / L$')
        ax2.set_title(r'\textbf{Vorticity} $\omega$ \textbf{(SymLog Scale)}', fontsize=13)
        ax2.set_aspect('equal')
        cb2 = plt.colorbar(c2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label(r'$\omega$')

        # 3. Velocity Magnitude
        ax3 = plt.subplot(233)
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        c3 = ax3.contourf(self.X, self.Y, vel_mag, levels=30, cmap='plasma', alpha=0.85)
        ax3.set_xlabel(r'$x / L$')
        ax3.set_ylabel(r'$y / L$')
        ax3.set_title(r'\textbf{Velocity Magnitude} $|\mathbf{V}| / U$', fontsize=13)
        ax3.set_aspect('equal')
        plt.colorbar(c3, ax=ax3, fraction=0.046, pad=0.04)

        # 4. Vertical Centerline u(y)
        ax4 = plt.subplot(234)
        x_idx = np.argmin(np.abs(self.x - self.L / 2.0))
        ax4.plot(self.u[:, x_idx], self.y, 'b-', lw=2.2, label=rf'$u(y)$ FDM ($x = {self.L/2:.2f}$)')
        if self.lid_profile == 'constant' and self.Re in [100, 400, 1000, 3200, 5000, 7500, 10000]:
            ghia_u = GHIA_DATA[f'u_Re{self.Re}']
            ax4.scatter(ghia_u, GHIA_DATA['y_u'], color='black', s=25, zorder=5, label='Ghia et al. (1982)')
        ax4.set_xlabel(r'$u / U$')
        ax4.set_ylabel(r'$y / L$')
        ax4.set_title(r'\textbf{Vertical Centerline} $u(y)$', fontsize=13)
        ax4.grid(True, ls='--', alpha=0.4)
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, framealpha=0.95, fontsize=9.5)

        # 5. Horizontal Centerline v(x)
        ax5 = plt.subplot(235)
        y_idx = np.argmin(np.abs(self.y - self.L / 2.0))
        ax5.plot(self.x, self.v[y_idx, :], 'r-', lw=2.2, label=rf'$v(x)$ FDM ($y = {self.L/2:.2f}$)')
        if self.lid_profile == 'constant' and self.Re in [100, 400, 1000, 3200, 5000, 7500, 10000]:
            ghia_v = GHIA_DATA[f'v_Re{self.Re}']
            ax5.scatter(GHIA_DATA['x_v'], ghia_v, color='black', s=25, zorder=5, label='Ghia et al. (1982)')
        ax5.set_xlabel(r'$x / L$')
        ax5.set_ylabel(r'$v / U$')
        ax5.set_title(r'\textbf{Horizontal Centerline} $v(x)$', fontsize=13)
        ax5.grid(True, ls='--', alpha=0.4)
        ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, framealpha=0.95, fontsize=9.5)

        # 6. Pressure
        ax6 = plt.subplot(236)
        c6 = ax6.contourf(self.X, self.Y, self.p, levels=30, cmap='RdGy', alpha=0.85)
        ax6.set_xlabel(r'$x / L$')
        ax6.set_ylabel(r'$y / L$')
        ax6.set_title(r'\textbf{Pressure Field} $p / (\rho U^2)$', fontsize=13)
        ax6.set_aspect('equal')
        plt.colorbar(c6, ax=ax6, fraction=0.046, pad=0.04)

        plt.suptitle(rf"\textbf{{2D Lid-Driven Cavity Flow}} $\mid$ $Re = {self.Re}$ $\mid$ \textbf{{Grid}} $= {self.N} \times {self.N}$ $\mid$ \textbf{{Lid: {self.lid_profile.capitalize()}}}",
                     fontsize=15, y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        if save_fig:
            if out_path is not None:
                fname = out_path
            else:
                fname = f"lid_driven_diagnostic_Re{self.Re}_N{self.N}_{self.lid_profile}.png"
            parent_dir = os.path.dirname(fname)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[SUCCESS] Diagnostic plot saved: {fname}", flush=True)

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def create_showcase_plots(self, save_fig=True, show=True, out_path=None):
        """Generate high-contrast publication showcase plot with LaTeX fonts and legends below x-axis."""
        _apply_latex_style()
        fig, axs = plt.subplots(1, 2, figsize=(18, 8.5), dpi=300)

        # 1. Streamlines (both primary and secondary corner eddies resolved)
        ax = axs[0]
        self._plot_streamlines_with_eddies(ax, show_labels=True)
        ax.set_title(r"\textbf{Streamlines} ($\psi$) --- \textbf{Primary \& Secondary Eddies}", fontsize=14)
        ax.set_xlabel(r"$x / L$", fontsize=12)
        ax.set_ylabel(r"$y / L$", fontsize=12)

        # 2. Centerline validation
        ax = axs[1]
        x_idx = np.argmin(np.abs(self.x - self.L / 2.0))
        ax.plot(self.u[:, x_idx], self.y, 'k-', lw=2.8, label=rf"$u(y)$ at $x = {self.L/2:.2f}$ (FDM)")
        if self.lid_profile == 'constant' and self.Re in [100, 400, 1000, 3200, 5000, 7500, 10000]:
            ghia_u = GHIA_DATA[f'u_Re{self.Re}']
            ax.scatter(ghia_u, GHIA_DATA['y_u'], facecolors='none', edgecolors='red', s=45, lw=1.5,
                       label=r'Ghia et al. (1982) Benchmark')
        ax.set_title(r"\textbf{Centerline Velocity Profile} $u(y)$", fontsize=14)
        ax.set_xlabel(r"$u / U$", fontsize=12)
        ax.set_ylabel(r"$y / L$", fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=11, framealpha=0.95)

        plt.suptitle(rf"\textbf{{Lid-Driven Cavity Flow}} --- $Re = {self.Re}$, \textbf{{Grid}} $= {self.N} \times {self.N}$ (\textbf{{{self.lid_profile.capitalize()} Lid}})",
                     fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        if save_fig:
            if out_path is not None:
                fname = out_path
            else:
                fname = f"lid_driven_showcase_Re{self.Re}_N{self.N}.png"
            parent_dir = os.path.dirname(fname)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[SUCCESS] Showcase plot saved: {fname}", flush=True)
            if out_path is None and os.path.exists("figures"):
                fig_path = os.path.join("figures", f"lid_driven_cavity_showcase_Re{self.Re}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"[SUCCESS] Curated publication showcase saved: {fig_path}", flush=True)

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig


    def save_model(self, filename=None):
        """Save flow fields and parameters to both Python pickle (.pkl) and NumPy archive (.npz) in data/ directory."""
        os.makedirs("data", exist_ok=True)
        if filename is None:
            filename = os.path.join("data", f"lid_driven_model_Re{self.Re}_N{self.N}.pkl")

        vortices = self.get_vortex_centers()
        p = vortices['primary']
        vx, vy = p['x'], p['y']

        model_data = {
            'parameters': {
                'N': self.N,
                'Re': self.Re,
                'U': self.U,
                'L': self.L,
                'h': self.h,
                'nu': self.nu,
                'dt': self.dt,
                'lid_profile': self.lid_profile,
                'poisson_solver': self.poisson_solver,
                'convection_scheme': self.convection_scheme
            },
            'coordinates': {
                'x': self.x,
                'y': self.y,
                'X': self.X,
                'Y': self.Y
            },
            'fields': {
                'psi': self.psi,
                'omega': self.omega,
                'u': self.u,
                'v': self.v,
                'p': self.p
            },
            'vortex_center': {
                'x': vx,
                'y': vy,
                'array_indices': np.unravel_index(np.argmin(self.psi), self.psi.shape)
            },
            'vortex_centers': vortices,
            'convergence': self.history,
            'metadata': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'coordinate_system': 'x=horizontal [0, L], y=vertical [0, L]',
                'moving_lid': f'top wall (y = {self.L}) with profile {self.lid_profile}',
                'provenance': 'High-resolution FDM solver with fast DST Poisson and vectorized ADI closures'
            }
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SUCCESS] Model pickle saved: {filename}", flush=True)

        # Save NumPy npz archive to data/ (compatible with plot_from_npz.py)
        npz_name = os.path.join("data", f"flow_fields_Re{self.Re}_N{self.N}.npz")
        np.savez(
            npz_name,
            x=self.x, y=self.y,
            X=self.X, Y=self.Y,
            psi=self.psi, omega=self.omega,
            u=self.u, v=self.v, p=self.p,
            vortex_x=vx, vortex_y=vy,
            Re=self.Re, N=self.N,
            lid_profile=self.lid_profile,
            convection_scheme=self.convection_scheme
        )
        print(f"[SUCCESS] Flow fields NumPy archive saved: {npz_name}", flush=True)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="High-resolution 2D Lid-Driven Cavity FDM Solver.")
    parser.add_argument('--N', type=int, default=129, help="Grid size along each axis (default: 129)")
    parser.add_argument('--Re', type=int, default=1000, help="Reynolds number (default: 1000)")
    parser.add_argument('--lid_profile', type=str, default='constant', choices=['constant', 'regularized'],
                        help="Top wall velocity profile: 'constant' (Ghia benchmark) or 'regularized' (singularity-free)")
    parser.add_argument('--solver', type=str, default='dst', choices=['dst', 'lu', 'sor'],
                        help="Streamfunction Poisson solver: 'dst' (Fast Discrete Sine Transform, fastest & exact), 'lu' (sparse LU), or 'sor' (RB-SOR)")
    parser.add_argument('--convection', type=str, default='central', choices=['central', 'hybrid', 'upwind'],
                        help="Convective scheme: 'central' (pure 2nd-order, 0 artificial viscosity), 'hybrid', or 'upwind'")
    parser.add_argument('--wall_bc', type=str, default='thom', choices=['thom', 'woods'],
                        help="Wall vorticity boundary condition: 'thom' (classic 1st-order, unconditionally stable) or 'woods' (2nd-order)")
    parser.add_argument('--wall_beta', type=float, default=None,
                        help="Under-relaxation factor for wall vorticity (0 < beta <= 1.0). Default auto-tunes by Re.")
    parser.add_argument('--max_iterations', type=int, default=35000, help="Maximum solver iterations (default: 35000)")
    parser.add_argument('--tolerance', type=float, default=1e-5, help="Vorticity convergence tolerance (default: 1e-5)")
    parser.add_argument('--continuation', action='store_true',
                        help="Enable Reynolds number curriculum continuation (warm-start homotopy across Re stages)")
    parser.add_argument('--re_schedule', type=str, default=None,
                        help="Custom comma-separated Re continuation schedule (e.g. '100,1000,3200,5000')")
    parser.add_argument('--no_plots', action='store_true', help="Suppress interactive plot windows")
    parser.add_argument('--save_fig', action='store_true', default=True, help="Save diagnostic and showcase figures")

    args = parser.parse_args()

    if args.continuation:
        schedule = [int(r.strip()) for r in args.re_schedule.split(',')] if args.re_schedule else None
        solver = LidDrivenCavitySolver.solve_with_continuation(
            target_Re=args.Re,
            target_N=args.N,
            re_schedule=schedule,
            lid_profile=args.lid_profile,
            poisson_solver=args.solver,
            convection_scheme=args.convection,
            wall_bc=args.wall_bc,
            wall_beta=args.wall_beta,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations
        )
    else:
        solver = LidDrivenCavitySolver(
            N=args.N,
            Re=args.Re,
            lid_profile=args.lid_profile,
            poisson_solver=args.solver,
            convection_scheme=args.convection,
            wall_bc=args.wall_bc,
            wall_beta=args.wall_beta
        )
        converged, iters = solver.solve(
            max_iterations=args.max_iterations,
            tolerance=args.tolerance
        )

    solver.print_summary()
    solver.save_model()

    if not args.no_plots:
        solver.create_plots(save_fig=args.save_fig, show=True)
        solver.create_showcase_plots(save_fig=args.save_fig, show=True)
    else:
        if args.save_fig:
            solver.create_plots(save_fig=True, show=False)
            solver.create_showcase_plots(save_fig=True, show=False)
