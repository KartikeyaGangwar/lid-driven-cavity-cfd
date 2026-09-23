"""
Unit test for restart continuity in unsteady_cavity_solver.py.
Verifies that interrupting a simulation at t_mid and resuming produces
identical flow fields and continuous telemetry matching an uninterrupted run
to machine / numerical precision.
"""

import os
import sys
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unsteady_cavity_solver import UnsteadyCavitySolver


def test_restart_continuity():
    temp_dir = tempfile.mkdtemp()
    try:
        N = 65
        Re = 1000
        dt = 1.0e-3
        t_mid = 0.05
        t_end = 0.10

        # 1. Uninterrupted Reference Run
        sim_ref = UnsteadyCavitySolver(N=N, Re=Re, dt=dt, convection_scheme='central', wall_bc='thom', wall_beta=1.0)
        # Initialize with arbitrary smooth field
        x = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, x, indexing='ij')
        sim_ref.solver.psi = np.sin(np.pi * X) * np.sin(np.pi * Y) * 0.01
        h = sim_ref.solver.h
        psi = sim_ref.solver.psi
        lap = (np.roll(psi, 1, 0) + np.roll(psi, -1, 0) + np.roll(psi, 1, 1) + np.roll(psi, -1, 1) - 4 * psi) / (h**2)
        sim_ref.solver.omega = -lap
        sim_ref.solver.apply_boundary_conditions()
        sim_ref.solver.calculate_velocities()

        # Save initial state so resumed run starts from exact same initial condition
        init_state_path = os.path.join(temp_dir, 'init_state.npz')
        np.savez_compressed(init_state_path, omega=sim_ref.solver.omega.copy(), psi=sim_ref.solver.psi.copy(), Re=Re, N=N)

        telemetry_ref = sim_ref.run_simulation(t_end=t_end, sample_interval=5, t_start=0.0)
        ref_final_omega = sim_ref.solver.omega.copy()
        ref_final_psi = sim_ref.solver.psi.copy()

        # 2. Segment 1: Run to t_mid
        sim_seg1 = UnsteadyCavitySolver(N=N, Re=Re, dt=dt, convection_scheme='central', wall_bc='thom', wall_beta=1.0)
        d_init = np.load(init_state_path)
        sim_seg1.solver.omega = d_init['omega'].copy()
        sim_seg1.solver.psi = d_init['psi'].copy()
        sim_seg1.solver.calculate_velocities()

        telemetry_seg1 = sim_seg1.run_simulation(t_end=t_mid, sample_interval=5, t_start=0.0)

        # Call diagnostic snapshots (must NOT mutate solver)
        fft_dummy = sim_seg1.analyze_frequency_and_attractor(probe_key='BL')
        snapshots = sim_seg1.capture_cycle_snapshots(period=0.02, n_phases=4)

        # Save checkpoint as done in production
        mid_npz = os.path.join(temp_dir, 'unsteady_mid.npz')
        np.savez_compressed(
            mid_npz,
            time=telemetry_seg1['time'],
            u_BL=telemetry_seg1['u_probes']['BL'],
            v_BL=telemetry_seg1['v_probes']['BL'],
            u_TR=telemetry_seg1['u_probes']['TR'],
            v_TR=telemetry_seg1['v_probes']['TR'],
            kinetic_energy=telemetry_seg1['kinetic_energy'],
            enstrophy=telemetry_seg1['enstrophy'],
            freqs=fft_dummy['freqs'],
            psd=fft_dummy['psd'],
            f_dom=fft_dummy['f_dom'],
            strouhal=fft_dummy['strouhal'],
            period=fft_dummy['period'],
            final_omega=sim_seg1.solver.omega,
            final_psi=sim_seg1.solver.psi,
            Re=Re,
            N=N
        )

        # 3. Segment 2: Resume from mid_npz to t_end
        sim_seg2 = UnsteadyCavitySolver(N=N, Re=Re, dt=dt, convection_scheme='central', wall_bc='thom', wall_beta=1.0)
        t_resumed, init_telem = sim_seg2.resume_from_unsteady_npz(mid_npz)
        assert abs(t_resumed - t_mid) < 1e-12, f"Expected t_resumed == t_mid, got {t_resumed}"

        telemetry_seg2 = sim_seg2.run_simulation(t_end=t_end, sample_interval=5, t_start=t_resumed, initial_telemetry=init_telem)
        resumed_final_omega = sim_seg2.solver.omega
        resumed_final_psi = sim_seg2.solver.psi

        # Assert final fields match to machine precision
        diff_omega = np.max(np.abs(ref_final_omega - resumed_final_omega))
        diff_psi = np.max(np.abs(ref_final_psi - resumed_final_psi))
        print(f"Max difference in final omega: {diff_omega:.4e}")
        print(f"Max difference in final psi: {diff_psi:.4e}")
        assert diff_omega < 1e-12, f"Restart diverged in omega: max diff = {diff_omega}"
        assert diff_psi < 1e-12, f"Restart diverged in psi: max diff = {diff_psi}"

        # Assert telemetry time and velocity continuity
        t_ref = np.array(telemetry_ref['time'])
        t_res = np.array(telemetry_seg2['time'])
        assert np.allclose(t_ref, t_res, atol=1e-12), "Telemetry times mismatch"

        u_ref = np.array(telemetry_ref['u_probes']['BL'])
        u_res = np.array(telemetry_seg2['u_probes']['BL'])
        assert np.allclose(u_ref, u_res, atol=1e-12), "Telemetry u_BL mismatch"

        ke_ref = np.array(telemetry_ref['kinetic_energy'])
        ke_res = np.array(telemetry_seg2['kinetic_energy'])
        assert np.allclose(ke_ref, ke_res, atol=1e-12), "Telemetry KE mismatch"

        # Check there is no step jump at concatenation index
        mid_idx = np.argmin(np.abs(t_res - t_mid))
        step_jump_u = abs(u_res[mid_idx + 1] - u_res[mid_idx])
        ref_step_u = abs(u_ref[mid_idx + 1] - u_ref[mid_idx])
        assert abs(step_jump_u - ref_step_u) < 1e-12, "Step jump detected across restart boundary!"

        print("[TEST PASSED] Restart continuity verified to machine precision (< 1e-12)!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    test_restart_continuity()
