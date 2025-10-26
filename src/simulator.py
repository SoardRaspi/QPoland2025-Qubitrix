"""
simulator.py

High-level simulator that uses jj_physics and ising_map to:
- Run a JJ-array pulse-based annealer (MQT or thermal)
- Run a classical Simulated Annealing baseline
- Provide an example experiment for Max-Cut on a random weighted graph

Run: python simulator.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from jj_physics import hbar
from jj_physics import P_switch, kb, hbar

from typing import Optional, Dict

from jj_physics import P_switch  # expects jj_physics.py in same module path
from ising_map import graph_to_Ising_J, ising_energy, cut_value, random_weighted_graph

# -----------------------------
# Mapping: local effective field -> reduced bias i in [0, 0.9999]
# -----------------------------
def local_field_to_reduced_bias(h_eff: np.ndarray, i0: float = 0.5, alpha: float = 0.45, beta: float = 1.0) -> np.ndarray:
    """
    Map local effective field (real numbers) to reduced bias i = I/Ic in [0, 0.9999].
    Uses a saturated tanh mapping; parameters tune sensitivity.
    """
    h_eff = np.array(h_eff, dtype=float)
    return np.clip(i0 + alpha * np.tanh(beta * h_eff), 0.0, 0.9999)


# -----------------------------
# JJ-array pulse-step Monte Carlo simulator
# -----------------------------
def jj_array_simulator(J: np.ndarray,
                       h: Optional[np.ndarray] = None,
                       pulse_s: float = 10e-6,
                       T: Optional[float] = None,
                       i0: float = 0.5,
                       alpha: float = 0.45,
                       beta: float = 1.0,
                       n_steps: int = 2000,
                       seed: Optional[int] = None,
                       verbose: bool = False) -> Dict:
    """
    Run a pulse-based JJ-array annealer.
    - J: Ising coupling matrix (N x N)
    - h: local fields (N,), optional
    - pulse_s: pulse length [s]
    - T: temperature in Kelvin (if None -> quantum (MQT) regime)
    - i0, alpha, beta: mapping parameters
    - n_steps: number of pulses
    Returns dict with energies, cut_vals, best_cut, best_s, flips_record, final_s
    """
    rng = np.random.default_rng(seed)
    N = J.shape[0]
    if h is None:
        h = np.zeros(N)
    s = rng.choice([-1, 1], size=N)
    energies = []
    cut_vals = []
    best_cut = cut_value(s, -J)  # recall for Max-Cut mapping J = -A
    best_s = s.copy()
    energies.append(ising_energy(s, J, h))
    cut_vals.append(best_cut)
    flips_record = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        h_eff = J @ s + h  # local effective field
        i_reduced = local_field_to_reduced_bias(h_eff, i0=i0, alpha=alpha, beta=beta)
        # vectorized P_switch calls (loop here for safety because P_switch accepts scalar)
        P = np.array([P_switch(float(i_reduced[i]), pulse_s, T) for i in range(N)])
        rand = rng.random(size=N)
        flips = (rand < P)
        flips_record[t] = int(flips.sum())
        s[flips] *= -1
        E = ising_energy(s, J, h)
        energies.append(E)
        cutv = cut_value(s, -J)
        cut_vals.append(cutv)
        if cutv > best_cut:
            best_cut = cutv
            best_s = s.copy()
        if verbose and (t % max(1, n_steps // 10) == 0):
            print(f"[{t}/{n_steps}] cut={cutv:.4f} best_cut={best_cut:.4f} flips={flips_record[t]}")
    return {
        'energies': np.array(energies),
        'cut_vals': np.array(cut_vals),
        'best_cut': float(best_cut),
        'best_s': best_s,
        'final_s': s,
        'flips_record': flips_record
    }


# -----------------------------
# Classical Simulated Annealing baseline
# -----------------------------
def simulated_annealing(J: np.ndarray,
                        h: Optional[np.ndarray] = None,
                        T_start: float = 1.0,
                        T_end: float = 0.01,
                        n_steps: int = 2000,
                        seed: Optional[int] = None) -> Dict:
    rng = np.random.default_rng(seed)
    N = J.shape[0]
    if h is None:
        h = np.zeros(N)
    s = rng.choice([-1, 1], size=N)
    E = ising_energy(s, J, h)
    best_E = E
    best_s = s.copy()
    energies = [E]
    temps = np.linspace(T_start, T_end, n_steps)
    for t, T in enumerate(temps):
        i = rng.integers(0, N)
        s_new = s.copy()
        s_new[i] *= -1
        E_new = ising_energy(s_new, J, h)
        dE = E_new - E
        if dE < 0 or rng.random() < np.exp(-dE / (kb * max(T, 1e-300))):
            s = s_new
            E = E_new
        energies.append(E)
        if E < best_E:
            best_E = E
            best_s = s.copy()
    return {
        'energies': np.array(energies),
        'best_E': float(best_E),
        'best_s': best_s,
        'final_s': s
    }



# -----------------------------
# Example experiment (Max-Cut) when run as script
# -----------------------------
def example_experiment():
    # Graph parameters
    N = 24
    A = random_weighted_graph(N, p_edge=0.25, w_scale=1.0, seed=123)
    J = graph_to_Ising_J(A)  # J = -A for Max-Cut mapping

    # Simulator parameters
    pulse_s = 10e-6
    n_steps = 1200
    i0 = 0.5
    alpha = 0.42
    beta = 1.0

    # Run MQT (quantum-dominated)
    t0 = time.time()
    res_mqt = jj_array_simulator(J, pulse_s=pulse_s, T=None, i0=i0, alpha=alpha, beta=beta,
                                 n_steps=n_steps, seed=1, verbose=False)
    t_mqt = time.time() - t0

    # Run Thermal (classical thermal escapes at 50 mK)
    t0 = time.time()
    res_th = jj_array_simulator(J, pulse_s=pulse_s, T=0.05, i0=i0, alpha=alpha, beta=beta,
                                n_steps=n_steps, seed=1, verbose=False)
    t_th = time.time() - t0

    # Run Simulated Annealing baseline
    t0 = time.time()
    res_sa = simulated_annealing(J, T_start=1.0, T_end=0.001, n_steps=n_steps, seed=1)
    t_sa = time.time() - t0

    print(f"Finished runs: MQT time={t_mqt:.2f}s Thermal time={t_th:.2f}s SA time={t_sa:.2f}s")
    cut_sa = cut_value(res_sa['final_s'], A)
    print(f"SA final cut (direct): {cut_sa:.4f}")

    # Plot cut traces
    plt.figure(figsize=(10, 4))
    plt.plot(res_mqt['cut_vals'], label='MQT (T≈0)')
    plt.plot(res_th['cut_vals'], label='Thermal T=50 mK')
    plt.axhline(y=cut_sa, color='C2', linestyle='--', label='SA final cut (baseline)')
    plt.xlabel('Pulse step')
    plt.ylabel('Cut value (higher is better)')
    plt.title('Cut value vs Pulse step: MQT vs Thermal (SA baseline shown)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot flips per pulse
    plt.figure(figsize=(10, 3))
    plt.plot(res_mqt['flips_record'], label='MQT flips/pulse')
    plt.plot(res_th['flips_record'], label='Thermal flips/pulse')
    plt.xlabel('Pulse step')
    plt.ylabel('Number of flips')
    plt.title('Flip activity per pulse: MQT vs Thermal')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Adjacency and final spin configs
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(A, interpolation='nearest')
    plt.title('Graph adjacency (weights)')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    combined = np.vstack([res_mqt['final_s'], res_th['final_s'], res_sa['final_s']])
    plt.imshow(combined, aspect='auto', interpolation='nearest')
    plt.yticks([0, 1, 2], ['MQT final s', 'Thermal final s', 'SA final s'])
    plt.title('Final spin configurations (rows)')
    plt.tight_layout()
    plt.show()

    print("Summary:")
    print(f"MQT best cut: {res_mqt['best_cut']:.4f}  final cut: {res_mqt['cut_vals'][-1]:.4f}")
    print(f"Thermal best cut: {res_th['best_cut']:.4f}  final cut: {res_th['cut_vals'][-1]:.4f}")
    print(f"SA final cut: {cut_sa:.4f}  SA best E: {res_sa['best_E']:.4f}")


# small helper imports used in example_experiment (to avoid circular imports)
from ising_map import random_weighted_graph, graph_to_Ising_J  # noqa: E402
from ising_map import cut_value  # noqa: E402
from jj_physics import kb  # noqa: E402
import time  # noqa: E402

if __name__ == "__main__":
    example_experiment()
