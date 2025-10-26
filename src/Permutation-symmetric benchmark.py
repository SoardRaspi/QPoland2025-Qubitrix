"""
Permutation-symmetric benchmark problems for quantum optimization,
integrated with JJ annealer + PennyLane solvers.

Implements cost functions from:
Muthukrishnan, Albash, & Lidar (2015),
'Tunneling and speedup in quantum optimization for permutation-symmetric problems'
(arXiv:1511.03910)
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from typing import Tuple, Dict

# ----------------------------------------------------------
# 1. Permutation-symmetric cost functions
# ----------------------------------------------------------

def fixed_plateau(x: np.ndarray, l: int = 3, u: int = 6) -> float:
    """
    Fixed Plateau function from Eq. (14) of the paper.
    Cost depends only on Hamming weight w = sum(x_i)
    """
    w = np.sum(x)
    if l <= w <= u:
        return u - l  # constant plateau region
    else:
        return w


def spike(x: np.ndarray, spike_height: float = 10.0, spike_pos: int = 5) -> float:
    """
    Spike cost function with a single sharp barrier in Hamming weight space.
    """
    w = np.sum(x)
    if w == spike_pos:
        return w + spike_height
    return w


def hamming_ramp(x: np.ndarray) -> float:
    """
    Simple linear ramp cost = Hamming weight.
    (Used as a control problem without tunneling barrier.)
    """
    return np.sum(x)


# ----------------------------------------------------------
# 2. Convert symmetric cost function → Ising (J, h)
# ----------------------------------------------------------

def symmetric_to_ising(N: int, cost_fn) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a permutation-symmetric cost function f(x) to
    equivalent Ising couplings J, h by fitting over all bitstrings.
    """
    states = np.array(list(product([0, 1], repeat=N)))
    energies = np.array([cost_fn(x) for x in states])
    spins = 1 - 2 * states  # map {0,1} → {+1,-1}

    # Fit Ising model: E = -∑ h_i s_i - ∑_ij J_ij s_i s_j / 2
    h = np.zeros(N)
    J = np.zeros((N, N))

    for i in range(N):
        h[i] = -0.25 * np.dot(spins[:, i], energies) / len(states)
        for j in range(i + 1, N):
            J[i, j] = -0.25 * np.dot(spins[:, i] * spins[:, j], energies) / len(states)
            J[j, i] = J[i, j]
    return J, h


# ----------------------------------------------------------
# 3. Visualization utilities
# ----------------------------------------------------------

def plot_cost_vs_hamming(cost_fn, N=10, title="Cost Landscape"):
    ws = np.arange(0, N + 1)
    costs = [cost_fn(np.array([1]*w + [0]*(N-w))) for w in ws]
    plt.figure(figsize=(6, 4))
    plt.plot(ws, costs, "-o")
    plt.xlabel("Hamming weight w")
    plt.ylabel("Cost f(w)")
    plt.title(title)
    plt.grid(True)
    plt.show()


# ----------------------------------------------------------
# 4. Benchmark generator
# ----------------------------------------------------------

def load_benchmark(name: str, N: int = 8) -> Dict:
    """
    Load a predefined permutation-symmetric benchmark.
    Returns dict with J, h, and metadata.
    """
    if name.lower() == "plateau":
        fn = lambda x: fixed_plateau(x, l=int(0.3*N), u=int(0.6*N))
    elif name.lower() == "spike":
        fn = lambda x: spike(x, spike_height=8, spike_pos=int(0.5*N))
    elif name.lower() == "ramp":
        fn = hamming_ramp
    else:
        raise ValueError("Unknown benchmark: choose from ['plateau', 'spike', 'ramp']")

    J, h = symmetric_to_ising(N, fn)
    return {
        "J": J,
        "h": h,
        "cost_fn": fn,
        "N": N,
        "name": name
    }


# ----------------------------------------------------------
# 5. Example usage
# ----------------------------------------------------------

if __name__ == "__main__":
    N = 10
    for name in ["plateau", "spike", "ramp"]:
        bench = load_benchmark(name, N)
        print(f"\n{name.upper()} benchmark ({N} qubits)")
        print("J shape:", bench["J"].shape)
        print("h shape:", bench["h"].shape)
        plot_cost_vs_hamming(bench["cost_fn"], N, title=f"{name.capitalize()} landscape")
