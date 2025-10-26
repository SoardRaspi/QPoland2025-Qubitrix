%%writefile ising_map.py
"""
ising_map.py

Utilities for mapping combinatorial optimization problems to Ising form
and computing objective values.

- graph_to_Ising_J(A): converts adjacency matrix A for Max-Cut into Ising coupling J
- cut_value(s, A): computes Max-Cut value for spin vector s in {+1,-1}
- ising_energy(s, J, h=None): computes Ising energy E = -0.5 s^T J s - h^T s
- random_weighted_graph(N, p_edge, w_scale): generate random weighted graph adjacency
"""

from typing import Optional, Tuple
import numpy as np


def graph_to_Ising_J(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Convert adjacency matrix A (symmetric, zero diagonal) for Max-Cut into Ising coupling J.
    For Max-Cut we can use J = -A so that minimizing E_ising = -0.5 s^T J s corresponds to maximizing the cut.
    """
    A = np.array(adj_matrix, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    J = -A.copy()
    np.fill_diagonal(J, 0.0)
    return J


def cut_value(s: np.ndarray, A: np.ndarray) -> float:
    """
    Compute Max-Cut value for spin vector s ∈ {+1, -1} and adjacency A:
      cut = 0.5 * sum_{i<j} A_ij (1 - s_i s_j)
    """
    s = np.array(s, dtype=int).reshape(-1)
    A = np.array(A, dtype=float)
    N = s.size
    if A.shape != (N, N):
        raise ValueError("Adjacency matrix size must match spin vector length.")
    # Efficient vectorized computation:
    # cut = 0.25 * sum_{i,j} A_ij (1 - s_i s_j)  (since each pair counted twice)
    return 0.25 * np.sum(A * (1.0 - np.outer(s, s)))


def ising_energy(s: np.ndarray, J: np.ndarray, h: Optional[np.ndarray] = None) -> float:
    """
    Compute Ising energy:
      E = -0.5 * s^T J s - h^T s
    """
    s = np.array(s, dtype=float).reshape(-1)
    J = np.array(J, dtype=float)
    if h is None:
        h = np.zeros_like(s)
    return -0.5 * float(s @ J @ s) - float(np.dot(h, s))


def random_weighted_graph(N: int, p_edge: float = 0.3, w_scale: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a symmetric random weighted adjacency matrix A (zero diagonal).
    - N: number of nodes
    - p_edge: probability of an edge between i and j
    - w_scale: weight scaling factor (weights drawn ~ Uniform(0.5, 1.5) * w_scale)
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p_edge:
                val = rng.uniform(0.5, 1.5) * w_scale
                A[i, j] = val
                A[j, i] = val
    return A
