# jj_annealer_simulation.py

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

# -----------------------------
# Physical constants (SI)
# -----------------------------
hbar = 1.054571817e-34
e = 1.602176634e-19
phi0 = (6.62607015e-34) / (2*e)
kb = 1.380649e-23

# -----------------------------
# JJ device params (tunable)
# -----------------------------
Ic = 1.0e-6      # [A]
C  = 1.0e-12     # [F]
EJ = phi0 * Ic / (2*np.pi)

# Helper functions
def omega_p(i_reduced, Ic=Ic, C=C):
    i = np.clip(i_reduced, 0.0, 0.999999)
    return np.sqrt(2*e*Ic/(hbar*C)) * (1 - i**2)**0.25

def delta_U(i_reduced, EJ=EJ):
    i = np.clip(i_reduced, 0.0, 0.999999)
    return 2*EJ * (np.sqrt(1 - i**2) - i*np.arccos(i))

def gamma_th(i_reduced, T, Ic=Ic, C=C):
    op = omega_p(i_reduced, Ic, C)
    dU = delta_U(i_reduced, EJ)
    # Kramers prefactor ~ ω_p / 2π (underdamped simplified)
    return (op/(2*np.pi)) * np.exp(- dU / (kb * T))

def gamma_mqt(i_reduced, Ic=Ic, C=C):
    op = omega_p(i_reduced, Ic, C)
    dU = delta_U(i_reduced, EJ)
    # Cubic-potential WKB prefactor/exponent (approx)
    pref = (op/(2*np.pi)) * np.sqrt(864.0 * dU / (hbar * op))
    expo = np.exp(-36.0 * dU / (5.0 * hbar * op))
    return pref * expo

def P_switch(i_reduced, pulse_s, T=None):
    gamma = gamma_mqt(i_reduced) if (T is None) else gamma_th(i_reduced, T)
    # protect against overflow/underflow
    lam = gamma * pulse_s
    lam = np.clip(lam, 0, 50.0)
    return 1.0 - np.exp(-lam)

# -----------------------------
# Ising problem generator
# -----------------------------
def random_ising(N, p_edge=0.3, J_scale=1.0, seed=None):
    rng = np.random.default_rng(seed)
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < p_edge:
                val = rng.normal(loc=0.0, scale=J_scale)
                J[i,j] = val
                J[j,i] = val
    h = rng.normal(0.0, 0.1, size=N)
    return J, h

def ising_energy(s, J, h):
    # s: +/-1
    return -0.5 * s @ J @ s - h @ s

# -----------------------------
# Mapping local field -> reduced bias i in [0, 0.9999]
# -----------------------------
def local_field_to_reduced_bias(h_eff, i0=0.5, alpha=0.45, beta=1.0):
    # h_eff can be large; use tanh to saturate
    return np.clip(i0 + alpha * np.tanh(beta * h_eff), 0.0, 0.9999)

# -----------------------------
# Simulator: pulse-step Monte Carlo
# -----------------------------
def run_simulation(J, h, n_steps=2000, pulse_s=10e-6, T=None,
                   i0=0.5, alpha=0.45, beta=1.0, seed=None, verbose=False):
    rng = np.random.default_rng(seed)
    N = len(h)
    # initial random spins (+1 / -1)
    s = rng.choice([-1, 1], size=N)
    energies = []
    best_E = ising_energy(s, J, h)
    best_s = s.copy()
    energies.append(best_E)
    flips_record = np.zeros(n_steps)

    for t in range(n_steps):
        # compute local effective fields for each spin
        h_eff = J @ s + h  # local fields
        # map to reduced bias (higher field -> higher bias -> higher escape P)
        i_reduced = local_field_to_reduced_bias(h_eff, i0=i0, alpha=alpha, beta=beta)
        # compute switch probabilities (vectorized)
        if T is None:
            P = np.array([P_switch(i_reduced[i], pulse_s, T=None) for i in range(N)])
        else:
            P = np.array([P_switch(i_reduced[i], pulse_s, T=T) for i in range(N)])
        # decide which junctions flip (trial flips)
        rand = rng.random(size=N)
        flips = (rand < P)
        flips_record[t] = flips.sum()
        # apply flips: flipping a spin s_i -> -s_i
        s[flips] *= -1
        E = ising_energy(s, J, h)
        energies.append(E)
        if E < best_E:
            best_E = E
            best_s = s.copy()
        if verbose and (t % max(1, n_steps//10) == 0):
            print(f"[{t}/{n_steps}] E={E:.3f} best={best_E:.3f} flips={flips.sum()}")
    return {
        'energies': np.array(energies),
        'best_E': best_E,
        'best_s': best_s,
        'final_s': s,
        'flips_record': flips_record
    }

# -----------------------------
# Example run & plotting
# -----------------------------
def example_run():
    N = 64
    J, h = random_ising(N, p_edge=0.1, J_scale=1.0, seed=42)
    # Two runs: MQT (T=None) vs Thermal (T=0.05 K)
    res_q = run_simulation(J, h, n_steps=1200, pulse_s=10e-6, T=None, seed=1)
    res_th = run_simulation(J, h, n_steps=1200, pulse_s=10e-6, T=0.05, seed=1)

    plt.figure(figsize=(8,4))
    plt.plot(res_q['energies'], label='MQT (T≈0)')
    plt.plot(res_th['energies'], label='Thermal T=50 mK')
    plt.xlabel('Pulse step')
    plt.ylabel('Ising energy')
    plt.legend()
    plt.title('Energy trace: MQT vs Thermal')
    plt.tight_layout()
    plt.show()

    # histogram of flips
    plt.figure(figsize=(8,3))
    plt.plot(res_q['flips_record'], label='MQT flips / pulse')
    plt.plot(res_th['flips_record'], label='Thermal flips / pulse')
    plt.xlabel('Pulse step')
    plt.ylabel('Number of flips')
    plt.legend()
    plt.title('Flip activity per pulse')
    plt.tight_layout()
    plt.show()

    print("MQT best energy:", res_q['best_E'])
    print("Thermal best energy:", res_th['best_E'])

if __name__ == "__main__":
    example_run()
