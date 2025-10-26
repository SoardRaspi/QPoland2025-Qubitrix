%%writefile jj_physics.py
"""
jj_physics.py

Josephson-junction physics utilities:
- plasma frequency omega_p(i)
- washboard barrier delta_U(i)
- thermal escape rate gamma_th(i, T)
- macroscopic quantum tunnelling rate gamma_mqt(i)
- switching probability P_switch(i, t_p, T=None)

All formulas use SI units.
"""

from typing import Optional
import numpy as np

# Physical constants (SI)
h = 6.62607015e-34          # Planck constant [J·s]
hbar = 1.054571817e-34      # Reduced Planck constant [J·s]
e = 1.602176634e-19         # Elementary charge [C]
phi0 = h / (2 * e)          # Flux quantum [Wb]
kb = 1.380649e-23           # Boltzmann constant [J/K]

# Default device parameters (tunable)
DEFAULT_Ic = 1.0e-6    # Critical current [A]
DEFAULT_C  = 1.0e-12   # Junction capacitance [F]
DEFAULT_EJ = phi0 * DEFAULT_Ic / (2 * np.pi)


def omega_p(i_reduced: float, Ic: float = DEFAULT_Ic, C: float = DEFAULT_C) -> float:
    """
    Plasma angular frequency ω_p(i) [rad/s] at reduced bias i = I/Ic.
    Uses underdamped tunnel-junction approximation:
      ω_p(i) = sqrt(2 e Ic / (ħ C)) * (1 - i^2)^(1/4)
    """
    i = float(i_reduced)
    i = np.clip(i, 0.0, 0.999999)
    return np.sqrt(2.0 * e * Ic / (hbar * C)) * (1.0 - i**2)**0.25


def delta_U(i_reduced: float, EJ: float = DEFAULT_EJ) -> float:
    """
    Washboard barrier height ΔU(i) [J] for a current-biased JJ:
      ΔU(i) = 2 EJ [ sqrt(1 - i^2) - i arccos(i) ]
    Valid for 0 <= i < 1.
    """
    i = float(i_reduced)
    i = np.clip(i, 0.0, 0.999999)
    return 2.0 * EJ * (np.sqrt(1.0 - i**2) - i * np.arccos(i))


def gamma_th(i_reduced: float, T: float, Ic: float = DEFAULT_Ic, C: float = DEFAULT_C,
             EJ: float = DEFAULT_EJ) -> float:
    """
    Thermal (Kramers-like) escape rate Γ_th [1/s]:
      Γ_th ≈ (ω_p / 2π) * exp(-ΔU / (k_B T))
    Underdamped prefactor used (damping not explicitly included).
    """
    if T <= 0.0:
        return 0.0
    op = omega_p(i_reduced, Ic=Ic, C=C)
    dU = delta_U(i_reduced, EJ=EJ)
    exponent = - dU / (kb * T)
    # Avoid overflow / underflow for extreme numbers
    if exponent < -700:
        return 0.0
    return (op / (2.0 * np.pi)) * np.exp(exponent)


def gamma_mqt(i_reduced: float, Ic: float = DEFAULT_Ic, C: float = DEFAULT_C,
              EJ: float = DEFAULT_EJ) -> float:
    """
    Macroscopic Quantum Tunneling (MQT) escape rate Γ_MQT [1/s], cubic-potential WKB approximation:
      Γ_MQT ≈ (ω_p / 2π) * sqrt(864 ΔU / (ħ ω_p)) * exp( -36 ΔU / (5 ħ ω_p) )
    """
    op = omega_p(i_reduced, Ic=Ic, C=C)
    dU = delta_U(i_reduced, EJ=EJ)
    arg = 36.0 * dU / (5.0 * hbar * op)
    if arg > 700:  # exponent too large -> rate virtually zero
        return 0.0
    # guard small denominators
    factor = max(1e-300, (864.0 * dU) / (hbar * op))
    pref = (op / (2.0 * np.pi)) * np.sqrt(factor)
    return pref * np.exp(-arg)


def P_switch(i_reduced: float, pulse_s: float, T: Optional[float] = None,
             Ic: float = DEFAULT_Ic, C: float = DEFAULT_C, EJ: float = DEFAULT_EJ,
             hbar_eff: float = hbar) -> float:
    if pulse_s <= 0.0:
        return 0.0
    gamma = gamma_mqt(i_reduced, Ic=Ic, C=C, EJ=EJ, hbar_eff=hbar_eff) if T is None \
            else gamma_th(i_reduced, T, Ic=Ic, C=C, EJ=EJ)
    lam = gamma * pulse_s
    lam = float(np.clip(lam, 0.0, 50.0))
    return 1.0 - np.exp(-lam)


def crossover_temperature(i_reduced: float, Ic: float = DEFAULT_Ic, C: float = DEFAULT_C) -> float:
    """
    Rule-of-thumb crossover temperature T* ~ ħ ω_p / (2π k_B) at reduced bias i.
    Returns T* in Kelvins.
    """
    op = omega_p(i_reduced, Ic=Ic, C=C)
    return (hbar * op) / (2.0 * np.pi * kb)
def gamma_mqt(i_reduced: float, Ic: float = DEFAULT_Ic, C: float = DEFAULT_C,
              EJ: float = DEFAULT_EJ, hbar_eff: float = hbar) -> float:
    """
    Macroscopic Quantum Tunneling (MQT) escape rate with tunable effective Planck constant.
    """
    op = omega_p(i_reduced, Ic=Ic, C=C)
    dU = delta_U(i_reduced, EJ=EJ)
    arg = 36.0 * dU / (5.0 * hbar_eff * op)
    if arg > 700:  # exponent too large -> rate virtually zero
        return 0.0
    factor = max(1e-300, (864.0 * dU) / (hbar_eff * op))
    pref = (op / (2.0 * np.pi)) * np.sqrt(factor)
    return pref * np.exp(-arg)
