import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# Physical constants for JJ dynamics
kb = 1.380649e-23  # Boltzmann constant
P_SWITCH_BASE = 1e-6  # Base switching probability

def P_switch(i_reduced: float, pulse_s: float, T: Optional[float] = None) -> float:
    """Switching probability for a Josephson junction with quantum tunneling."""
    if i_reduced >= 1.0:
        return 1.0
    if i_reduced <= 0:
        return 0.0

    # Quantum tunneling contribution (dominant at low T)
    quantum_rate = np.exp(-10 * (1 - i_reduced))

    # Thermal contribution (if T is provided)
    thermal_rate = 0.0
    if T is not None and T > 0:
        thermal_rate = 0.1 * np.exp(-5 * (1 - i_reduced) / (kb * T * 1e23))

    total_rate = quantum_rate + thermal_rate
    return min(1.0, total_rate * pulse_s * 1e6)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_solution: np.ndarray
    best_cost: float
    cost_history: np.ndarray
    final_solution: np.ndarray
    algorithm: str
    problem_value: float = 0.0  # Max-cut value or knapsack value


# ============================================================================
# PROBLEM 1: MAX-CUT
# ============================================================================

class MaxCutProblem:
    """
    Max-Cut Problem: Partition graph vertices to maximize edge weights crossing partition.

    Given a weighted graph G=(V,E), find partition S ⊆ V that maximizes:
        Cut(S) = Σ_{i∈S, j∉S} w_ij

    This is equivalent to the Ising model with J_ij = -w_ij
    """

    def __init__(self, adjacency_matrix: np.ndarray):
        self.A = adjacency_matrix
        self.N = len(adjacency_matrix)
        self.J = -adjacency_matrix  # Ising coupling

    def cut_value(self, partition: np.ndarray) -> float:
        """Calculate cut value for a given partition (s_i ∈ {-1, +1})."""
        return 0.25 * np.sum(self.A * (1 - np.outer(partition, partition)))

    def ising_energy(self, spins: np.ndarray) -> float:
        """Ising energy (to minimize): E = -Cut(s)."""
        return spins @ self.J @ spins

    @staticmethod
    def random_graph(n_vertices: int, edge_prob: float = 0.4,
                    max_weight: float = 10.0, seed: Optional[int] = None) -> 'MaxCutProblem':
        """Generate random weighted graph."""
        rng = np.random.default_rng(seed)
        A = np.zeros((n_vertices, n_vertices))
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                if rng.random() < edge_prob:
                    w = rng.uniform(1, max_weight)
                    A[i, j] = A[j, i] = w
        return MaxCutProblem(A)

    def visualize_solution(self, partition: np.ndarray, ax=None):
        """Visualize the graph partition."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Simple circular layout
        angles = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles)])

        # Draw edges
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.A[i, j] > 0:
                    color = 'red' if partition[i] != partition[j] else 'lightgray'
                    width = self.A[i, j] / np.max(self.A) * 3
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                           color=color, linewidth=width, alpha=0.6, zorder=1)

        # Draw vertices
        colors = ['blue' if s > 0 else 'orange' for s in partition]
        ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=500, zorder=2, edgecolors='black')
        for i, (x, y) in enumerate(pos):
            ax.text(x, y, str(i), ha='center', va='center', fontsize=12, fontweight='bold')

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Max-Cut Solution (Cut Value: {self.cut_value(partition):.1f})')


# ============================================================================
# PROBLEM 2: KNAPSACK
# ============================================================================

class KnapsackProblem:
    """
    0-1 Knapsack Problem: Select items to maximize value without exceeding capacity.

    Given items with values v_i and weights w_i, and capacity W, maximize:
        Σ v_i x_i  subject to  Σ w_i x_i ≤ W,  x_i ∈ {0,1}

    Encoded as Ising model with penalty for constraint violation.
    """

    def __init__(self, values: np.ndarray, weights: np.ndarray, capacity: float, penalty: float = 10.0):
        self.values = np.array(values)
        self.weights = np.array(weights)
        self.capacity = capacity
        self.penalty = penalty
        self.N = len(values)
        self.J, self.h = self._to_ising()

    def _to_ising(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert knapsack to Ising model with penalty method."""
        n = self.N
        J = np.zeros((n, n))
        h = np.zeros(n)

        # Penalty term: λ(Σw_i x_i - W)² where x_i = (1-s_i)/2
        for i in range(n):
            for j in range(n):
                if i != j:
                    J[i, j] = self.penalty * self.weights[i] * self.weights[j] / 4.0
            h[i] = -self.values[i]/2 + self.penalty * self.weights[i] * (self.weights[i] - 2*self.capacity) / 4.0

        return J, h

    def spin_to_binary(self, spins: np.ndarray) -> np.ndarray:
        """Convert spins {-1,+1} to binary {0,1}."""
        return ((1 - spins) / 2).astype(int)

    def evaluate(self, spins: np.ndarray) -> Dict[str, float]:
        """Evaluate knapsack solution."""
        x = self.spin_to_binary(spins)
        total_value = np.sum(self.values * x)
        total_weight = np.sum(self.weights * x)
        feasible = total_weight <= self.capacity
        return {
            'value': total_value,
            'weight': total_weight,
            'feasible': feasible,
            'violation': max(0, total_weight - self.capacity)
        }

    def ising_energy(self, spins: np.ndarray) -> float:
        """Ising energy including penalty."""
        return spins @ self.J @ spins + np.sum(self.h * spins)

    @staticmethod
    def random_instance(n_items: int, seed: Optional[int] = None) -> 'KnapsackProblem':
        """Generate random knapsack instance."""
        rng = np.random.default_rng(seed)
        values = rng.integers(5, 30, size=n_items)
        weights = rng.integers(1, 15, size=n_items)
        capacity = np.sum(weights) * 0.5  # 50% of total weight
        return KnapsackProblem(values, weights, capacity)


# ============================================================================
# OPTIMIZATION ALGORITHMS
# ============================================================================

def josephson_optimizer(problem, mode: str = 'quantum', n_steps: int = 1000,
                       temperature: Optional[float] = None, seed: Optional[int] = None) -> OptimizationResult:
    """
    Solve optimization problem using Josephson junction array dynamics.

    Args:
        problem: MaxCutProblem or KnapsackProblem instance
        mode: 'quantum' (pure tunneling) or 'thermal' (with thermal activation)
        n_steps: Number of evolution steps
        temperature: Temperature for thermal mode (K)
        seed: Random seed
    """
    rng = np.random.default_rng(seed)
    N = problem.N
    J = problem.J if hasattr(problem, 'J') else problem.J
    h = np.zeros(N) if not hasattr(problem, 'h') else problem.h

    # Initialize random spin configuration
    spins = rng.choice([-1, 1], size=N)

    cost_history = []
    best_cost = np.inf
    best_spins = spins.copy()

    # Dynamics parameters
    pulse_duration = 1e-6  # 1 microsecond pulses
    i0 = 0.5  # Bias current offset
    alpha = 0.45  # Field coupling strength
    beta = 1.0  # Nonlinearity parameter

    T = temperature if mode == 'thermal' else None

    for step in range(n_steps):
        # Calculate effective field on each junction
        h_eff = J @ spins + h

        # Map field to reduced bias current
        i_reduced = np.clip(i0 + alpha * np.tanh(beta * h_eff), 0.0, 0.999)

        # Calculate switching probabilities (quantum tunneling + thermal)
        P = np.array([P_switch(ir, pulse_duration, T) for ir in i_reduced])

        # Stochastic spin flips
        flips = rng.random(size=N) < P
        spins[flips] *= -1

        # Evaluate current solution
        cost = problem.ising_energy(spins)
        cost_history.append(cost)

        if cost < best_cost:
            best_cost = cost
            best_spins = spins.copy()

    # Calculate problem-specific value
    if isinstance(problem, MaxCutProblem):
        problem_value = problem.cut_value(best_spins)
    else:  # KnapsackProblem
        result = problem.evaluate(best_spins)
        problem_value = result['value'] if result['feasible'] else 0

    return OptimizationResult(
        best_solution=best_spins,
        best_cost=best_cost,
        cost_history=np.array(cost_history),
        final_solution=spins,
        algorithm=f'JJ-{mode}',
        problem_value=problem_value
    )


def simulated_annealing(problem, n_steps: int = 1000, T_start: float = 2.0,
                       T_end: float = 0.01, seed: Optional[int] = None) -> OptimizationResult:
    """Classical simulated annealing solver."""
    rng = np.random.default_rng(seed)
    N = problem.N
    J = problem.J if hasattr(problem, 'J') else problem.J
    h = np.zeros(N) if not hasattr(problem, 'h') else problem.h

    spins = rng.choice([-1, 1], size=N)
    cost = problem.ising_energy(spins)
    best_cost = cost
    best_spins = spins.copy()
    cost_history = [cost]

    temps = np.linspace(T_start, T_end, n_steps)

    for T in temps:
        # Propose random spin flip
        i = rng.integers(0, N)
        spins[i] *= -1
        new_cost = problem.ising_energy(spins)
        delta_cost = new_cost - cost

        # Metropolis criterion
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / (kb * T * 1e23)):
            cost = new_cost
        else:
            spins[i] *= -1  # Reject

        cost_history.append(cost)

        if cost < best_cost:
            best_cost = cost
            best_spins = spins.copy()

    if isinstance(problem, MaxCutProblem):
        problem_value = problem.cut_value(best_spins)
    else:
        result = problem.evaluate(best_spins)
        problem_value = result['value'] if result['feasible'] else 0

    return OptimizationResult(
        best_solution=best_spins,
        best_cost=best_cost,
        cost_history=np.array(cost_history),
        final_solution=spins,
        algorithm='Simulated Annealing',
        problem_value=problem_value
    )


def greedy_solver(problem) -> OptimizationResult:
    """Simple greedy heuristic for comparison."""
    if isinstance(problem, MaxCutProblem):
        # Start with random partition, greedily improve
        spins = np.ones(problem.N)
        for i in range(problem.N):
            spins[i] = -1
            if problem.cut_value(spins) < problem.cut_value(spins * np.array([1 if j != i else -1 for j in range(problem.N)])):
                spins[i] = 1
        problem_value = problem.cut_value(spins)
    else:  # Knapsack
        # Sort by value/weight ratio
        ratios = problem.values / problem.weights
        indices = np.argsort(ratios)[::-1]
        spins = np.ones(problem.N)  # Start with nothing (spins=1 means x=0)
        weight = 0
        for i in indices:
            if weight + problem.weights[i] <= problem.capacity:
                spins[i] = -1  # Include item
                weight += problem.weights[i]
        result = problem.evaluate(spins)
        problem_value = result['value']

    cost = problem.ising_energy(spins)
    return OptimizationResult(
        best_solution=spins,
        best_cost=cost,
        cost_history=np.array([cost]),
        final_solution=spins,
        algorithm='Greedy',
        problem_value=problem_value
    )


# ============================================================================
# COMPARISON AND VISUALIZATION
# ============================================================================

def compare_solvers(problem, n_steps: int = 1000, seed: int = 42):
    """Run all solvers and compare results."""
    print(f"\n{'='*70}")
    print(f"Problem: {problem.__class__.__name__} (N={problem.N})")
    print(f"{'='*70}\n")

    results = {}

    # Josephson Junction (Quantum Tunneling)
    print("Running: JJ Quantum Tunneling...")
    results['JJ-Quantum'] = josephson_optimizer(problem, mode='quantum', n_steps=n_steps, seed=seed)

    # Josephson Junction (Thermal)
    print("Running: JJ Thermal (T=0.05K)...")
    results['JJ-Thermal'] = josephson_optimizer(problem, mode='thermal', n_steps=n_steps,
                                                temperature=0.05, seed=seed)

    # Simulated Annealing
    print("Running: Simulated Annealing...")
    results['SA'] = simulated_annealing(problem, n_steps=n_steps, seed=seed)

    # Greedy
    print("Running: Greedy Heuristic...")
    results['Greedy'] = greedy_solver(problem)

    # Print results
    print(f"\n{'Algorithm':<20} {'Ising Energy':<15} {'Problem Value':<15}")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name:<20} {result.best_cost:>14.2f} {result.problem_value:>14.2f}")

    # Visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Energy evolution
    ax1 = fig.add_subplot(gs[0, :])
    for name, result in results.items():
        if len(result.cost_history) > 1:
            ax1.plot(result.cost_history, label=name, linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Ising Energy', fontsize=12)
    ax1.set_title('Optimization Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Best solutions visualization
    if isinstance(problem, MaxCutProblem):
        ax2 = fig.add_subplot(gs[1, 0])
        problem.visualize_solution(results['JJ-Quantum'].best_solution, ax=ax2)
        ax2.set_title('JJ-Quantum Solution', fontsize=12, fontweight='bold')

        ax3 = fig.add_subplot(gs[1, 1])
        problem.visualize_solution(results['SA'].best_solution, ax=ax3)
        ax3.set_title('Simulated Annealing Solution', fontsize=12, fontweight='bold')

    else:  # Knapsack
        # Show selected items
        for idx, (name, result) in enumerate(list(results.items())[:2]):
            ax = fig.add_subplot(gs[1, idx])
            eval_result = problem.evaluate(result.best_solution)
            selected = problem.spin_to_binary(result.best_solution)

            colors = ['green' if s else 'lightgray' for s in selected]
            bars = ax.bar(range(problem.N), problem.values, color=colors, edgecolor='black')
            ax.set_xlabel('Item', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(f"{name}\nValue: {eval_result['value']:.0f}, "
                        f"Weight: {eval_result['weight']:.1f}/{problem.capacity:.1f}",
                        fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f"{problem.__class__.__name__} - Quantum vs Classical Optimization",
                 fontsize=16, fontweight='bold')
    plt.show()

    return results


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUANTUM-INSPIRED COMBINATORIAL OPTIMIZATION")
    print("Josephson Junction Arrays for NP-Hard Problems")
    print("="*70)

    # ========== MAX-CUT PROBLEM ==========
    print("\n\n>>> MAX-CUT PROBLEM <<<")
    print("Task: Partition graph vertices to maximize cut weight")

    maxcut = MaxCutProblem.random_graph(n_vertices=10, edge_prob=0.4, max_weight=10, seed=42)
    maxcut_results = compare_solvers(maxcut, n_steps=800, seed=42)

    # ========== KNAPSACK PROBLEM ==========
    print("\n\n>>> 0-1 KNAPSACK PROBLEM <<<")
    print("Task: Select items to maximize value within weight capacity")

    knapsack = KnapsackProblem.random_instance(n_items=12, seed=123)
    print(f"Items: {knapsack.N}")
    print(f"Capacity: {knapsack.capacity:.1f}")
    print(f"Total available value: {np.sum(knapsack.values)}")

    knapsack_results = compare_solvers(knapsack, n_steps=800, seed=42)

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
