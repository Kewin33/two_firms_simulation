import matplotlib.pyplot as plt
import numpy as np
from util import calc_xhat, integral
from distribution import GaussianDistribution, UniformDistribution
from equilibrium import calculate_equilibrium
import math

def plot_f(mu, cost, a, b, dist):
    def f(x):
        return integral(lambda e: (e - x) * dist.pdf(e), x, b) - cost / mu

    x_values = np.linspace(a, b, 500)
    y_values = [f(x) for x in x_values]

    plt.plot(x_values, y_values, label="f(x)")
    plt.axhline(0, color="red", linestyle="--", label="y=0")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Plot of f(x)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_equilibrium_single_variable(mu, cost, n, dist, a, b):
    """
    Plot the equilibrium as a function of a single variable (mu, cost, or n).

    Parameters:
    - mu: Fixed value for mu.
    - cost: Fixed value for search cost.
    - n: Fixed value for the number of firms.
    - dist: Distribution object.
    - a, b: Integration bounds.
    """
    from equilibrium import calculate_equilibrium

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def safe_equilibrium(mu_val, cost_val, n_val):
        try:
            return calculate_equilibrium(mu_val, cost_val, n_val, dist, a, b)
        except Exception:
            print(f"Failed to calculate equilibrium for mu={mu_val}, cost={cost_val}, n={n_val}")
            return np.nan

    # Plot equilibrium vs. mu
    mu_values = np.concatenate((np.linspace(0.01, 0.5, 100), np.linspace(1.0, 2.0, 25)))
    eq_values_mu = [safe_equilibrium(mu_val, cost, n) for mu_val in mu_values]
    mu_failures = int(np.isnan(eq_values_mu).sum())
    axes[0].plot(mu_values, eq_values_mu, label=f"cost={cost}, n={n}")
    axes[0].set_title("Equilibrium vs. mu")
    axes[0].set_xlabel("mu")
    axes[0].set_ylabel("Equilibrium")
    axes[0].legend()
    axes[0].grid()

    # Plot equilibrium vs. cost
    cost_values = np.linspace(0.1, 1.0, 50)
    eq_values_cost = [safe_equilibrium(mu, cost_val, n) for cost_val in cost_values]
    cost_failures = int(np.isnan(eq_values_cost).sum())
    axes[1].plot(cost_values, eq_values_cost, label=f"mu={mu}, n={n}")
    axes[1].set_title("Equilibrium vs. cost")
    axes[1].set_xlabel("Search cost")
    axes[1].set_ylabel("Equilibrium")
    axes[1].legend()
    axes[1].grid()

    # Plot equilibrium vs. n
    n_values = range(1, 20)
    eq_values_n = [safe_equilibrium(mu, cost, n_val) for n_val in n_values]
    n_failures = int(np.isnan(eq_values_n).sum())
    axes[2].plot(n_values, eq_values_n, label=f"mu={mu}, cost={cost}")
    axes[2].set_title("Equilibrium vs. n")
    axes[2].set_xlabel("Number of firms (n)")
    axes[2].set_ylabel("Equilibrium")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    print(f"Failed points: mu={mu_failures}, cost={cost_failures}, n={n_failures}")
    plt.show()

if __name__ == "__main__":
    # 20, 30, 10  Gaussian(500, 500)
    mu = 20
    cost = 30
    n = 10
    dist = GaussianDistribution(50, 1000)
    a, b = -math.inf, math.inf

    plot_equilibrium_single_variable(mu, cost, n, dist, a, b)