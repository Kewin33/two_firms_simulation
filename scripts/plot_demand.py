# Plot both prices to given decision variable

import json
import matplotlib.pyplot as plt
from utils import calculate_demand
import torch
from pathlib import Path

n1 = 20
n2 = 20
mu = 6.7
cost = 3
dist1 = torch.distributions.Normal(50.0, 50.0)
dist2 = torch.distributions.Normal(80.0, 50.0)


# Read JSON
json_path = Path(__file__).resolve().with_name("sensitivity_distribution_shift.json")
with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

mu_vals = data["shift"]
p1_y = data["array1"]
p2_y = data["array2"]
x1_y = data["x1"]
x2_y = data["x2"]

fig, ax = plt.subplots(figsize=(8, 5))

demand1 = []
demand2 = []

for i in range(len(mu_vals)):
    dem1, dem2 = calculate_demand(dist1, torch.distributions.Normal(50.0 + mu_vals[i], 50.0), n1, n2, mu, x1_y[i], x2_y[i], p1_y[i], p2_y[i])
    demand1.append(dem1)
    demand2.append(dem2)

ax.plot(mu_vals, demand1, 'o-', label=r'$d_1^*$ (Player 1)')
ax.plot(mu_vals, demand2, 's-', label=r'$d_2^*$ (Player 2)')

ax.set_xlabel(r'Product Differentiation ($\mu$)')
ax.set_ylabel('Demand')
ax.set_title(r'Effect of $\mu$ on Demand')

ax.grid(True, ls='--')
ax.legend()

plt.tight_layout()
plt.savefig("demands.png", dpi=300, bbox_inches="tight")
plt.close(fig)