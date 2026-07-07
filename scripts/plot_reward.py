# Plot both prices to given decision variable

import json
import matplotlib.pyplot as plt
from utils import calculate_demand
import torch
from pathlib import Path

n1 = 20
n2 = 20
cost = 3
dist1 = torch.distributions.Normal(50.0, 50.0)
dist2 = torch.distributions.Normal(80.0, 50.0)


# Read JSON
json_path = Path(__file__).resolve().with_name("sensitivity_mu.json")
with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

mu_vals = data["mu"]
p1_y = data["array1"]
p2_y = data["array2"]
x1_y = data["x1"]
x2_y = data["x2"]

fig, ax = plt.subplots(figsize=(8, 5))

reward1 = []
reward2 = []

for i in range(len(mu_vals)):
    dem1, dem2 = calculate_demand(dist1, dist2, n1, n2, mu_vals[i], x1_y[i], x2_y[i], p1_y[i], p2_y[i])
    reward1.append(dem1 * p1_y[i])
    reward2.append(dem2 * p2_y[i])

ax.plot(mu_vals, reward1, 'o-', label=r'$p_1^*$ (Player 1)')
ax.plot(mu_vals, reward2, 's-', label=r'$p_2^*$ (Player 2)')

ax.set_xlabel(r'Product Differentiation ($\mu$)')
ax.set_ylabel('Rewards')
ax.set_title(r'Effect of $\mu$ on Rewards')

ax.grid(True, ls='--')
ax.legend()

plt.tight_layout()
plt.savefig("rewards.png", dpi=300, bbox_inches="tight")
plt.close(fig)