# Plot both prices to given decision variable

import json
import matplotlib.pyplot as plt

# Read JSON
with open("sensitivity_mu.json", "r") as f:
    data = json.load(f)

mu_vals = data["mu"]
p1_y = data["array1"]
p2_y = data["array2"]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(mu_vals, p1_y, 'o-', label=r'$p_1^*$ (Player 1)')
ax.plot(mu_vals, p2_y, 's-', label=r'$p_2^*$ (Player 2)')

ax.set_xlabel(r'Product Differentiation ($\mu$)')
ax.set_ylabel('Equilibrium Prices')
ax.set_title(r'Effect of $\mu$ on Prices')

ax.grid(True, ls='--')
ax.legend()

plt.tight_layout()
plt.savefig("sensitivity_mu.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 