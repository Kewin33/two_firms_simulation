import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from quality_eq import calcPrice

# Globale Konfiguration für die Absicherung gegen NaN-Abstürze
torch.distributions.Distribution.set_default_validate_args(False)

# Gemeinsame Basiswerte für die Fixierung der anderen Parameter
BASE_MU = 6.7
BASE_COST = 3
BASE_N1 = 20
BASE_N2 = 20
EPOCHS = 40000  # Höhere Iterationen dank deines Early Stoppings
LR = 1

INIT_X1 = 10.0
INIT_X2 = 50.0
INIT_P1 = 100.0
INIT_P2 = 100.0

def get_base_dists(shift=0.0):
    """Hilfsfunktion für die beiden Normal-Verteilungen."""
    dist1 = torch.distributions.Normal(50.0, 50.0)
    dist2 = torch.distributions.Normal(80.0 + shift, 50.0)
    return dist1, dist2

def create_dense_linspace(low_start, low_end, high_end, num_low=5, num_high=5):
    """
    Erstellt zwei verbundene Linspaces.
    Der Bereich [low_start, low_end] wird dichter abgetastet.
    """
    low_part = np.linspace(low_start, low_end, num=num_low)
    high_part = np.linspace(low_end, high_end, num=num_high + 1)[1:] # Verhindert Doppelung des Trennpunkts
    return np.concatenate([low_part, high_part])


def plot_sensitivity_mu():
    print("Starte Analyse für mu...")
    mu_vals = np.geomspace(0.013, 2, num=10)
    p1_y, p2_y, x1_y, x2_y = [], [], [], []
    dist1, dist2 = get_base_dists()

    counter = 0
    last_p1, last_p2 = INIT_P1, INIT_P2
    last_x1, last_x2 = INIT_X1, INIT_X2

    for m in mu_vals:
        calc = calcPrice(float(m), BASE_COST, BASE_N1, BASE_N2, dist1, dist2)
        res = calc.optimize(init_x1=last_x1, init_x2=last_x2, init_p1=last_p1, init_p2=last_p2, epochs=EPOCHS, lr=LR)
        p1_y.append(res['p1_star'])
        p2_y.append(res['p2_star'])
        x1_y.append(res['x1'])
        x2_y.append(res['x2'])
        print(f"-> Fortschritt: {counter + 1}/{len(mu_vals)} | mu={m:.4f} | p1*: {res['p1_star']:.4f}, p2*: {res['p2_star']:.4f}")
        counter += 1
        last_p1, last_p2 = res['p1_star'], res['p2_star']
        last_x1, last_x2 = res['x1'], res['x2']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mu_vals, p1_y, 'o-', label='$p_1^*$ (Player 1)')
    ax.plot(mu_vals, p2_y, 's-', label='$p_2^*$ (Player 2)')
    ax.set_xlabel('Product Differentiation ($\mu$)')
    ax.set_ylabel('Equilibrium Prices')
    ax.set_title('Effect of $\mu$ on prices')
    ax.grid(True, ls="--")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('sensitivity_mu.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    daten = {"mu":mu_vals.tolist(),"array1": p1_y, "array2": p2_y, "x1": x1_y, "x2": x2_y}
    with open("sensitivity_mu.json", "w") as f:
        json.dump(daten, f)
        
    print("-> Daten gespeichert als 'sensitivity_mu.json'")
    print("-> Plot gespeichert als 'sensitivity_mu.png'")


def plot_sensitivity_cost():
    print("Starte Analyse für cost...")
    x = np.linspace(-np.pi / 2, np.pi / 2, num=10)
    x = np.sin(x)
    cost_vals = (x + 1) / 2 * 1000

    p1_y, p2_y, x1_y, x2_y = [], [], [], []
    dist1, dist2 = get_base_dists()
    counter = 0

    last_p1, last_p2 = INIT_P1, INIT_P2
    last_x1, last_x2 = INIT_X1, INIT_X2

    for c in cost_vals:
        calc = calcPrice(BASE_MU, float(c), BASE_N1, BASE_N2, dist1, dist2)
        res = calc.optimize(init_x1=last_x1, init_x2=last_x2, init_p1=last_p1, init_p2=last_p2, epochs=EPOCHS, lr=LR)
        p1_y.append(res['p1_star'])
        p2_y.append(res['p2_star'])
        x1_y.append(res['x1'])
        x2_y.append(res['x2'])
        print(f"-> Fortschritt: {counter + 1}/{len(cost_vals)} | cost={c:.4f} | p1*: {res['p1_star']:.4f}, p2*: {res['p2_star']:.4f}")
        counter += 1
        last_p1, last_p2 = res['p1_star'], res['p2_star']
        last_x1, last_x2 = res['x1'], res['x2']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_vals, p1_y, 'o-', label='$p_1^*$ (Player 1)')
    ax.plot(cost_vals, p2_y, 's-', label='$p_2^*$ (Player 2)')
    ax.set_xlabel('Search Cost (c)')
    ax.set_ylabel('Equilibrium Prices')
    ax.set_title('Effect of search cost on prices')
    ax.grid(True, ls="--")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('sensitivity_cost.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    daten = {"cost": cost_vals.tolist(), "array1": p1_y, "array2": p2_y, "x1": x1_y, "x2": x2_y}
    with open("sensitivity_cost.json", "w") as f:
        json.dump(daten, f)
        
    print("-> Daten gespeichert als 'sensitivity_cost.json'")
    print("-> Plot gespeichert als 'sensitivity_cost.png'")


def plot_sensitivity_market_size():
    print("Starte Analyse für n1 (Marktgröße)...")
    # n1 ist diskret, hier ist kein doppelter Linspace nötig
    n1_vals = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
    p1_y, p2_y, x1_y, x2_y = [], [], [], []
    dist1, dist2 = get_base_dists()
    counter = 0

    last_p1, last_p2 = INIT_P1, INIT_P2
    last_x1, last_x2 = INIT_X1, INIT_X2
    
    for n1_val in n1_vals:
        calc = calcPrice(BASE_MU, BASE_COST, n1_val, BASE_N2, dist1, dist2)
        res = calc.optimize(init_x1=last_x1, init_x2=last_x2, init_p1=last_p1, init_p2=last_p2, epochs=EPOCHS, lr=LR)
        p1_y.append(res['p1_star'])
        p2_y.append(res['p2_star'])
        x1_y.append(res['x1'])
        x2_y.append(res['x2'])
        print(f"-> Fortschritt: {counter + 1}/{len(n1_vals)} | n1={n1_val} | p1*: {res['p1_star']:.4f}, p2*: {res['p2_star']:.4f}")
        counter += 1
        last_p1, last_p2 = res['p1_star'], res['p2_star']
        last_x1, last_x2 = res['x1'], res['x2']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n1_vals, p1_y, 'o-', label='$p_1^*$ (Player 1)')
    ax.plot(n1_vals, p2_y, 's-', label='$p_2^*$ (Player 2)')
    ax.set_xlabel('Number of Player 1 ($n_1$) [Base $n_2=2$]')
    ax.set_ylabel('Equilibrium Prices')
    ax.set_title('Price Development with Asymmetric Market Size')
    ax.grid(True, ls="--")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('sensitivity_market_size.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    daten = {"n1": n1_vals.tolist(), "array1": p1_y, "array2": p2_y, "x1": x1_y, "x2": x2_y}
    with open("sensitivity_market_size.json", "w") as f:
        json.dump(daten, f)

    print("-> Daten gespeichert als 'sensitivity_market_size.json'")
    print("-> Plot gespeichert als 'sensitivity_market_size.png'")


def plot_sensitivity_distribution_shift():
    print("Starte Analyse für den Verteilungs-Abstand...")
    # Da der Abstand negativ und positiv sein kann, tasten wir um die Null herum dichter ab
    # Bereich von 0 bis 10 (dicht) und 60 bis 80 (weiter gefasst)
    shift_vals = [-150, -100, -50, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 50, 100, 150]
    p1_y, p2_y, x1_y, x2_y = [], [], [], []
    counter = 0
    last_p1, last_p2 = INIT_P1, INIT_P2
    last_x1, last_x2 = INIT_X1, INIT_X2

    for shift in shift_vals:
        d1, d2 = get_base_dists(shift)
        calc = calcPrice(BASE_MU, BASE_COST, BASE_N1, BASE_N2, d1, d2)
        # Dynamische Anpassung des Startwerts für x2 basierend auf dem Shift
        res = calc.optimize(init_x1=last_x1, init_x2=last_x2, init_p1=last_p1, init_p2=last_p2, epochs=EPOCHS, lr=LR)
        p1_y.append(res['p1_star'])
        p2_y.append(res['p2_star'])
        x1_y.append(res['x1'])
        x2_y.append(res['x2'])
        print(f"-> Fortschritt: {counter + 1}/{len(shift_vals)} | Shift={shift:.4f} | p1*: {res['p1_star']:.4f}, p2*: {res['p2_star']:.4f}")
        counter += 1
        last_p1, last_p2 = res['p1_star'], res['p2_star']
        last_x1, last_x2 = res['x1'], res['x2']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shift_vals, p1_y, 'o-', label='$p_1^*$ (Player 1)')
    ax.plot(shift_vals, p2_y, 's-', label='$p_2^*$ (Player 2)')
    ax.set_xlabel('Additional Shift of Distribution 2')
    ax.set_ylabel('Equilibrium Prices')
    ax.set_title('Price Development in Dependency of the Distribution Shift')
    ax.grid(True, ls="--")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('sensitivity_distribution_shift.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    daten = {"shift": shift_vals, "array1": p1_y, "array2": p2_y, "x1": x1_y, "x2": x2_y}
    with open("sensitivity_distribution_shift.json", "w") as f:
        json.dump(daten, f)
        
    print("-> Daten gespeichert als 'sensitivity_distribution_shift.json'")
    print("-> Plot gespeichert als 'sensitivity_distribution_shift.png'")


if __name__ == "__main__":
    # Führt alle Analysen nacheinander aus
    #plot_sensitivity_mu()
    # plot_sensitivity_cost()
    #plot_sensitivity_market_size()
    plot_sensitivity_distribution_shift()
    print("\nAlle Sensitivitätsanalysen erfolgreich abgeschlossen!")