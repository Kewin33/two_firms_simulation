import json
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from utils import calculate_demand

# Globale Basiswerte (identisch mit deinem Generierungsskript)
BASE_MU = 0.1 #6.7
BASE_COST = 3
BASE_N1 = 20
BASE_N2 = 20
DIST1 = torch.distributions.Normal(50.0, 50.0)
'''
# Konfiguration für alle Sensitivitätsanalysen
PLOTS_CONFIG = [
    {
        "json_file": "sensitivity_mu.json",
        "x_key": "mu",
        "x_label": r"Product Differentiation ($\mu$)",
        "out_suffix": "mu"
    },
    {
        "json_file": "sensitivity_cost.json",
        "x_key": "cost",
        "x_label": "Search Cost (c)",
        "out_suffix": "cost"
    },
    {
        "json_file": "sensitivity_market_size_n1.json",
        "x_key": "n1",
        "x_label": r"Market Size Player 1 ($n_1$)",
        "out_suffix": "market_size_n1"
    },
    {
        "json_file": "sensitivity_market_size_n2.json",
        "x_key": "n2",
        "x_label": r"Market Size Player 2 ($n_2$)",
        "out_suffix": "market_size_n2"
    },
    {
        "json_file": "sensitivity_distribution_shift.json",
        "x_key": "shift",
        "x_label": "Additional Shift of Distribution 2",
        "out_suffix": "distribution_shift"
    }
]
'''

PLOTS_CONFIG = [
    {
        "json_file": "sensitivity_cost.json",
        "x_key": "cost",
        "x_label": "Search Cost (c)",
        "out_suffix": "cost"
    }
]

def process_and_plot():
    base_path = Path(__file__).resolve().parent
    
    for cfg in PLOTS_CONFIG:
        json_path = base_path / cfg["json_file"]
        print(f"Verarbeite Datei: {json_path}")
        if not json_path.exists():
            print(f"Überspringe {cfg['json_file']}, da die Datei nicht existiert.")
            continue
            
        print(f"Verarbeite {cfg['json_file']}...")
        
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        x_vals = data[cfg["x_key"]]
        p1_y = data["array1"]
        p2_y = data["array2"]
        x1_y = data["x1"]
        x2_y = data["x2"]
        
        demand1, demand2 = [], []
        reward1, reward2 = [], []
        
        # Berechnungen für jeden Punkt auf der X-Achse
        for i in range(len(x_vals)):
            # Dynamische Parameterauflösung: Wenn der Wert im JSON variiert, nimm ihn, sonst Basiswert
            mu = data["mu"][i] if "mu" in data else BASE_MU
            n1 = data["n1"][i] if "n1" in data else BASE_N1
            n2 = data["n2"][i] if "n2" in data else BASE_N2
            
            # Verteilungen bestimmen (Shift ist bei allen außer 'shift' fix auf 30 gesetzt)
            shift = data["shift"][i] if "shift" in data else 30.0
            dist2 = torch.distributions.Normal(50.0 + shift, 50.0)
            
            # Demand berechnen
            dem1, dem2 = calculate_demand(
                DIST1, dist2, n1, n2, mu, 
                x1_y[i], x2_y[i], p1_y[i], p2_y[i]
            )
            
            demand1.append(dem1)
            demand2.append(dem2)
            
            # Reward berechnen (Demand * Preis)
            reward1.append(dem1 * p1_y[i])
            reward2.append(dem2 * p2_y[i])
            
        # --- 1. Demand Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_vals, demand1, 'o-', label=r'$d_1^*$ (Player 1)')
        ax.plot(x_vals, demand2, 's-', label=r'$d_2^*$ (Player 2)')
        ax.set_xlabel(cfg["x_label"])
        ax.set_ylabel('Demand')
        ax.set_title(f'Effect of Variable on Demand')
        ax.grid(True, ls='--')
        ax.legend()
        plt.tight_layout()
        plt.savefig(base_path / f"demands_{cfg['out_suffix']}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # --- 2. Reward Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_vals, reward1, 'o-', label=r'$R_1^*$ (Player 1)')
        ax.plot(x_vals, reward2, 's-', label=r'$R_2^*$ (Player 2)')
        ax.set_xlabel(cfg["x_label"])
        ax.set_ylabel('Rewards')
        ax.set_title(f'Effect of Variable on Rewards')
        ax.grid(True, ls='--')
        ax.legend()
        plt.tight_layout()
        plt.savefig(base_path / f"rewards_{cfg['out_suffix']}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"-> Plots gespeichert: demands_{cfg['out_suffix']}.png & rewards_{cfg['out_suffix']}.png")

if __name__ == "__main__":
    process_and_plot()
    print("\nAlle Plots erfolgreich generiert!")