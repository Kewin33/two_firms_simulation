import torch

def _calculate_stop_demand(dist, x, bar_F, n):
    """Berechnet die unmittelbare Nachfrage (Immediate Buyers / Stop Demand)."""
    # F(x) für die gegebene Verteilung bestimmen
    F_x = dist.cdf(torch.tensor(x, dtype=torch.float32))
    
    geom_sum = (1.0 - bar_F**n) / (1.0 - bar_F)
        
    return (1.0 / n) * (1.0 - F_x) * geom_sum


def _calculate_recall_demand(dist_target, dist_other, x_target, n_target, n_other, p_target, p_other, mu, steps=150):
    """Berechnet die Nachfrage der zurückkehrenden Konsumenten mittels numerischer Integration."""
    x_target_t = torch.tensor(x_target, dtype=torch.float32)
    
    lower_bound = dist_target.mean - 5 * dist_target.stddev

    if lower_bound >= x_target_t:
        return torch.tensor(0.0, device=x_target_t.device)

    # Integrationsgitter erstellen (von -unendlich/lower_bound bis x_i)
    epsilon = torch.linspace(lower_bound.item(), x_target_t.item(), steps=steps, device=x_target_t.device)
    
    # 1. Term: [F_i(eps)]^(n_i - 1)
    F_target = dist_target.cdf(epsilon)
    term1 = torch.pow(F_target, n_target - 1) if n_target > 1 else torch.ones_like(epsilon)
    
    # 2. Term: [F_j(eps - (p_i - p_j)/mu)]^n_j
    price_diff = (p_target - p_other) / mu
    F_other = dist_other.cdf(epsilon - price_diff)
    term2 = torch.pow(F_other, n_other) if n_other > 0 else torch.ones_like(epsilon)
    
    # Dichte f_i(eps)
    f_target = torch.exp(dist_target.log_prob(epsilon))
    
    # Gesamter Integrand
    integrand = term1 * term2 * f_target
    
    # Numerische Integration via Trapezoid-Regel
    demand_recall = torch.trapezoid(integrand, epsilon)
    return demand_recall


def calculate_demand(dist1, dist2, n1, n2, mu, x1, x2, p1, p2):
    """Hauptfunktion zur Berechnung der Gesamtnachfrage für Typ 1 und Typ 2."""
    n = n1 + n2
    
    # Konvertierung für die CDF-Berechnung auf Tensoren
    x1_t = torch.tensor(x1, dtype=torch.float32)
    x2_t = torch.tensor(x2, dtype=torch.float32)
    
    # Gewichtete durchschnittliche Verteilungsfunktion bar_F berechnen
    F1_x1 = dist1.cdf(x1_t)
    F2_x2 = dist2.cdf(x2_t)
    bar_F = (n1 / n) * F1_x1 + (n2 / n) * F2_x2
    
    # --- Typ 1 Nachfrage ---
    d1_stop = _calculate_stop_demand(dist1, x1, bar_F, n)
    d1_recall = _calculate_recall_demand(dist1, dist2, x1, n1, n2, p1, p2, mu)
    d1_total = d1_stop + d1_recall
    
    # --- Typ 2 Nachfrage (indizes gespiegelt) ---
    d2_stop = _calculate_stop_demand(dist2, x2, bar_F, n)
    d2_recall = _calculate_recall_demand(dist2, dist1, x2, n2, n1, p2, p1, mu)
    d2_total = d2_stop + d2_recall
    
    return d1_total, d2_total

def main():
    # Beispielhafte Parameter
    mu = 2
    cost = 0.2
    n1 = 2
    n2 = 2
    dist1 = torch.distributions.Uniform(0, 100)
    dist2 = torch.distributions.Uniform(30, 130)
    
    # Beispielhafte x und p Werte
    x1 = 95.31
    x2 = 96.38
    p1 = 43.11
    p2 = 45.22
    
    d1, d2 = calculate_demand(dist1, dist2, n1, n2, mu, x1, x2, p1, p2)
    
    print(f"Gesamtnachfrage Typ 1: {d1.item():.4f}, Reward: {d1.item() * p1:.4f}")
    print(f"Gesamtnachfrage Typ 2: {d2.item():.4f}, Reward: {d2.item() * p2:.4f}")

if __name__ == "__main__":
    main()