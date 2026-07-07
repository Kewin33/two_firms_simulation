import torch
import matplotlib.pyplot as plt

class calcPrice:
    def __init__(self, mu, cost, n1, n2, dist1, dist2, integration_steps=100, num_std=5):
        self.mu = mu
        self.cost = cost
        self.n1 = n1
        self.n2 = n2        
        self.n = n1 + n2
        self.dist1 = dist1
        self.dist2 = dist2
        self.integration_steps = integration_steps
        self.num_std = num_std

    def _safe_cdf(self, dist, val):
        """Verhindert den ValueError bei beschränkten Verteilungen wie Uniform."""
        if hasattr(dist, 'low') and hasattr(dist, 'high'):
            return dist.cdf(torch.clamp(val, dist.low, dist.high))
        return dist.cdf(val)

    def _safe_pdf(self, dist, val):
        """Gibt 0 zurück, wenn außerhalb des Supports, statt einen Fehler zu werfen."""
        if hasattr(dist, 'low') and hasattr(dist, 'high'):
            within_support = (val >= dist.low) & (val <= dist.high)
            clamped = torch.clamp(val, dist.low, dist.high)
            pdf = torch.exp(dist.log_prob(clamped))
            return torch.where(within_support, pdf, torch.zeros_like(pdf))
        return torch.exp(dist.log_prob(val))

    def r(self, x1, x2, p1, p2, player=1):
        device = x1.device if isinstance(x1, torch.Tensor) else 'cpu'
        
        F1_x1 = self._safe_cdf(self.dist1, x1)
        F2_x2 = self._safe_cdf(self.dist2, x2)
        F_bar = (self.n1 / self.n) * F1_x1 + (self.n2 / self.n) * F2_x2

        if torch.abs(F_bar - 1.0) < 1e-6:
            geom_term = torch.tensor(float(self.n), device=device)
        else:
            geom_term = (1.0 - F_bar**self.n) / (1.0 - F_bar)

        if player == 1:
            dist_own, dist_opp = self.dist1, self.dist2
            n_own, n_opp = self.n1, self.n2
            x_own, p_own, p_opp = x1, p1, p2
        else:
            dist_own, dist_opp = self.dist2, self.dist1
            n_own, n_opp = self.n2, self.n1
            x_own, p_own, p_opp = x2, p2, p1

        f_own_x_own = self._safe_pdf(dist_own, x_own)
        std_own = dist_own.stddev if hasattr(dist_own, 'stddev') else torch.tensor(1.0, device=device)
        
        lower_bound = x_own - self.num_std * std_own
        if hasattr(dist_own, 'low'):
            lower_bound = torch.clamp(lower_bound, min=dist_own.low)
        
        t = torch.linspace(0.0, 1.0, steps=self.integration_steps, device=device)
        epsilon = lower_bound + (x_own - lower_bound) * t
        epsilon.requires_grad_(True)

        F_own_eps = self._safe_cdf(dist_own, epsilon)
        f_own_eps = self._safe_pdf(dist_own, epsilon)
        
        # --- HIER IST DIE KORREKTUR ---
        # Wenn die PDF keine grad_fn hat (weil sie konstant ist wie bei Uniform), ist die Ableitung 0
        if isinstance(dist_own, torch.distributions.Normal):
            f_own_prime = f_own_eps * (-(epsilon - dist_own.loc) / (dist_own.scale ** 2))
        elif f_own_eps.grad_fn is not None:
            f_own_prime = torch.autograd.grad(f_own_eps.sum(), epsilon, create_graph=True)[0]
        else:
            f_own_prime = torch.zeros_like(epsilon)

        f2_arg = epsilon - (p_own - p_opp) / self.mu
        F_opp_eps_arg = self._safe_cdf(dist_opp, f2_arg)

        # Integranden zusammensetzen
        shared_factor = (F_own_eps ** (n_own - 1)) * (F_opp_eps_arg ** n_opp)
        integrand_num = shared_factor * f_own_eps
        integrand_denom = f_own_prime * shared_factor

        int_num = torch.trapezoid(integrand_num, epsilon)
        int_denom = torch.trapezoid(integrand_denom, epsilon)

        numerator = (1.0 / self.n) * (1.0 - F_own_eps[-1]) * geom_term + int_num
        eps = 1e-6
        denominator = (f_own_x_own / self.n) * geom_term - int_denom
        stable_denominator = torch.where(denominator >= 0, 
                                        torch.clamp(denominator, min=eps), 
                                        torch.clamp(denominator, max=-eps))

        return self.mu * (numerator / stable_denominator)

    def s(self, x1, x2):
        device = x1.device if isinstance(x1, torch.Tensor) else 'cpu'
        std1 = self.dist1.stddev if hasattr(self.dist1, 'stddev') else torch.tensor(1.0, device=device)
        std2 = self.dist2.stddev if hasattr(self.dist2, 'stddev') else torch.tensor(1.0, device=device)

        b1 = x1 + self.num_std * std1
        if hasattr(self.dist1, 'high'):
            b1 = torch.clamp(b1, max=self.dist1.high)
            
        b2 = x2 + self.num_std * std2
        if hasattr(self.dist2, 'high'):
            b2 = torch.clamp(b2, max=self.dist2.high)

        t1 = torch.linspace(0.0, 1.0, steps=self.integration_steps, device=device)
        eps1 = x1 + (b1 - x1) * t1
        int1 = torch.trapezoid((eps1 - x1) * self._safe_pdf(self.dist1, eps1), eps1)

        t2 = torch.linspace(0.0, 1.0, steps=self.integration_steps, device=device)
        eps2 = x2 + (b2 - x2) * t2
        int2 = torch.trapezoid((eps2 - x2) * self._safe_pdf(self.dist2, eps2), eps2)

        return (self.n1 / self.n) * (self.mu * int1) + (self.n2 / self.n) * (self.mu * int2) - self.cost
    
    def optimize(self, init_x1=0.0, init_x2=0.0, init_p1=1.0, init_p2=1.0, epochs=500, lr=1):
        x1 = torch.tensor(init_x1, dtype=torch.float32, requires_grad=True)
        x2 = torch.tensor(init_x2, dtype=torch.float32, requires_grad=True)
        p1 = torch.tensor(init_p1, dtype=torch.float32, requires_grad=True)
        p2 = torch.tensor(init_p2, dtype=torch.float32, requires_grad=True)

        optimizer = torch.optim.Adam([x1, x2, p1, p2], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,      # Halbiert die LR
            patience=200,    # Wartet 100 Epochen ohne Verbesserung
            threshold=1e-4,  # Mindestverbesserung
            eps=1e-4,        # Verhindert zu kleine LR
        )
        
        penalty_weight = 10.0
        max_penalty = 10000.0

        # --- Early Stopping Variablen ---
        p1_prev, p2_prev = None, None
        stable_epochs = 0

        for epoch in range(epochs):
            optimizer.zero_grad()

            r1 = self.r(x1, x2, p1, p2, player=1)
            r2 = self.r(x1, x2, p1, p2, player=2)
            eq4 = self.s(x1, x2)
            eq3 = self.mu * (x1 - x2) - (p1 - p2)

            if torch.isnan(r1) or torch.isnan(r2) or torch.isnan(eq4):
                print(f"\n[Abbruch] Numerische Instabilität in Epoche {epoch}. Parameter wurden NaN.")
                print(f"Constraint Error: {constraint_loss.item():.2e}", f"p1*: {p1.item():.5f}, p2*: {p2.item():.5f}")
                break

            objective = -(p1 + p2)
            constraint_loss = (p1 - r1)**2 + (p2 - r2)**2 + (eq4)**2 + (eq3)**2
            total_loss = penalty_weight * constraint_loss + objective
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([x1, x2, p1, p2], max_norm=float(self.cost))
            optimizer.step()
            scheduler.step(constraint_loss.item())

            #with torch.no_grad():
            #    x1.clamp_(self.dist1.low, self.dist1.high)
            #    x2.clamp_(self.dist2.low, self.dist2.high)

            # --- Early Stopping Logik ---
            p1_round = round(p1.item(), 1)
            p2_round = round(p2.item(), 1)

            if p1_round == p1_prev and p2_round == p2_prev and constraint_loss.item() < 1e-3:
                stable_epochs += 1
            else:
                stable_epochs = max(0, stable_epochs - 1)  # Reduziert die Zählung, wenn sich die Preise ändern
                p1_prev, p2_prev = p1_round, p2_round

            if stable_epochs >= 400:
                print(f"Epoch {epoch:5d} | Total Loss: {total_loss.item():12.4f} | Constraint Error: {constraint_loss.item():8.6f} | p1*: {p1.item():.5f}, p2*: {p2.item():.5f}")
                print(f"\n[Early Stopping] Preise stabil auf 2 Nachkommastellen für 400 Iterationen in Epoche {epoch}.")

                break

            # Logging
            if epoch % 1000 == 0 or epoch == epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:5d} | Total Loss: {total_loss.item():12.4f} | Constraint Error: {constraint_loss.item():8.6f} | LR: {current_lr:.2e} | p1*: {p1.item():.5f}, p2*: {p2.item():.5f}")

            if epoch % 50 == 0 and epoch > 0:
                penalty_weight = min(penalty_weight * 1.5, max_penalty)

        return {
            "p1_star": p1.item(),
            "p2_star": p2.item(),
            "x1": x1.item(),
            "x2": x2.item(),
            "constraint_error": constraint_loss.item()
        }

def main():
    #torch.distributions.Distribution.set_default_validate_args(False)

    mu = 6.7
    cost = 1000
    n1 = 20
    n2 = 20
    dist1 = torch.distributions.Normal(50, 50)
    dist2 = torch.distributions.Normal(50, 50)

    calc_price = calcPrice(mu, cost, n1, n2, dist1, dist2)
    result = calc_price.optimize(init_x1=dist1.sample().item(), init_x2=dist2.sample().item(), init_p1=72100, init_p2=72100, epochs=200000, lr=2)

    print("\nFinales Optimierungsergebnis:")
    print(f"p1*: {result['p1_star']:.4f}")
    print(f"p2*: {result['p2_star']:.4f}")
    print(f"x1:  {result['x1']:.4f}")
    print(f"x2:  {result['x2']:.4f}")
    print(f"Constraint Error: {result['constraint_error']:.2e}")

if __name__ == "__main__":
    main()