from single_firm.distribution import GaussianDistribution, UniformDistribution
from single_firm.util import calc_xhat, integral
import math


def calculate_equilibrium(mu, cost, n, dist, a, b):
    xhat = calc_xhat(mu, cost, a, b, dist)
    # x_eff = min(b, xhat)
    # fx = dist.cdf(x_eff)
    # lower_bound = max(a, xhat)
    ## Stable evaluation of (1 - fx**n) / (1 - fx) when fx is close to 1.
    # geom_sum = float(n) if math.isclose(fx, 1.0, rel_tol=0.0, abs_tol=1e-12) else (1 - fx**n) / (1 - fx)
    #denom = geom_sum * dist.pdf(xhat) - n * integral(
    #    lambda e: dist.derivative_pdf(e) * dist.cdf(e)**(n-1), lower_bound, x_eff
    #)
    if xhat < a:
        print(f"xhat={xhat} below a: Diamond model with infty price")
        denom = 0
    else: 
        denom = (1 - dist.cdf(xhat)**n) / (1 - dist.cdf(xhat)) * dist.pdf(xhat) - n * integral(
            lambda e: dist.derivative_pdf(e) * dist.cdf(e)**(n-1), a, xhat
        )
    if not math.isfinite(denom) or abs(denom) < 1e-12:
        return math.inf
    return mu / denom


def main():
    print("Calculating equilibrium...")
    res = calculate_equilibrium(6.7, 1000, 40, UniformDistribution(0, 100), 0, 100)
    xhat = calc_xhat(6.7, 1000, -300, 350, UniformDistribution(0, 100))
    print(f"Equilibrium: {res}, xhat: {xhat}")

if __name__ == "__main__":
    main()