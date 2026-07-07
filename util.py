from distribution import Distribution, GaussianDistribution, UniformDistribution
import math
from scipy import integrate

def calc_xhat(mu, cost, a, b, dist):
    if mu <= 0:
        raise ValueError("mu must be positive.")

    # If cost is above mu * (E[e] - a), the root lies left of a.
    # For distributions with support starting at a, f(x) = E[e] - x in that region.
    if cost > mu * (dist.mean() - a):
        print("high cost with params mu, cost, a, b:", mu, cost, a, b)
        return dist.mean() - cost / mu

    def f(x):
        return dist.excess_expectation(x) - cost / mu
    try:
        x_hat = find_root_bisection(f, a, b, dist=dist)
    except ValueError:
        # Robust fallback if the initial interval does not bracket the root.
        x_hat = find_root_bisection(f, -math.inf, b, dist=dist)
    return x_hat

def integral(func, a, b, epsabs=1e-8, epsrel=1e-8):
    res, err = integrate.quad(func, a, b, epsabs=epsabs, epsrel=epsrel)
    #print(f"Result: {res}, Error: {err}")
    return res

def find_root_bisection(f, a, b, tol=1e-8, max_iter=100, dist=None):
    """
    Find the root of a strictly monotonically decreasing function f(x) using the bisection method.

    Parameters:
    - f: Callable, the function f(x) to find the root for.
    - a: Float, the lower bound of the interval (can be -infinity).
    - b: Float, the upper bound of the interval.
    - tol: Float, the tolerance for the root (default: 1e-8).
    - max_iter: Int, the maximum number of iterations (default: 100).

    Returns:
    - Float, the root of the function within the given tolerance.

    Raises:
    - ValueError: If f(a) * f(b) >= 0, indicating the root is not bracketed.
    """
    import math

    if (math.isinf(a) or math.isinf(b)) and dist is not None:
        # Build a sensible finite start interval around the distribution center.
        try:
            center = float(dist.mean())
        except Exception:
            center = 0.0
        try:
            variance = float(dist.var())
            spread = max(1.0, 2.0 * math.sqrt(variance)) if variance >= 0 else 10.0
        except Exception:
            spread = 10.0

        center_value = f(center)
        if abs(center_value) < tol:
            return center

        if math.isinf(a) and math.isinf(b):
            if center_value > 0:
                a = center
                b = center + spread
            else:
                a = center - spread
                b = center
        elif math.isinf(a):
            a = center - spread
        elif math.isinf(b):
            b = center + spread
    else:
        if math.isinf(a):
            a = -1.0
        if math.isinf(b):
            b = 1.0

    fa = f(a)
    fb = f(b)

    # Expand only toward the side where the root can still lie.
    expansion_limit = 80
    expansions = 0
    while fa * fb > 0 and expansions < expansion_limit:
        if fa > 0 and fb > 0:
            step = max(1.0, abs(b))
            b += step
            fb = f(b)
        elif fa < 0 and fb < 0:
            step = max(1.0, abs(a))
            a -= step
            fa = f(a)
        else:
            break
        expansions += 1

    if fa * fb > 0:
        midpoint = 0.5 * (a + b)
        fm = f(midpoint)
        # If midpoint is effectively a root, return it to avoid a hard failure.
        if abs(fm) < tol:
            return midpoint
        raise ValueError("The function must have opposite signs at the interval endpoints.")

    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)

        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    raise RuntimeError("Maximum number of iterations reached without finding the root.")


def  main():
    #print("calc x_hat...")
    res = calc_xhat(0.5, 7, 0, 10, UniformDistribution(0, 10))
    print(f"x_hat: {res}")

if __name__ == "__main__":
    main()
    #res = integral(lambda x: -x**2+3*x, -100, 100)
    #print(f"Integral result: {res}")