"""Distribution classes.

Provides a simple `Distribution` base class and two implementations:
- `UniformDistribution`
- `GaussianDistribution` (normal)

Each class implements `pdf`, `cdf`, `sample`, `mean` and `var`.
"""
from __future__ import annotations

import math
import random
from typing import List, Union

from scipy import integrate

Number = Union[int, float]


class Distribution:
	def pdf(self, x: Number) -> float:
		raise NotImplementedError

	def cdf(self, x: Number) -> float:
		raise NotImplementedError

	def sample(self, n: int = 1):
		raise NotImplementedError

	def mean(self) -> float:
		raise NotImplementedError

	def var(self) -> float:
		raise NotImplementedError

	def excess_expectation(self, x: Number) -> float:
		"""Return E[(X - x)^+] for the distribution."""
		res, _ = integrate.quad(lambda t: 1.0 - self.cdf(t), float(x), math.inf, epsabs=1e-8, epsrel=1e-8)
		return res

	def derivative_pdf(self, x: Number, h: float = 1e-5) -> float:
		"""
		Numerically compute the derivative of the PDF at x using central difference.

		Parameters:
		- x: The point at which to compute the derivative.
		- h: The step size for the finite difference (default: 1e-5).

		Returns:
		- The numerical derivative of the PDF at x.
		"""
		return (self.pdf(x + h) - self.pdf(x - h)) / (2 * h)


class UniformDistribution(Distribution):
	"""Uniform distribution on [a, b].

	pdf(x) = 1/(b-a) for x in [a, b], else 0.
	cdf(x) = 0 for x<a, (x-a)/(b-a) for a<=x<b, 1 for x>=b.
	"""

	def __init__(self, a: float = 0.0, b: float = 1.0):
		if b <= a:
			raise ValueError("b must be greater than a")
		self.a = float(a)
		self.b = float(b)
		self._width = self.b - self.a

	def pdf(self, x: Number) -> float:
		x = float(x)
		return 1.0 / self._width if self.a <= x <= self.b else 0.0

	def cdf(self, x: Number) -> float:
		x = float(x)
		if x < self.a:
			return 0.0
		if x >= self.b:
			return 1.0
		return (x - self.a) / self._width

	def sample(self, n: int = 1):
		if n == 1:
			return random.uniform(self.a, self.b)
		return [random.uniform(self.a, self.b) for _ in range(n)]

	def mean(self) -> float:
		return 0.5 * (self.a + self.b)

	def var(self) -> float:
		return (self._width ** 2) / 12.0

	def derivative_pdf(self, x: Number, h: float = 1e-5) -> float:
		# Away from boundary points, the uniform PDF is constant.
		return 0.0

	def excess_expectation(self, x: Number) -> float:
		x = float(x)
		if x <= self.a:
			return self.mean() - x
		if x >= self.b:
			return 0.0
		return ((self.b - x) ** 2) / (2.0 * self._width)

	def __repr__(self) -> str:  # pragma: no cover - convenience
		return f"UniformDistribution(a={self.a}, b={self.b})"


class GaussianDistribution(Distribution):
	"""Gaussian (normal) distribution with mean `mu` and stddev `sigma`.

	Uses math.erf for the CDF and random.gauss for sampling.
	"""

	def __init__(self, mu: float = 0.0, sigma: float = 1.0):
		if sigma <= 0:
			raise ValueError("sigma must be positive")
		self.mu = float(mu)
		self.sigma = float(sigma)
		self._coeff = 1.0 / (math.sqrt(2.0 * math.pi) * self.sigma)

	def pdf(self, x: Number) -> float:
		x = float(x)
		z = (x - self.mu) / self.sigma
		return self._coeff * math.exp(-0.5 * z * z)

	def cdf(self, x: Number) -> float:
		x = float(x)
		return 0.5 * (1.0 + math.erf((x - self.mu) / (self.sigma * math.sqrt(2.0))))

	def sample(self, n: int = 1):
		if n == 1:
			return random.gauss(self.mu, self.sigma)
		return [random.gauss(self.mu, self.sigma) for _ in range(n)]

	def mean(self) -> float:
		return self.mu

	def var(self) -> float:
		return self.sigma ** 2

	def derivative_pdf(self, x: Number, h: float = 1e-5) -> float:
		x = float(x)
		return -((x - self.mu) / (self.sigma ** 2)) * self.pdf(x)

	def excess_expectation(self, x: Number) -> float:
		x = float(x)
		z = (x - self.mu) / self.sigma
		phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
		upper_tail = 0.5 * math.erfc(z / math.sqrt(2.0))
		return self.sigma * phi + (self.mu - x) * upper_tail

	def __repr__(self) -> str:  # pragma: no cover - convenience
		return f"GaussianDistribution(mu={self.mu}, sigma={self.sigma})"


__all__ = ["Distribution", "UniformDistribution", "GaussianDistribution"]

