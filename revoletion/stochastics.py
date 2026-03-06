import math
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn as sk


@dataclass
class DistanceDistribution:
    mu: float
    sigma: float

    @classmethod
    def from_mu_sigma(cls, mu: float, sigma: float) -> Self:
        return cls(mu=mu, sigma=sigma)

    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> Self:
        return cls(
            mu=math.log(mean**2 / math.sqrt(std**2 + mean**2)),
            sigma=math.sqrt(math.log(1 + std**2 / mean**2)),
        )

    @classmethod
    def from_mean_mode(cls, mean: float, mode: float) -> Self:
        if mode <= 0 or mean <= 0:
            raise ValueError("mode and mean must be > 0")
        if mean <= mode:
            raise ValueError("mean must be greater than mode for a valid lognormal")

        sigma2 = (2.0 / 3.0) * math.log(mean / mode)
        return cls(mu=math.log(mode) + sigma2, sigma=math.sqrt(sigma2))

    @classmethod
    def from_data(cls, data) -> Self:
        shape, _, scale = stats.lognorm.fit(data, floc=0)
        return cls.from_mu_sigma(mu=np.log(scale), sigma=shape)

    @property
    def mode(self) -> float:
        return math.exp(self.mu - self.sigma**2)

    @property
    def mean(self) -> float:
        return math.exp(self.mu + 0.5 * self.sigma**2)

    @property
    def std(self) -> float:
        return math.sqrt((math.exp(self.sigma**2) - 1) * math.exp(2 * self.mu + self.sigma**2))

    @property
    def variance(self) -> float:
        return self.std**2

    def sample(self, size):
        return stats.lognorm.rvs(s=self.sigma, loc=0, scale=np.exp(self.mu), size=size)

    def pdf(self, xvals):
        return stats.lognorm.pdf(x=xvals, s=self.sigma, loc=0, scale=np.exp(self.mu))

    def cdf(self, xvals):
        return stats.lognorm.cdf(x=xvals, s=self.sigma, loc=0, scale=np.exp(self.mu))


@dataclass
class DepartureDistribution:
    model: sk.mixture.GaussianMixture

    @classmethod
    def from_mean_std(
        cls,
        weight1: float,
        weight2: float,
        mean1: float,
        mean2: float,
        std1: float,
        std2: float,
    ) -> Self:
        model = sk.mixture.GaussianMixture(n_components=2, covariance_type="full")
        model.weights_ = np.array([weight1, weight2])
        model.means_ = np.array([mean1, mean2]).reshape(-1, 1)
        model.covariances_ = (np.array([std1, std2]) ** 2).reshape(-1, 1, 1)
        model.precisions_cholesky_ = 1.0 / np.sqrt(model.covariances_)
        return cls(model=model)

    @classmethod
    def from_data(cls, data: pd.Series) -> Self:
        return cls(model=sk.mixture.GaussianMixture(n_components=2).fit(data.to_frame()))

    @property
    def weights(self):
        return self.model.weights_

    @property
    def means(self):
        return self.model.means_.ravel()

    @property
    def stds(self):
        return np.sqrt(self.model.covariances_.ravel())

    def sample(self, size):
        samples, _ = self.model.sample(n_samples=size)
        return samples.ravel()

    def pdf(self, xvals):
        return np.exp(self.model.score_samples(xvals.reshape(-1, 1)))

    def cdf(self, xvals):
        cdf = np.zeros_like(xvals)
        for w, m, s in zip(self.weights, self.means, self.stds):
            cdf += w * stats.norm.cdf(xvals, loc=m, scale=s)
        return cdf


@dataclass
class SpeedDistribution:
    mean: float
    std: float

    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> Self:
        return cls(mean=mean, std=std)

    @classmethod
    def from_data(cls, data: pd.Series) -> Self:
        mean, std = stats.norm.fit(data)
        return cls(mean=mean, std=std)

    def sample(self, size):
        return stats.norm.cdf(loc=self.mean, scale=self.std, size=size)

    def pdf(self, xvals):
        return stats.norm.pdf(x=xvals, loc=self.mean, scale=self.std)

    def cdf(self, xvals):
        return stats.norm.cdf(x=xvals, loc=self.mean, scale=self.std)


@dataclass
class IdleDistribution:
    p0: float
    a: float
    c: float
    scale: float

    @classmethod
    def from_data(cls, data: pd.Series) -> Self:
        p0 = (data == 0).mean()
        a, c, _, scale = stats.gengamma.fit(data.replace(to_replace=0, value=np.nan).dropna(), floc=0)
        return cls(p0=p0, a=a, c=c, scale=scale)

    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> Self:
        """
        This method assumes the idle time to be (a) nonzero and (b) exponentially decaying to fit a non-generalized gamma
        distribution (p0=0, c=1).
        """
        if mean <= 0 or std <= 0:
            raise ValueError("mean and std must be positive")

        shape = (mean / std) ** 2
        scale = (std**2) / mean

        return cls(p0=0.0, a=shape, c=1.0, scale=scale)

    @classmethod
    def from_mode_std(cls, mode: float, std: float) -> Self:
        """
        This method assumes the idle time to be (a) nonzero and (b) exponentially decaying to fit a non-generalized gamma
        distribution (p0=0, c=1).
        """

        if mode <= 0 or std <= 0:
            raise ValueError("mode and std must be positive")

        A = std**2
        B = -2 * std**2 - mode**2
        C = std**2

        discriminant = B**2 - 4 * A * C

        if discriminant < 0:
            raise ValueError("No valid gamma distribution for given mode/std")

        shape1 = (-B + np.sqrt(discriminant)) / (2 * A)
        shape2 = (-B - np.sqrt(discriminant)) / (2 * A)
        shape = max(shape1, shape2)

        if shape <= 1:
            raise ValueError("No gamma distribution with mode>0 exists (shape<=1)")

        scale = mode / (shape - 1)

        return cls(p0=0.0, a=shape, c=1.0, scale=scale)

    def sample(self, size):
        # choose which samples are zero
        is_zero = np.random.rand(size) < self.p0

        # Draw from gengamma for the non-zero ones
        samples = np.zeros(size)
        n_nonzero = (~is_zero).sum()

        if n_nonzero > 0:
            samples[~is_zero] = stats.gengamma.rvs(a=self.a, c=self.c, loc=0, scale=self.scale, size=n_nonzero)

        return samples

    def pdf(self, xvals):
        """
        PDF of the mixed distribution:
        - At x=0: a point mass of p0 (Dirac delta, not representable as a density)
        - For x>0: (1-p0) * gengamma.pdf(...)

        For numeric arrays, return:
        - p0 at x == 0
        - (1 - p0)*pdf for x > 0
        """
        x = np.asarray(xvals)
        out = np.zeros_like(x, dtype=float)

        # Continuous density for x > 0
        mask = x > 0
        out[mask] = (1 - self.p0) * stats.gengamma.pdf(x=x[mask], a=self.a, c=self.c, loc=0, scale=self.scale)

        # x == 0: return probability mass p0
        out[x == 0] = self.p0

        return out

    def cdf(self, xvals):
        x = np.asarray(xvals)
        out = np.zeros_like(x, dtype=float)

        neg = x < 0
        out[neg] = 0.0

        pos = ~neg
        out[pos] = self.p0 + (1 - self.p0) * stats.gengamma.cdf(x=x[pos], a=self.a, c=self.c, loc=0, scale=self.scale)
        return out
