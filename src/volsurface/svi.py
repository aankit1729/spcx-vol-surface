"""Raw SVI (Gatheral 2004) slice parameterisation.

Total implied variance as a function of log-moneyness k = log(K/F):

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

Parameters: a, b >= 0, |rho| < 1, m, sigma > 0.
Reference: Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces" (2014).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self):
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])


def total_variance(k, p: SVIParams):
    k = np.asarray(k, dtype=float)
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2))


def implied_vol(k, T, p: SVIParams):
    return np.sqrt(np.maximum(total_variance(k, p), 1e-12) / T)


def _g(k, p: SVIParams):
    """Durrleman's g(k). g >= 0 everywhere <=> no butterfly arbitrage."""
    w = total_variance(k, p)
    dk = k - p.m
    root = np.sqrt(dk**2 + p.sigma**2)
    w1 = p.b * (p.rho + dk / root)
    w2 = p.b * p.sigma**2 / root**3
    return (1 - k * w1 / (2 * w)) ** 2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2


def butterfly_arbitrage_free(p: SVIParams, k_grid=None) -> bool:
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 601)
    return bool(np.all(_g(k_grid, p) >= -1e-10))


def calendar_arbitrage_free(p_short: SVIParams, p_long: SVIParams, k_grid=None) -> bool:
    """Total variance must be non-decreasing in T for every k."""
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 601)
    return bool(np.all(total_variance(k_grid, p_long) >= total_variance(k_grid, p_short) - 1e-10))


def fit_slice(k, iv, T, weights=None, penalise_butterfly=True) -> SVIParams:
    """Calibrate SVI to one expiry.

    k: log-moneyness log(K/F); iv: implied vols; T: expiry in years.
    Loss is weighted squared error in total variance, plus a soft penalty on
    butterfly arbitrage (negative g).
    """
    k = np.asarray(k, dtype=float)
    w_mkt = np.asarray(iv, dtype=float) ** 2 * T
    wts = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    wts = wts / wts.sum()
    k_pen = np.linspace(k.min() - 0.3, k.max() + 0.3, 200)

    def unpack(x):
        return SVIParams(a=x[0], b=x[1], rho=np.tanh(x[2]), m=x[3], sigma=np.exp(x[4]))

    def loss(x):
        p = unpack(x)
        err = np.sum(wts * (total_variance(k, p) - w_mkt) ** 2)
        if penalise_butterfly:
            g = _g(k_pen, p)
            err += 1e-2 * np.sum(np.minimum(g, 0.0) ** 2)
        # keep total variance positive at the minimum: a + b*sigma*sqrt(1-rho^2) >= 0
        min_w = p.a + p.b * p.sigma * np.sqrt(1 - p.rho**2)
        err += 1e2 * min(min_w, 0.0) ** 2
        return err

    # heuristic initial guess
    atm = w_mkt[np.argmin(np.abs(k))]
    x0 = np.array([0.5 * atm, 0.1, np.arctanh(-0.3), 0.0, np.log(0.1)])
    bounds = [(-1.0, None), (0.0, 10.0), (None, None), (-2.0, 2.0), (np.log(1e-3), np.log(2.0))]
    best = None
    for seed_b in (0.05, 0.2, 0.5):
        for seed_rho in (-0.7, -0.3, 0.3):
            for seed_sig in (0.05, 0.2):
                x = x0.copy()
                x[1], x[2], x[4] = seed_b, np.arctanh(seed_rho), np.log(seed_sig)
                res = minimize(loss, x, method="L-BFGS-B", bounds=bounds)
                if best is None or res.fun < best.fun:
                    best = res
    # polish with a derivative-free method
    res = minimize(loss, best.x, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 5000})
    return unpack(res.x if res.fun < best.fun else best.x)


def rmse_iv(k, iv, T, p: SVIParams) -> float:
    return float(np.sqrt(np.mean((implied_vol(k, T, p) - iv) ** 2)))
