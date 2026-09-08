"""Black-Scholes(-Merton) pricing, Greeks and implied volatility.

All functions are vectorised over numpy arrays. Conventions:
    S     spot
    K     strike
    T     time to expiry in years
    r     continuously compounded risk-free rate
    q     continuous dividend yield
    sigma annualised volatility
    cp    +1 for call, -1 for put
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EPS = 1e-12


def _d1d2(S, K, T, r, q, sigma):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    sqT = np.sqrt(np.maximum(T, _EPS))
    vol = np.maximum(sigma, _EPS) * sqT
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol
    return d1, d1 - vol


def price(S, K, T, r, q, sigma, cp=1):
    cp = np.asarray(cp)
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    return cp * (S * np.exp(-q * T) * norm.cdf(cp * d1) - K * np.exp(-r * T) * norm.cdf(cp * d2))


def delta(S, K, T, r, q, sigma, cp=1):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    return cp * np.exp(-q * T) * norm.cdf(cp * d1)


def gamma(S, K, T, r, q, sigma):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, q, sigma):
    """dPrice/dSigma (per unit of vol, not per 1%)."""
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, q, sigma, cp=1):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    first = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    second = -cp * r * K * np.exp(-r * T) * norm.cdf(cp * d2)
    third = cp * q * S * np.exp(-q * T) * norm.cdf(cp * d1)
    return first + second + third


def intrinsic_bounds(S, K, T, r, q, cp=1):
    """No-arbitrage lower and upper bounds for European prices."""
    F = S * np.exp((r - q) * T)
    df = np.exp(-r * T)
    lower = np.maximum(cp * df * (F - K), 0.0)
    upper = np.where(cp == 1, S * np.exp(-q * T), K * df)
    return lower, upper


def implied_vol(price_mkt, S, K, T, r, q, cp=1, tol=1e-8, max_iter=100):
    """Vectorised implied volatility.

    Newton iterations on vega with a bisection safeguard. Returns NaN where the
    market price violates no-arbitrage bounds or T <= 0.
    """
    price_mkt, S, K, T, cp = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price_mkt, S, K, T, cp))
    )
    lower, upper = intrinsic_bounds(S, K, T, r, q, cp)
    valid = (price_mkt > lower + 1e-10) & (price_mkt < upper - 1e-10) & (T > 0)

    lo = np.full(price_mkt.shape, 1e-4)
    hi = np.full(price_mkt.shape, 5.0)
    sigma = np.full(price_mkt.shape, 0.3)

    for _ in range(max_iter):
        p = price(S, K, T, r, q, sigma, cp)
        diff = p - price_mkt
        # maintain bracket
        hi = np.where(diff > 0, sigma, hi)
        lo = np.where(diff < 0, sigma, lo)
        v = vega(S, K, T, r, q, sigma)
        newton = sigma - diff / np.where(v > 1e-12, v, np.nan)
        # fall back to bisection if Newton leaves the bracket or is NaN
        bad = ~np.isfinite(newton) | (newton <= lo) | (newton >= hi)
        sigma_new = np.where(bad, 0.5 * (lo + hi), newton)
        converged = np.abs(sigma_new - sigma) < tol
        sigma = sigma_new
        if np.all(converged | ~valid):
            break

    return np.where(valid, sigma, np.nan)
