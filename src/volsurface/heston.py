"""Heston (1993) stochastic volatility model.

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1,W2> = rho dt

Pricing via the COS method (Fang & Oosterlee 2008), which is fast and stable
enough to sit inside a calibration loop. Characteristic function uses the
"little trap" form of Albrecher et al. (2007) to avoid branch-cut issues.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from . import bs


@dataclass
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def feller(self) -> float:
        """2*kappa*theta - xi^2; positive means the Feller condition holds."""
        return 2 * self.kappa * self.theta - self.xi**2


def char_func(u, T, r, q, p: HestonParams):
    """Characteristic function of log(S_T/S_0) (drift included)."""
    u = np.asarray(u, dtype=complex)
    a = p.kappa * p.theta
    b = p.kappa - 1j * p.rho * p.xi * u
    d = np.sqrt(b**2 + p.xi**2 * (u**2 + 1j * u))
    g = (b - d) / (b + d)
    exp_dT = np.exp(-d * T)
    C = (r - q) * 1j * u * T + a / p.xi**2 * (
        (b - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = (b - d) / p.xi**2 * (1 - exp_dT) / (1 - g * exp_dT)
    return np.exp(C + D * p.v0)


def _cos_coeffs_call(a, b, c, d, k):
    """Chi and psi coefficients for a call payoff on [c, d] within [a, b]."""
    omega = k * np.pi / (b - a)
    chi = (
        np.cos(omega * (d - a)) * np.exp(d)
        - np.cos(omega * (c - a)) * np.exp(c)
        + omega * np.sin(omega * (d - a)) * np.exp(d)
        - omega * np.sin(omega * (c - a)) * np.exp(c)
    ) / (1 + omega**2)
    psi = np.where(
        k == 0,
        d - c,
        (np.sin(omega * (d - a)) - np.sin(omega * (c - a))) / np.where(k == 0, 1, omega),
    )
    return chi, psi


def price(S, K, T, r, q, p: HestonParams, cp=1, N=256, L=12.0):
    """European option price(s) by the COS method for a single expiry T.

    K may be an array; cp is applied via put-call parity.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    x = np.log(S / K)  # log(S/K) per strike

    # truncation range from cumulants (approximate, standard choice)
    c1 = (r - q) * T + (1 - np.exp(-p.kappa * T)) * (p.theta - p.v0) / (2 * p.kappa) - 0.5 * p.theta * T
    c2 = (
        1 / (8 * p.kappa**3)
        * (
            p.xi * T * p.kappa * np.exp(-p.kappa * T) * (p.v0 - p.theta) * (8 * p.kappa * p.rho - 4 * p.xi)
            + p.kappa * p.rho * p.xi * (1 - np.exp(-p.kappa * T)) * (16 * p.theta - 8 * p.v0)
            + 2 * p.theta * p.kappa * T * (-4 * p.kappa * p.rho * p.xi + p.xi**2 + 4 * p.kappa**2)
            + p.xi**2 * ((p.theta - 2 * p.v0) * np.exp(-2 * p.kappa * T) + p.theta * (6 * np.exp(-p.kappa * T) - 7) + 2 * p.v0)
            + 8 * p.kappa**2 * (p.v0 - p.theta) * (1 - np.exp(-p.kappa * T))
        )
    )
    c2 = max(c2, 1e-8)
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)

    k = np.arange(N)
    omega = k * np.pi / (b - a)
    chi, psi = _cos_coeffs_call(a, b, 0.0, b, k)
    V = 2.0 / (b - a) * (chi - psi)  # call payoff coefficients for strike-normalised payoff
    V[0] *= 0.5

    phi = char_func(omega, T, r, q, p)  # shape (N,)
    # sum over k for each strike: Re[ phi_k * exp(i*omega_k*(x - a)) ] * V_k
    expo = np.exp(1j * np.outer(x, omega) - 1j * omega * a)  # (nK, N)
    call = K * np.exp(-r * T) * np.real(expo @ (phi * V))

    cp = np.asarray(cp)
    put = call - S * np.exp(-q * T) + K * np.exp(-r * T)
    out = np.where(cp == 1, call, put)
    return out if out.size > 1 else float(out[0])


def calibrate(chain, S, r, q, p0: HestonParams | None = None, weights=None) -> HestonParams:
    """Calibrate to a DataFrame with columns [T, K, iv, cp] by matching implied vols.

    Matching in vol space (not price) equalises the influence of cheap OTM
    options versus expensive ITM ones.
    """
    p0 = p0 or HestonParams(v0=0.09, kappa=2.0, theta=0.09, xi=0.5, rho=-0.5)
    wts = np.ones(len(chain)) if weights is None else np.asarray(weights, dtype=float)
    groups = [(T, g) for T, g in chain.groupby("T")]

    def unpack(x):
        return HestonParams(*x)

    def resid(x):
        p = unpack(x)
        res = []
        for T, g in groups:
            model_px = price(S, g["K"].values, T, r, q, p, cp=g["cp"].values)
            model_iv = bs.implied_vol(model_px, S, g["K"].values, T, r, q, g["cp"].values)
            model_iv = np.where(np.isfinite(model_iv), model_iv, 5.0)
            res.append(model_iv - g["iv"].values)
        return np.concatenate(res) * np.sqrt(wts)

    lb = [1e-4, 1e-3, 1e-4, 1e-3, -0.999]
    ub = [4.0, 20.0, 4.0, 5.0, 0.999]
    sol = least_squares(resid, p0.as_array() if hasattr(p0, "as_array") else
                        [p0.v0, p0.kappa, p0.theta, p0.xi, p0.rho], bounds=(lb, ub), xtol=1e-8)
    return unpack(sol.x)


def min_variance_delta(S, K, T, r, q, p: HestonParams, cp=1, dS=1e-3):
    """Numerical dPrice/dS holding v0 fixed (naive Heston delta).

    A minimum-variance hedge would additionally shift v0 by rho*xi*dS/S; see
    Bakshi, Cao & Chen (1997) and Hull & White (2017) for the argument.
    """
    h = S * dS
    up = price(S + h, K, T, r, q, p, cp)
    dn = price(S - h, K, T, r, q, p, cp)
    return (up - dn) / (2 * h)
