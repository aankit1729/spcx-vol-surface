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
import pandas as pd
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

    cp = np.atleast_1d(np.asarray(cp)).ravel()
    if cp.size not in (1, K.size):
        raise ValueError(
            f"cp has {cp.size} entries but there are {K.size} strikes. "
            "A common cause is a duplicated 'cp' column in the input DataFrame, "
            "which makes df['cp'] return a 2-D frame instead of a vector."
        )
    put = call - S * np.exp(-q * T) + K * np.exp(-r * T)
    out = np.where(cp == 1, call, put)
    return out if out.size > 1 else float(out[0])


def subsample_surface(chain, max_expiries=10, k_max=0.6, min_days=5, max_days=400,
                      max_per_expiry=25, S=None, r=0.0, q=0.0):
    """Thin a chain down to a calibration set.

    Heston has five parameters; fitting them to 4,000 quotes is slow and mostly
    redundant, since neighbouring strikes carry almost the same information.
    Keeps liquid maturities, a sane moneyness band, and evenly spaced strikes.
    """
    df = chain.copy()
    df = df[(df["T"] * 365 >= min_days) & (df["T"] * 365 <= max_days)]
    if "k" not in df:
        F = S * np.exp((r - q) * df["T"])
        df["k"] = np.log(df["K"] / F)
    df = df[df["k"].abs() <= k_max]

    expiries = np.sort(df["T"].unique())
    if len(expiries) > max_expiries:
        idx = np.unique(np.linspace(0, len(expiries) - 1, max_expiries).round().astype(int))
        expiries = expiries[idx]
    df = df[df["T"].isin(expiries)]

    out = []
    for T, g in df.groupby("T"):
        g = g.sort_values("k")
        if len(g) > max_per_expiry:
            idx = np.unique(np.linspace(0, len(g) - 1, max_per_expiry).round().astype(int))
            g = g.iloc[idx]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def calibrate(chain, S, r, q, p0: HestonParams | None = None, weights=None,
              max_nfev=200, verbose=False, thin=True) -> HestonParams:
    """Calibrate to a DataFrame with columns [T, K, iv, cp].

    Residuals are price errors divided by market vega, which approximates the
    implied-vol error to first order while keeping the objective smooth. Inverting
    model prices to vols inside the loop is both slower and non-differentiable
    wherever the inversion fails, which stalls the optimiser.
    """
    if thin:
        chain = subsample_surface(chain, S=S, r=r, q=q)
    p0 = p0 or HestonParams(v0=0.09, kappa=2.0, theta=0.09, xi=0.5, rho=-0.5)

    # market prices and vegas, computed once
    T_a = chain["T"].values
    K_a = chain["K"].values
    cp_a = chain["cp"].values
    iv_a = chain["iv"].values
    px_mkt = bs.price(S, K_a, T_a, r, q, iv_a, cp_a)
    vega_a = np.maximum(bs.vega(S, K_a, T_a, r, q, iv_a), 1e-4)
    wts = np.ones(len(chain)) if weights is None else np.asarray(weights, dtype=float)
    scale = np.sqrt(wts) / vega_a

    groups = [(T, np.where(T_a == T)[0]) for T in np.sort(chain["T"].unique())]
    n_eval = [0]

    def resid(x):
        p = HestonParams(*x)
        out = np.empty_like(px_mkt)
        for T, idx in groups:
            out[idx] = price(S, K_a[idx], T, r, q, p, cp=cp_a[idx])
        n_eval[0] += 1
        res = (out - px_mkt) * scale
        if verbose and n_eval[0] % 25 == 0:
            print(f"    eval {n_eval[0]:4d}  rmse {np.sqrt(np.mean(res**2))*1e4:7.1f} bp")
        return res

    lb = [1e-4, 1e-2, 1e-4, 1e-2, -0.95]
    ub = [4.0, 20.0, 4.0, 5.0, 0.95]
    x0 = np.clip([p0.v0, p0.kappa, p0.theta, p0.xi, p0.rho], lb, ub)
    sol = least_squares(resid, x0, bounds=(lb, ub), xtol=1e-8, ftol=1e-10,
                        diff_step=1e-4, max_nfev=max_nfev)
    if verbose:
        print(f"    done: {n_eval[0]} evals, status {sol.status} ({sol.message.strip()})")
    return HestonParams(*sol.x)


def surface_rmse_bp(chain, S, r, q, p: HestonParams) -> float:
    """Model-vs-market RMSE in implied-vol basis points (for reporting, not fitting)."""
    err = []
    for T, g in chain.groupby("T"):
        px = price(S, g["K"].values, T, r, q, p, cp=g["cp"].values)
        iv = bs.implied_vol(px, S, g["K"].values, T, r, q, g["cp"].values)
        err.append(iv - g["iv"].values)
    err = np.concatenate(err)
    return float(np.sqrt(np.nanmean(err**2)) * 1e4)


def min_variance_delta(S, K, T, r, q, p: HestonParams, cp=1, dS=1e-3):
    """Numerical dPrice/dS holding v0 fixed (naive Heston delta).

    A minimum-variance hedge would additionally shift v0 by rho*xi*dS/S; see
    Bakshi, Cao & Chen (1997) and Hull & White (2017) for the argument.
    """
    h = S * dS
    up = price(S + h, K, T, r, q, p, cp)
    dn = price(S - h, K, T, r, q, p, cp)
    return (up - dn) / (2 * h)
