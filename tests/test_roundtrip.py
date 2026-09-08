import numpy as np
import pandas as pd

from volsurface import bs, svi, heston


def test_bs_iv_roundtrip():
    S, r, q = 100.0, 0.04, 0.0
    K = np.linspace(60, 160, 41)
    T = np.array([0.05, 0.25, 1.0])[:, None]
    sig = 0.2 + 0.3 * (np.log(K / S)) ** 2
    cp = np.where(K > S, 1, -1)
    px = bs.price(S, K, T, r, q, sig, cp)
    iv = bs.implied_vol(px, S, K, T, r, q, cp)
    assert np.nanmax(np.abs(iv - sig)) < 1e-6


def test_heston_matches_bs_when_vol_flat():
    # xi -> 0, v0 = theta reduces Heston to BS with sigma = sqrt(theta)
    p = heston.HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=1e-4, rho=0.0)
    S, r, q, T = 100.0, 0.02, 0.01, 0.5
    K = np.array([80.0, 100.0, 120.0])
    h = heston.price(S, K, T, r, q, p)
    b = bs.price(S, K, T, r, q, 0.2)
    assert np.max(np.abs(h - b)) < 1e-3


def test_svi_and_heston_calibration_recover_surface():
    true = heston.HestonParams(v0=0.09, kappa=3.0, theta=0.06, xi=0.6, rho=-0.6)
    S, r, q = 100.0, 0.03, 0.0
    rows = []
    for T in (0.1, 0.25, 0.5, 1.0):
        F = S * np.exp((r - q) * T)
        K = F * np.exp(np.linspace(-0.4, 0.4, 21) * np.sqrt(T) / np.sqrt(0.25))
        cp = np.where(K > F, 1, -1)
        px = heston.price(S, K, T, r, q, true, cp)
        iv = bs.implied_vol(px, S, K, T, r, q, cp)
        rows.append(pd.DataFrame({"T": T, "K": K, "iv": iv, "cp": cp, "k": np.log(K / F)}))
    chain = pd.concat(rows, ignore_index=True).dropna()

    # SVI slice fits should be tight and butterfly-free
    for T, g in chain.groupby("T"):
        p = svi.fit_slice(g["k"].values, g["iv"].values, T)
        assert svi.rmse_iv(g["k"].values, g["iv"].values, T, p) < 2e-3
        assert svi.butterfly_arbitrage_free(p)

    # Heston recalibration from a wrong start should recover the truth
    fit = heston.calibrate(chain, S, r, q,
                           p0=heston.HestonParams(0.04, 1.0, 0.04, 0.3, -0.3))
    assert abs(fit.v0 - true.v0) < 5e-3
    assert abs(fit.rho - true.rho) < 0.05
    assert abs(np.sqrt(fit.theta) - np.sqrt(true.theta)) < 0.02
