"""Turn a raw chain snapshot into a cleaned implied-vol dataset.

Steps (each is a common interview question, so keep them explicit):
  1. mid price from bid/ask, drop zero-bid or wide-spread quotes
  2. time to expiry in years (calendar days / 365)
  3. implied forward per expiry from put-call parity at the ATM strike
  4. use OTM options only (calls for K > F, puts for K < F) - most liquid, and
     avoids early-exercise contamination for American-style single stocks
  5. recompute IV ourselves (do not trust the vendor field)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bs


def clean_chain(raw: pd.DataFrame, r: float, q: float = 0.0,
                max_rel_spread: float = 0.25, min_T_days: int = 2) -> pd.DataFrame:
    df = raw.copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])]
    df = df[(df["ask"] - df["bid"]) / df["mid"] <= max_rel_spread]

    snap = pd.Timestamp(df["snapshot_utc"].iloc[0]).tz_localize(None).normalize()
    df["T"] = (df["expiry"] - snap).dt.days / 365.0
    df = df[df["T"] * 365 >= min_T_days]
    df["cpi"] = np.where(df["cp"] == "C", 1, -1)

    out = []
    for T, g in df.groupby("T"):
        F = _implied_forward(g, r, T)
        g = g.assign(F=F, k=np.log(g["strike"] / F))
        otm = ((g["cpi"] == 1) & (g["strike"] >= F)) | ((g["cpi"] == -1) & (g["strike"] < F))
        g = g[otm]
        g = g.assign(iv=bs.implied_vol(g["mid"].values, g["spot"].values, g["strike"].values,
                                       T, r, q, g["cpi"].values))
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    return res.dropna(subset=["iv"]).sort_values(["T", "strike"]).reset_index(drop=True)


def _implied_forward(g: pd.DataFrame, r: float, T: float) -> float:
    """F = K + e^{rT}(C - P) using the strike where |C - P| is smallest."""
    piv = g.pivot_table(index="strike", columns="cp", values="mid")
    piv = piv.dropna()
    if piv.empty:
        return float(g["spot"].iloc[0])
    K = piv.index[np.argmin(np.abs(piv["C"] - piv["P"]))]
    return float(K + np.exp(r * T) * (piv.loc[K, "C"] - piv.loc[K, "P"]))
