"""Turn a raw chain snapshot into a cleaned implied-vol dataset.

Every filter here exists because of a specific failure mode observed on real
quotes. Pass report=True to get a table of how many quotes each filter removed;
worth checking whenever the data source changes.

Filters, in order:
  1. two-sided quote with a positive bid
  2. absolute price floor - a 4-cent option is one tick of noise, not a price
  3. relative spread cap
  4. liquidity: some open interest, or traded today
  5. time to expiry measured from the *session* close to the *expiry* close
  6. forward from put-call parity, per expiry
  7. OTM only (calls above F, puts below): liquid, and avoids early exercise
  8. standardised moneyness |k| / (sigma*sqrt(T)) cap - a fixed log-moneyness
     band is wrong, since the same k is far deeper OTM at 2 days than at 2 years
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bs

# US equity options expire at the 16:00 New York close (20:00 UTC in EDT).
_CLOSE_UTC_HOUR = 20


def session_date(snapshot_utc) -> pd.Timestamp:
    """The trading session a snapshot belongs to.

    A job running at 00:05 UTC captures the *previous* New York session. Getting
    this wrong shifts T by a full day, which badly distorts weekly expiries.
    """
    ts = pd.Timestamp(snapshot_utc)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return pd.Timestamp(ts.tz_convert("America/New_York").date())


def clean_chain(raw: pd.DataFrame, r: float, q: float = 0.0,
                max_rel_spread: float = 0.25, min_mid: float = 0.10,
                min_bid: float = 0.05, min_oi: int = 10, min_T_days: float = 1.0,
                max_z: float = 3.0, report: bool = False):
    df = raw.copy()
    n0 = len(df)
    drops = {}

    def note(name, before):
        drops[name] = before - len(df)

    n = len(df); df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])]; note("no two-sided quote", n)

    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    n = len(df); df = df[(df["mid"] >= min_mid) & (df["bid"] >= min_bid)]; note("price below floor", n)

    n = len(df); df = df[(df["ask"] - df["bid"]) / df["mid"] <= max_rel_spread]; note("spread too wide", n)

    oi = df["openInterest"].fillna(0) if "openInterest" in df else pd.Series(0, index=df.index)
    vol = df["volume"].fillna(0) if "volume" in df else pd.Series(0, index=df.index)
    n = len(df); df = df[(oi >= min_oi) | (vol > 0)]; note("illiquid", n)

    sess = session_date(df["snapshot_utc"].iloc[0])
    expiry_close = pd.to_datetime(df["expiry"]).dt.normalize() + pd.Timedelta(hours=_CLOSE_UTC_HOUR)
    session_close = sess + pd.Timedelta(hours=_CLOSE_UTC_HOUR)
    df["T"] = (expiry_close - session_close).dt.total_seconds() / (365.25 * 24 * 3600)
    n = len(df); df = df[df["T"] * 365.25 >= min_T_days]; note("expired or too short", n)
    df["cpi"] = np.where(df["cp"] == "C", 1, -1)

    out = []
    for T, g in df.groupby("T"):
        F = _implied_forward(g, r, T)
        # Black-76: work off the implied forward, never the raw spot. Using S with
        # q=0 assumes F = S*exp(rT), which is wrong by the dividend yield and by any
        # move between the 16:00 stock close and the 16:15 options close. Either
        # error shows up as a put/call vol gap right at the money.
        S_fwd = F * np.exp(-r * T)
        g = g.assign(F=F, S_fwd=S_fwd, k=np.log(g["strike"] / F),
                     q_implied=r - np.log(F / g["spot"]) / T)
        otm = ((g["cpi"] == 1) & (g["strike"] >= F)) | ((g["cpi"] == -1) & (g["strike"] < F))
        g = g[otm]
        if g.empty:
            continue
        g = g.assign(iv=bs.implied_vol(g["mid"].values, S_fwd, g["strike"].values,
                                       T, r, 0.0, g["cpi"].values))
        out.append(g)
    n = len(df)
    df = pd.concat(out, ignore_index=True) if out else df.iloc[:0].assign(F=np.nan, k=np.nan, iv=np.nan)
    note("ITM (kept OTM only)", n)

    n = len(df); df = df.dropna(subset=["iv"]); note("IV inversion failed", n)

    # Standardised moneyness, measured against the ATM vol of that expiry.
    # Using each option's own IV is self-defeating: deep OTM puts have inflated
    # IV, which sits in the denominator and lets exactly the worst quotes through.
    atm = df.loc[df.groupby("T")["k"].transform(lambda x: x.abs() == x.abs().min())] \
            .groupby("T")["iv"].first()
    df["atm_iv"] = df["T"].map(atm)
    df["z"] = df["k"] / (df["atm_iv"] * np.sqrt(df["T"]))
    n = len(df); df = df[df["z"].abs() <= max_z]; note("beyond wing cutoff", n)

    # vega is the natural fitting weight: it is the sensitivity the fit should care about
    df["vega"] = bs.vega(df["S_fwd"].values, df["strike"].values, df["T"].values, r, 0.0, df["iv"].values)
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"]
    df["fit_weight"] = df["vega"] / np.maximum(df["rel_spread"], 0.005)

    df = df.sort_values(["T", "strike"]).reset_index(drop=True)
    if report:
        rep = pd.DataFrame({"dropped": pd.Series(drops)})
        rep["pct_of_raw"] = (100 * rep["dropped"] / n0).round(1)
        rep.loc["KEPT", :] = [len(df), round(100 * len(df) / n0, 1)]
        return df, rep
    return df


def _implied_forward(g: pd.DataFrame, r: float, T: float) -> float:
    """F = K + e^{rT}(C - P) at the strike where |C - P| is smallest.

    Backs dividends and borrow cost out of the market rather than assuming them,
    which matters for a recent IPO where borrow can be expensive.
    """
    piv = g.pivot_table(index="strike", columns="cp", values="mid").dropna()
    if piv.empty or not {"C", "P"}.issubset(set(piv.columns)):
        return float(g["spot"].iloc[0]) * np.exp(r * T)
    K = piv.index[np.argmin(np.abs(piv["C"] - piv["P"]))]
    return float(K + np.exp(r * T) * (piv.loc[K, "C"] - piv.loc[K, "P"]))
