"""One implied-volatility surface per trading session, and a history of them.

A surface here is a stack of SVI slices, one per listed expiry, plus a rule for
the gaps between expiries:

    strikes      within an expiry, the SVI curve
    maturities   between expiries, total variance w = sigma^2 * T is interpolated
                 linearly in T at fixed log-moneyness k. This is the standard
                 choice: if the slices themselves do not cross, the interpolated
                 surface has no calendar arbitrage either.
    outside      flat volatility before the first expiry and after the last

Everything is quoted in forward log-moneyness k = log(K / F), with F the forward
implied by put-call parity for that expiry.

Typical use:

    raw  = pd.read_parquet("data/raw/SPCX/2026-11-11.parquet")
    surf = build_surface(raw, r=0.04)
    surf.iv(k=-0.10, T=0.25)          # vol of a 3-month, 10%-OTM put
    surf.strike_vol(K=120, T=0.25)    # same thing by strike
    surf.summary()                    # ATM vol and skew at standard tenors
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import clean, svi

TENORS = {"1m": 30, "3m": 91, "6m": 182, "1y": 365}
_PARAM_COLS = ["a", "b", "rho", "m", "sigma"]
_YEAR = 365.25


@dataclass
class Surface:
    ticker: str
    session: pd.Timestamp
    spot: float
    r: float
    slices: pd.DataFrame            # one row per expiry, sorted by T
    quotes: pd.DataFrame | None = None   # the cleaned quotes behind the fit, if kept

    # ------------------------------------------------------------------ core
    def _slice_variances(self, k):
        """Total variance of every slice at log-moneyness k. Shape (n_slices, *k.shape)."""
        k = np.asarray(k, dtype=float)
        return np.stack([svi.total_variance(k, svi.SVIParams(*row))
                         for row in self.slices[_PARAM_COLS].values])

    def total_variance(self, k, T):
        """w(k, T) = sigma^2 * T, for any k and T (arrays broadcast)."""
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        Ts = self.slices["T"].values
        W = self._slice_variances(k)
        if len(Ts) == 1:
            return W[0] * T / Ts[0]
        i = np.clip(np.searchsorted(Ts, T, side="right") - 1, 0, len(Ts) - 2)
        w_lo = np.take_along_axis(W, i[None, ...], axis=0)[0]
        w_hi = np.take_along_axis(W, (i + 1)[None, ...], axis=0)[0]
        frac = (T - Ts[i]) / (Ts[i + 1] - Ts[i])
        w = w_lo + (w_hi - w_lo) * np.clip(frac, 0.0, 1.0)
        w = np.where(T < Ts[0], W[0] * T / Ts[0], w)       # flat vol at the short end
        w = np.where(T > Ts[-1], W[-1] * T / Ts[-1], w)    # and at the long end
        return w

    def iv(self, k, T):
        """Implied volatility at log-moneyness k and maturity T (years)."""
        T = np.asarray(T, dtype=float)
        return np.sqrt(np.maximum(self.total_variance(k, T), 1e-12) / T)

    def forward(self, T):
        """Forward price for maturity T, interpolated between the parity-implied forwards."""
        Ts, F = self.slices["T"].values, self.slices["F"].values
        carry = np.log(F / self.spot) / Ts               # implied r - q per expiry
        return self.spot * np.exp(np.interp(T, Ts, carry) * np.asarray(T, dtype=float))

    def strike_vol(self, K, T):
        """Implied volatility for a strike in price terms."""
        return self.iv(np.log(np.asarray(K, dtype=float) / self.forward(T)), T)

    def atm_vol(self, T):
        """At-the-money-forward volatility."""
        return self.iv(0.0, T)

    # ------------------------------------------------------- smile measures
    def delta_vol(self, delta: float, T: float, cp: int) -> float:
        """Volatility at the strike with the given forward delta (cp=+1 call, -1 put).

        Forward delta is N(d1) for a call and N(d1) - 1 for a put, with no
        discounting, so a "25-delta" strike means the same thing at every maturity.
        """
        atm = float(self.atm_vol(T))
        k = np.linspace(-6.0, 6.0, 4001) * atm * np.sqrt(T)
        w = self.total_variance(k, T)
        call_delta = norm.cdf((-k + 0.5 * w) / np.sqrt(w))     # decreasing in k
        target = delta if cp == 1 else 1.0 - delta
        k_star = np.interp(target, call_delta[::-1], k[::-1])
        return float(self.iv(k_star, T))

    def skew_25d(self, T: float) -> float:
        """25-delta put vol minus 25-delta call vol. Positive = downside costs more."""
        return self.delta_vol(0.25, T, -1) - self.delta_vol(0.25, T, +1)

    def fly_25d(self, T: float) -> float:
        """Average of the 25-delta wings minus ATM: how curved the smile is."""
        return 0.5 * (self.delta_vol(0.25, T, -1) + self.delta_vol(0.25, T, +1)) - float(self.atm_vol(T))

    def covers(self, T: float) -> bool:
        """True if T lies between the first and last fitted expiry (no extrapolation)."""
        Ts = self.slices["T"].values
        return bool(Ts[0] <= T <= Ts[-1])

    def summary(self, tenors: dict | None = None) -> dict:
        """ATM vol, skew and curvature at fixed tenors, for comparing sessions.

        Listed expiries roll forward every day, so "the December expiry" is a
        different maturity each session. Fixed tenors are what make a time series.
        """
        out = {"session": self.session, "ticker": self.ticker, "spot": self.spot}
        for name, days in (tenors or TENORS).items():
            T = days / _YEAR
            if not self.covers(T):
                out.update({f"atm_{name}": np.nan, f"skew_{name}": np.nan, f"fly_{name}": np.nan})
                continue
            out[f"atm_{name}"] = float(self.atm_vol(T))
            out[f"skew_{name}"] = self.skew_25d(T)
            out[f"fly_{name}"] = self.fly_25d(T)
        return out

    def grid(self, moneyness=None, days=None) -> pd.DataFrame:
        """Vol on a regular grid: rows are days to expiry, columns are K/F."""
        moneyness = np.linspace(0.7, 1.3, 25) if moneyness is None else np.asarray(moneyness)
        if days is None:
            lo, hi = self.slices["days"].iloc[0], self.slices["days"].iloc[-1]
            days = np.unique(np.round(np.geomspace(max(lo, 7), hi, 30)))
        days = np.asarray(days, dtype=float)
        vol = self.iv(np.log(moneyness)[None, :], (days / _YEAR)[:, None])
        return pd.DataFrame(vol, index=pd.Index(days, name="days"),
                            columns=pd.Index(np.round(moneyness, 4), name="K/F"))

    # ------------------------------------------------------------ storage
    def to_frame(self) -> pd.DataFrame:
        """The slices as a flat table, with enough columns to rebuild the surface."""
        df = self.slices.copy()
        df.insert(0, "ticker", self.ticker)
        df.insert(1, "session", self.session)
        df["spot"], df["r"] = self.spot, self.r
        return df

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Surface":
        """Rebuild a surface from rows written by to_frame (one ticker, one session)."""
        df = df.sort_values("T").reset_index(drop=True)
        first = df.iloc[0]
        return cls(ticker=str(first["ticker"]), session=pd.Timestamp(first["session"]),
                   spot=float(first["spot"]), r=float(first["r"]),
                   slices=df.drop(columns=["ticker", "session", "spot", "r"]))


# ---------------------------------------------------------------- building
def build_surface(raw: pd.DataFrame, r: float, ticker: str | None = None,
                  min_quotes: int = 8, keep_quotes: bool = True, **clean_kwargs) -> Surface:
    """Raw chain snapshot -> fitted surface.

    An expiry becomes a slice only if it has at least `min_quotes` clean quotes
    and they straddle the forward; SVI has five parameters and cannot be pinned
    down by a handful of points on one side.
    """
    session = clean.infer_session(raw)
    c = clean.clean_chain(raw, r=r, session=session, **clean_kwargs)
    rows = []
    for T, g in c.groupby("T"):
        if len(g) < min_quotes or g["k"].min() >= 0 or g["k"].max() <= 0:
            continue
        k, iv, w = g["k"].values, g["iv"].values, g["fit_weight"].values
        p = svi.fit_slice(k, iv, T, weights=w)
        rows.append(dict(
            expiry=pd.Timestamp(g["expiry"].iloc[0]), T=T, days=T * _YEAR,
            F=float(g["F"].iloc[0]), q_implied=float(g["q_implied"].iloc[0]),
            a=p.a, b=p.b, rho=p.rho, m=p.m, sigma=p.sigma,
            n=len(g), k_lo=float(k.min()), k_hi=float(k.max()),
            rmse_bp=svi.rmse_iv(k, iv, T, p, weights=w) * 1e4,
            atm_vol=float(svi.implied_vol(0.0, T, p)),
            butterfly_ok=svi.butterfly_arbitrage_free(p, k_range=(k.min(), k.max())),
        ))
    if not rows:
        raise ValueError("no expiry had enough clean quotes to fit")
    sl = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    cal = [True]
    for i in range(1, len(sl)):
        lo, hi = max(sl["k_lo"][i - 1], sl["k_lo"][i]), min(sl["k_hi"][i - 1], sl["k_hi"][i])
        cal.append(svi.calendar_arbitrage_free(svi.SVIParams(*sl.loc[i - 1, _PARAM_COLS]),
                                               svi.SVIParams(*sl.loc[i, _PARAM_COLS]),
                                               k_range=(lo, hi)) if hi > lo else True)
    sl["calendar_ok"] = cal

    name = ticker or (str(raw["ticker"].iloc[0]) if "ticker" in raw else "?")
    return Surface(ticker=name, session=session, spot=float(raw["spot"].iloc[0]), r=r,
                   slices=sl, quotes=c if keep_quotes else None)


# ----------------------------------------------------------------- history
def list_snapshots(raw_dir, ticker: str) -> pd.DataFrame:
    """One snapshot per trading session, chosen from whatever files exist.

    File names are ignored; the session is read from each file's contents. If a
    session has several snapshots (a manual run, a holiday re-run of a stale
    chain), the first one taken after that session's close is used.
    """
    rows = []
    for f in sorted(Path(raw_dir, ticker).glob("*.parquet")):
        head = pd.read_parquet(f, columns=["snapshot_utc", "lastTradeDate"])
        session = clean.infer_session(head)
        taken = pd.Timestamp(head["snapshot_utc"].iloc[0])
        if taken.tz is None:
            taken = taken.tz_localize("UTC")
        close = (session + pd.Timedelta(hours=16, minutes=15)).tz_localize("America/New_York")
        rows.append(dict(session=session, path=str(f), snapshot_utc=taken, after_close=taken >= close))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["session", "after_close", "snapshot_utc"], ascending=[True, False, True])
    return df.drop_duplicates("session", keep="first").reset_index(drop=True)


def update_history(raw_dir, ticker: str, r: float, out_path, verbose: bool = True) -> pd.DataFrame:
    """Fit a surface for every session not yet in `out_path`, and save the lot.

    Incremental, so the daily job only pays for the newest session.
    """
    out_path = Path(out_path)
    hist = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    done = set(pd.to_datetime(hist["session"])) if len(hist) else set()
    new = []
    for snap in list_snapshots(raw_dir, ticker).itertuples():
        if snap.session in done:
            continue
        if not snap.after_close:
            if verbose:
                print(f"{ticker} {snap.session.date()}: only an intraday snapshot, skipped")
            continue
        try:
            surf = build_surface(pd.read_parquet(snap.path), r=r, ticker=ticker, keep_quotes=False)
        except Exception as exc:                      # one bad day must not stop the rest
            if verbose:
                print(f"{ticker} {snap.session.date()}: FAILED ({exc})")
            continue
        new.append(surf.to_frame())
        if verbose:
            s = surf.slices
            print(f"{ticker} {snap.session.date()}: {len(s)} expiries, "
                  f"median fit error {s['rmse_bp'].median():.0f} bp")
    if new:
        hist = pd.concat([hist] + new, ignore_index=True).sort_values(["session", "T"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(out_path, index=False)
    return hist


def surfaces_from_history(hist: pd.DataFrame) -> dict:
    """{session: Surface} from a stored history table."""
    return {pd.Timestamp(s): Surface.from_frame(g) for s, g in hist.groupby("session")}


def summarise_history(hist: pd.DataFrame, tenors: dict | None = None) -> pd.DataFrame:
    """Fixed-tenor ATM vol, skew and curvature for every session: one row per day."""
    rows = [surf.summary(tenors) for surf in surfaces_from_history(hist).values()]
    return pd.DataFrame(rows).set_index("session").sort_index()
