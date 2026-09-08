"""Daily snapshot of option chains -> data/raw/<TICKER>/<YYYY-MM-DD>.parquet

Run once per trading day after the close (cron / GitHub Actions). yfinance
gives the *current* chain only, so history is whatever you accumulate.

    python scripts/snapshot.py SPCX SPY QQQ
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

KEEP = [
    "contractSymbol", "strike", "lastPrice", "bid", "ask", "volume",
    "openInterest", "impliedVolatility", "inTheMoney", "lastTradeDate",
]


def snapshot(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    spot_hist = tk.history(period="1d")
    spot = float(spot_hist["Close"].iloc[-1])
    now = datetime.now(timezone.utc)
    frames = []
    for exp in tk.options:
        chain = tk.option_chain(exp)
        for cp, df in (("C", chain.calls), ("P", chain.puts)):
            d = df[KEEP].copy()
            d["cp"] = cp
            d["expiry"] = pd.Timestamp(exp)
            frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = ticker
    out["spot"] = spot
    out["snapshot_utc"] = now
    return out


def main(tickers):
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for t in tickers:
        df = snapshot(t)
        dest = RAW / t
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{day}.parquet"
        df.to_parquet(path, index=False)
        print(f"{t}: {len(df):6d} contracts, {df['expiry'].nunique():3d} expiries -> {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["SPCX", "SPY"])
