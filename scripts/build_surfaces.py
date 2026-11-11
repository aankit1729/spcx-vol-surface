"""Fit a volatility surface for every new session -> data/surfaces/<TICKER>.parquet

Incremental: sessions already in the file are skipped, so the daily run only
fits the newest one. Delete the file to refit everything from scratch.

    python scripts/build_surfaces.py SPCX SPY QQQ
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volsurface import surface  # noqa: E402

RISK_FREE = 0.042   # flat rate for discounting; vols are forward-based, so this is second order


def main(tickers):
    for t in tickers:
        hist = surface.update_history(ROOT / "data" / "raw", t, r=RISK_FREE,
                                      out_path=ROOT / "data" / "surfaces" / f"{t}.parquet")
        print(f"{t}: {hist['session'].nunique() if len(hist) else 0} sessions in history\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["SPCX", "SPY", "QQQ"])
