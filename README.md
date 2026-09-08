# volsurface — implied-vol surfaces and hedging on a newly listed mega-cap

Research question: **how does the options market price a stock with no history?**
SpaceX (SPCX) listed in June 2026 with the largest options debut on record. There is
no long return series to estimate realised vol, the term structure contains known
events (lockup expiry, index inclusion) and order flow is retail-heavy. This repo
builds the surface from raw quotes, fits parametric (SVI) and stochastic-volatility
(Heston) models, and tests whether the models are worth anything in a hedging backtest.
SPY/QQQ serve as the mature-market control.

## Pipeline
1. `scripts/snapshot.py` — daily EOD chain snapshot (GitHub Action, cron after US close)
2. `volsurface.clean` — mid prices, spread filters, put-call-parity forward, OTM-only, own IV solver
3. `volsurface.svi` — SVI slice fits with Durrleman butterfly and calendar arbitrage checks
4. `volsurface.heston` — COS-method pricer, calibration in IV space, model deltas
5. Notebooks:
   - `01_surface_evolution` — smile/term structure since listing, SPCX vs SPY
   - `02_model_comparison` — BS vs SVI vs Heston, in-sample vs next-day out-of-sample RMSE by moneyness/tenor
   - `03_variance_risk_premium` — implied vs subsequently realised vol per expiry window; event premia around lockup expiry
   - `04_delta_hedging` — short ATM straddle, daily rehedge, BS vs Heston delta, transaction costs, P&L distribution

## Data
Free chain data has no history, so snapshots accumulate from the day the Action is enabled.
Historical backfill options: Cboe DataShop (EOD, single-ticker), OptionMetrics via WRDS (academic).
Rates: use the SOFR/T-bill curve for `r`; single stocks are American — OTM-only filtering keeps
early-exercise premium negligible, which is verified in notebook 02.

## Run
```
pip install -e ".[dev]"
pytest
python scripts/snapshot.py SPCX SPY
```
