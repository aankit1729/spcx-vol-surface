# How does the options market price a stock with no history?

Implied volatility surface analysis of SpaceX (SPCX) — listed June 2026, the largest IPO and
options debut on record — against SPY as a mature-market control. Raw quotes → cleaned surface →
parametric (SVI) and stochastic-volatility (Heston) fits → hedging backtest.

**Status:** notebook 01 complete on the first snapshot. Daily collection running; time-series
notebooks (03, 04) follow once enough history accumulates.

## Findings from the first snapshot (session 2026-09-08)

### 1. SPCX has inverted skew out to one year

The 25-delta skew — put vol minus call vol at matched delta — is the standard measure of how much
more the market charges for downside than upside protection. For SPY it is +4 to +6 vol points at
every tenor beyond a week, as it is for essentially every equity index and mega-cap.

For SPCX it is **negative** from 10 days to roughly 370 days, bottoming near −1.5 vol points at one
month. Calls are more expensive than puts. It crosses zero near one year and reaches only +1.3 at
two years. The smile is not merely symmetric; at short tenors it leans toward the upside, which is
consistent with heavy retail call demand dominating whatever institutional put hedging exists.

### 2. The level is 3.4–4.2× SPY, and the term structures have opposite shapes

| tenor | SPCX ATM vol | SPY ATM vol | ratio |
|---|---|---|---|
| 3 days | 66.1% | 13.7% | 4.8× |
| 1 month | 53.7% | 12.7% | 4.2× |
| 6 months | 56.5% | 15.1% | 3.7× |
| 1 year | 57.7% | 16.9% | 3.4× |
| 2 years | 59.3% | 18.5% | 3.2× |

SPY's term structure rises monotonically, the normal shape: near-term uncertainty resolves and
long-run vol mean-reverts to a higher level. SPCX spikes at the front (next week's event risk),
troughs at one month, then rises slowly and never comes down. The market does not expect the
uncertainty about this company to resolve on any horizon it can price.

### 3. The forward carries a borrow cost

The implied forward from put–call parity gives an implied dividend/borrow yield per expiry. For SPY
this converges to 0.95% at long tenors, matching its dividend yield — a sanity check on the
pipeline. SPCX pays no dividend, yet shows 0.5–1.0% at long tenors and 1–2% at three to six months.
That is the cost of borrowing shares to short, embedded in the forward, in a name with a lockup
expiry pending. (The very front expiries show larger values for both names; that is the 16:00
stock close versus 16:15 options close, and is why nothing downstream uses spot.)

### 4. Heston fits SPY and structurally fails on SPCX

| | v0 | κ | θ | ξ | ρ | Feller | RMSE (calib / full) |
|---|---|---|---|---|---|---|---|
| SPY | 0.016 (12.7%) | 4.0 | 0.050 (22.4%) | 1.24 | −0.67 | violated | 50 / 69 bp |
| SPCX | 0.332 (57.6%) | 7.4 | 0.358 (59.8%) | **3.00 (at bound)** | **−0.015** | violated | 235 / 329 bp |

SPY's parameters are textbook, including the Feller violation that every SPY calibration produces.
SPCX's are degenerate: ξ pinned at its ceiling and ρ ≈ 0. Heston generates skew through ρ alone,
and SPCX's skew changes sign across the term structure, so no single ρ can fit it. The optimiser
sets ρ to zero and inflates ξ to widen a symmetric smile. A model that can fit SPCX needs either a
jump component or term-structure in the correlation. That the standard model cannot describe this
surface is the result, not a calibration failure.

### Data-quality notes
- SVI fits each expiry to 2–25 bp (vega-weighted) with no butterfly violations over the observed
  strike range. Calendar violations at 22d and 204d on SPY trace to a handful of stale quotes above
  the smile; they are flagged, not smoothed away.
- The 1-day expiry is tick-noise-dominated and excluded from all conclusions.

## Pipeline lessons

Four bugs that each produced plausible-looking wrong answers, all caught by inspecting output
against priors rather than by tests:

1. **Spot vs. forward.** Inverting IVs with spot and q = 0 while classifying OTM by the parity-implied
   forward produced a put/call vol gap at the money — visible as a kink at k = 0 that SVI could not
   fit. Once a forward exists, nothing downstream touches spot.
2. **A self-defeating wing filter.** Standardised moneyness k/(σ√T) using each option's *own* IV lets
   deep OTM puts through, because their inflated IV sits in the denominator. The yardstick must be
   the ATM vol of that expiry.
3. **Inverse-spread weighting.** A penny-wide quote on a four-cent option is 25% wide relative and
   pure noise, yet received 5× the weight of an ATM quote. Weights are vega / relative spread.
4. **Session date.** A 00:05 UTC snapshot captures the previous New York session. Off by one day,
   the front expiry's vol was inflated by ~√(3/2).

The first run reported butterfly arbitrage on 27 of 29 SPY expiries. A result implying the pipeline
had out-priced every market maker on the most liquid surface in the world was, on priors, a bug —
and it was three.

## Pipeline
1. `scripts/snapshot.py` — daily EOD chain snapshot (GitHub Action after US close)
2. `volsurface.clean` — liquidity filters, parity forward, Black-76 IV, standardised-moneyness cutoff,
   per-filter drop report
3. `volsurface.svi` — SVI per expiry with butterfly and calendar checks over the data range
4. `volsurface.heston` — COS-method pricer, vega-weighted calibration on a thinned surface, degenerate-fit detection
5. `notebooks/01_surface_evolution.ipynb` — everything above; runs in Colab with no setup

Planned: `02_model_comparison` (next-day out-of-sample error), `03_variance_risk_premium` (implied vs
realised per expiry, lockup-expiry event premium), `04_delta_hedging` (short straddle, BS vs Heston
delta, transaction costs, P&L attribution).

## Run
```
pip install -e ".[dev]" && pytest && python scripts/snapshot.py SPCX SPY
```
