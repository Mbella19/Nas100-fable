# NAS100 RL Meta-Controller

Three TradingView (Pine v6) NAS100 strategies — **S2 5-Bar Momentum Burst** (5m),
**PROJECT 1.8 DMI** (1m), **CPMT v12 chart patterns** (30m multi-TF) — ported to Python as
verified 1:1 logic replicas, plus an RL meta-controller that decides per signal whether to
**take / skip / half-size** each strategy's trade. Strategy logic is frozen; the RL only
gates entries on a shared portfolio account with hard risk caps.

## Results (live-cost basis: measured bid/ask spread included)

| period | RL net / Sharpe | always-take | best individual |
|---|---|---|---|
| train 2020-01..2025-05 | $3.55M / 2.51 | $1.27M / 1.39 | $215K / 1.47 |
| validation 2025-06..12 | $49.8K / 2.02 | $30.5K / 1.09 | $20.1K / 1.44 |
| locked OOS 2026-01..06 | $60.6K / 2.44 | $43.3K / 1.80 | $15.0K / 2.50 |
| held-out year combined | $125.5K / 2.15 | $79.6K / 1.34 | $33.5K / 1.06 |

Verification, Monte Carlo overfitting battery (paired block bootstrap, action-matched
nulls, deflated Sharpe), data-hygiene table and honest caveats: `nas100_rl/reports/`.

## Layout

```
Tradingview stratagies/   original Pine sources (frozen logic reference)
Tradingview Results/      TV strategy-tester exports (per-trade ground truth)
nas100_rl/
  config.py               frozen strategy params (as run on TV), paths, RL settings
  data.py                 MT5 CSV -> tz-resolved bars -> resamples (parquet cache)
  indicators.py           TV-exact ATR/RMA/EMA/DMI/pivots
  engine.py               TV-semantics fills (next-bar-open, 1-min intrabar stops)
  strategies/             s2.py, dmi.py, cpmt/ (pattern engine + strategy)
  verify/                 TV xlsx loader + trade-by-trade comparison
  rl/                     signals, features, env, PPO, train, evaluate, report, montecarlo
  tests/                  indicator exactness, fill semantics, no-lookahead tests
  reports/                verification + final reports, equity/MC plots
  checkpoints/            trained ensembles (deploy_live = production candidate)
```

## Reproduce

1. Place the three MT5 1-min CSVs in `NAS100 DATA/` (tab-separated MT5 export format;
   not committed — ~130 MB).
2. `pip install pandas numpy torch pyarrow openpyxl matplotlib scipy pytest`
3. `python3 -m nas100_rl.data` (build caches + tz validation), then `pytest nas100_rl/tests/`
4. Verification: see `nas100_rl/reports/verification_*.md` (built from `verify/`)
5. RL: `python3 -m nas100_rl.rl.train --mode grid --basis live` then
   `--mode final --seeds 9 --basis live [--fit-through val]`;
   evaluate via `nas100_rl/rl/report.py`, Monte Carlo via `python3 -m nas100_rl.rl.montecarlo live deploy_live`.

Not investment advice; see the go-live caveats in `nas100_rl/reports/final_report.md`.
