# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# prerequisites: place the 3 MT5 1-min CSVs in "NAS100 DATA/" (not in git, ~130MB)
pip install pandas numpy torch pyarrow openpyxl matplotlib scipy pytest

python3 -m nas100_rl.data                  # build parquet caches + timezone validation
python3 -m pytest nas100_rl/tests/ -q      # full test suite
python3 -m pytest nas100_rl/tests/test_core.py::test_s2_no_lookahead -q   # single test

python3 -m nas100_rl.rl.signals            # rebuild unified signal dataset (tv basis)
python3 -m nas100_rl.rl.evaluate           # baselines (solos / always-take) per period

# RL training (basis: "tv" = zero-spread verification parity, "live" = spread-adjusted)
python3 -m nas100_rl.rl.train --mode grid --basis live --updates 60
python3 -m nas100_rl.rl.train --mode final --seeds 9 --basis live                  # val-gate candidate
python3 -m nas100_rl.rl.train --mode final --seeds 9 --basis live --fit-through val  # deploy refit
python3 -m nas100_rl.rl.montecarlo live deploy_live   # overfitting battery

# live runner (see nas100_rl/live/README.md for the staged go-live protocol)
python3 -m nas100_rl.live.runner --selftest [--quick]  # parity proof vs frozen artifacts
python3 -m nas100_rl.live.runner --paper 5             # live loop over the val tail, simulated fills
python3 -m nas100_rl.live.runner --status              # health: process, feed age, heartbeat, positions
python3 -m nas100_rl.live.runner --live --gateway file # MT5 via NasBridge.mq5 (Wine) / mt5 (Windows)
```

Training is resumable (`--resume`, checkpoints every 10 updates). Everything runs on CPU
in minutes (M2/8GB-class hardware is sufficient).

The live layer (`nas100_rl/live/`) re-runs the FROZEN strategy code over trailing
windows (no re-port, no logic duplication); `--selftest` must stay bit-exact
(emissions/closures vs `signals_live.parquet`, features vs the cached matrix, decisions
vs `SignalEnv`) — run it after ANY change that could touch the live path, and treat a
selftest failure as a hard stop. The model account mirrors `SignalEnv` at 100k scale;
broker equity must never feed the policy's observations.

**This machine may be RUNNING LIVE.** A runner (`--live --gateway file`) may be trading
through the user's MT5-under-Wine terminal (`config.LIVE["bridge_dir"]` points at its
`MQL5/Files`; NasBridge.mq5 attached to a NAS100 chart). Check `--status` (or
`pgrep -fl nas100_rl.live.runner`) before killing python processes, restarting, or
editing live-path code; a flock in `data_cache/live_runner.lock` rejects a second
instance. Decisions/orders/heartbeats append to `data_cache/live_journal.jsonl`;
crash output goes to `data_cache/live_runner.log`. If a runner died with open
positions, they are NOT re-adopted on restart (server-side stops still protect them) —
surface this to the user instead of silently restarting. The user starts/stops the
runner themselves; never launch it unasked.

## Cardinal rules

1. **Strategy logic is FROZEN.** `nas100_rl/strategies/` (s2.py, dmi.py, cpmt/) are
   verified 1:1 ports of the Pine scripts in `Tradingview stratagies/`, matched
   trade-by-trade against the xlsx exports in `Tradingview Results/` (97.6%/91%/96%;
   see `nas100_rl/reports/verification_*.md`). The apparent bugs are deliberate
   replications of Pine behavior (e.g. DMI's entryTaken-once-per-NY-day, its
   NY-midnight var reset that freezes overnight stops, rejected same-direction entries
   that still overwrite stop/TP state; S2's dailyCount incrementing while in a trade).
   Any edit to strategies, `engine.py` fill semantics, or the strategy params in
   `config.py` invalidates verification and requires re-running the `verify/` comparison.
   The RL layer only gates/sizes entries; it never alters strategy behavior.
2. **The locked OOS (2026-01..06) is never trained on and never iterated against.**
   Every OOS evaluation must be logged via `report.log_oos_look()` (audit trail in
   `checkpoints/oos_looks.json`). Model selection happens on walk-forward folds inside
   train + the validation gate. The user-approved success gate is the COMBINED held-out
   year (val+OOS) vs the best individual strategy on both net and Sharpe.
3. **The model is the 9-seed ensemble, never a single seed.** Deployment policy =
   mean logits over `deploy_live_seed0-8.pt` (`train.load_ensemble("deploy_live")`).
   Per-seed val Sharpe ranges ~0.5-2.0 on identical configs; picking a seed post-hoc
   (one scores 2.59 on OOS) is the selection bias this design exists to prevent.
4. **Don't mix bases.** "tv" and "live" have parallel caches (`signals[_live].parquet`,
   `features[_live].parquet`), grid results, and checkpoint tags ({final|deploy}[_live]).
   A model trained on one basis must be evaluated on the same basis
   (`evaluate.load_all(basis=...)`).

## Architecture

Two layers connected by a precomputed signal stream:

**Layer 1 — verified backtest stack.** `data.py` parses the MT5 CSVs (server time =
New York + 7h year-round; session day runs 01:00→23:59 server; intraday HTF resamples
are session-anchored — validated against the data and the TV trade matches). UTC epoch
ms is the universal timestamp; TV exports are UTC. `engine.py` implements TradingView
broker semantics: market orders fill at the NEXT chart-bar open, stop/limit exits are
walked over the real 1-min sub-bars (bar-magnifier equivalent), gap-through fills at the
sub-bar open, same-bar stop+limit ties resolved by TV's open-to-nearest-extreme path
rule. `indicators.py` reproduces Pine `ta.*` exactly (RMA seeding, pivot strictness).
`verify/` aligns port trades to the TV xlsx trade lists with equity re-basing via the
exports' cumulative-PnL column (S2/CPMT size from equity; TV history starts 2019, our
data 2020).

**Layer 2 — RL overlay.** `rl/signals.py` runs the frozen strategies once over the full
timeline and freezes every trade's lifecycle (entry/exit/per-contract PnL). Copy-trading
semantics: skipping a signal does NOT change later signals. `rl/env.py` is a shared
portfolio account over that stream — actions {skip, 0.5x, 1.0x} of native sizing
(1.5x was removed: blanket re-leverage is Sharpe-invariant and Sharpe is the binding
gate metric) under hard caps (3% per trade / 6% concurrent). `rl/ppo.py` trains with
contextual-bandit credit (gamma=0; each decision immediately rewarded with its own
trade's variance-penalized equity contribution — portfolio-return credit failed: the
per-decision signal drowns in concurrent-position noise) and a KL anchor to the
always-take prior, with train-collection-only feature noise/dropout. `rl/features.py`
builds ~45 strictly causal features (last CLOSED 30m/daily bar; rolling strategy
performance from trades with exit <= decision time; z-stats from TRAIN rows only).
`rl/train.py` selects hyperparameters/budget on six ~5-month walk-forward folds scored
as Sharpe-margin-over-best-solo with a net constraint (folds mimic the short-window
gate geometry, which is luck-inflated). `rl/evaluate.py`/`report.py`/`montecarlo.py`
produce the gate tables, negative controls, and bootstrap/null/deflated-Sharpe battery.

Evaluator semantics (corrected 2026-06-11 after external audit — do not regress):
`_curve` marks boundary-crossing trades to market only (realized PnL is credited only
when the exit is in-window), and `metrics()`/MC prepend the day-0 equity anchor.
Numbers in docs predating the correction are slightly stale; current canonical results
live in `reports/final_report.md` (gate still passes: combined year 119,259/2.043 vs
best solo 33,509/1.124). `report.negative_controls` runs a 20-draw permutation null
plus signal-only/port-only ablations — never judge the model on a single permutation
draw.

Periods (train/val/oos) come from the three CSVs and are carried as labels through every
dataset; `evaluate.load_all(basis)` returns the `world` dict (signals, feature matrix,
mark prices, day grid, period index ranges) that all evaluation entry points share.

`data_cache/` is fully derived (delete to force rebuild; loaders take `refresh=True`).
`resample()`/`daily()` caches carry a `.meta.json` input fingerprint: a sliced input
computes fresh instead of reading the full-history cache, and a narrower input can
never overwrite a wider cache (only `refresh=True` forces a write) — keep this guard
when touching `data.py`. Tests in `nas100_rl/tests/` cover indicator exactness, fill
semantics, and no-lookahead truncation invariance for S2/CPMT — keep them passing
after any engine/indicator change.
