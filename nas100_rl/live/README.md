# Live runner — verified Python brain, MT5 hands

The frozen, TV-verified strategies and the 9-seed `deploy_live` ensemble run in
Python exactly as in backtest; MT5 only supplies bars and executes orders.
Nothing is re-ported: the engines re-run the frozen strategy code over trailing
windows (sized for bit-converged indicators and all path-dependent state), and
**parity is proven, not assumed** — `--selftest` asserts the live pipeline
reproduces the frozen signal stream, the cached feature matrix, and SignalEnv's
decisions on historical data before you ever trade.

## Components

| piece | what it does |
|---|---|
| `engines.py` | windowed replay of frozen `s2/dmi/cpmt` -> emissions + closed trades |
| `features_live.py` | the 46 features, frozen train z-stats, one signal at a time |
| `account.py` | 100k-scale mirror of `SignalEnv` — feeds the policy its portfolio state (never raw broker equity) |
| `runner.py` | loop, decisions, executor, journal, kill switch (halt at 35% model-account DD), `--selftest` / `--paper` |
| `gateway.py` | `ReplayGateway` (offline), `FileBridgeGateway` (Wine), `Mt5Gateway` (Windows VPS) |
| `bridge_ea/NasBridge.mq5` | EA for the file bridge (Wine testing on the Mac) |

## Order of operations

```bash
# 1) parity proof on historical data (no broker needed; ~minutes with --quick)
python3 -m nas100_rl.live.runner --selftest --quick     # last 30 val days
python3 -m nas100_rl.live.runner --selftest             # full validation period

# 2) offline paper run through the live loop (replay gateway, simulated fills)
python3 -m nas100_rl.live.runner --paper 10

# 3) Mac + MT5-under-Wine (demo account!):
#    - compile bridge_ea/NasBridge.mq5 in MetaEditor, attach to a NAS100 chart,
#      enable Algo Trading; find the terminal's MQL5/Files dir
#      (File > Open Data Folder), set config.LIVE["bridge_dir"] to it
python3 -m nas100_rl.live.runner --live --gateway file

# 4) Windows VPS production: pip install MetaTrader5, then
python3 -m nas100_rl.live.runner --live --gateway mt5
```

## Non-negotiables

- **Same broker/symbol the data came from** (Pepperstone NAS100). The tz
  convention (server = NY+7) and the live-cost spread basis were measured on
  this feed; a different feed invalidates both.
- **Demo first, then small.** The staged protocol is: selftest -> paper ->
  weeks on demo with nightly reconciliation -> 10-25% risk scale
  (`LIVE["risk_scale"]`) -> full size. Pre-registered halt: model-account
  drawdown > 35% (MC 95th pct was ~28%), any selftest/parity failure, broker
  spec change.
- **The model account is the policy's reality.** Broker fills/swap/slippage are
  journaled and reconciled, but never fed into the observations — that would
  push the policy out of its training distribution.
- **Quirks ship as-is.** DMI's frozen overnight stops, S2's risk floor etc. are
  part of the verified behavior; "fixing" them live destroys parity.
- After downtime just restart: bars backfill, copy-trading semantics make the
  mechanical stream self-healing; server-side SL/TP protect open positions
  while the bot is dark.

## Known, accepted live-vs-backtest gaps (measure in reconciliation)

- Exits the strategies manage intra-bar (DMI/CPMT trails, time stops) are
  executed by the bot at the close of the 1-min/30-min bar that triggered them
  (the backtest fills at the level inside the bar). Initial SL (and DMI TP) are
  server-side at exact levels, so stop-outs before any trail ratchet are
  parity-exact. S2's stop never trails -> S2 exits are parity-exact.
- Financing/swap and weekend rollover are NOT in the backtest basis (~4%/yr of
  account at always-take exposure; the RL's skips reduce it). Expect live to
  run below backtest by roughly this.
- Feed noise: the verification showed net PnL is feed-sensitive (gross flows
  match within a few %; net is a small residual). Budget ±10-25% of strategy
  net before concluding anything from a live month.
