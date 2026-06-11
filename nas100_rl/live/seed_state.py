"""One-time migration: build live_state.json for positions opened BEFORE state
persistence existed (2026-06-11), so the first restart with the new runner code
adopts them instead of orphaning them.

Reconstructs the model account from the journal (decisions/opens/closes) and
keeps only trades whose broker position is still open in positions.csv.
Refuses to run while a live runner holds the lock. Harmless to re-run, and
pointless once a runner has written its own state (it would overwrite newer
state — don't run it after the new runner has traded).

    python3 -m nas100_rl.live.seed_state          # with the runner STOPPED
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from .. import config

L = config.LIVE
BARLEN = {"s2": 5, "dmi": 1, "cpmt": 30}


def main() -> None:
    try:
        import fcntl
        lock = open(config.CACHE_DIR / "live_runner.lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit("REFUSED: a live runner is running — stop it first.")
    except ImportError:
        pass

    # broker truth: tickets still open, and their server-side SL levels
    pos_sl: dict[str, float] = {}
    pos_path = Path(L["bridge_dir"]) / "positions.csv"
    if pos_path.exists():
        for line in pos_path.read_text().strip().splitlines():
            p = line.split(",")
            if len(p) >= 5:
                pos_sl[p[0]] = float(p[4])

    decisions: dict = {}
    opens: dict = {}
    closes: dict = {}
    hb = None
    n_dec = 0
    for line in open(config.CACHE_DIR / L["journal"]):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = e.get("ev")
        if ev == "decision":
            n_dec += 1
            decisions[e["key"]] = e
        elif ev == "broker_open" and e.get("ok"):
            opens[e["key"]] = e
        elif ev == "mech_close":
            closes[e["key"]] = e
        elif ev == "heartbeat":
            hb = e

    cash = config.RL["initial_capital"]
    n_closed = 0
    for k, c in closes.items():
        d = decisions.get(k)
        if d is not None and k in opens and d.get("qty", 0) > 0:
            cash += float(d["qty"]) * float(c["pc_pnl"])   # the old runner's
            n_closed += 1                                  # realized closes

    open_trades, tickets = [], []
    for k, op in opens.items():
        t = str(op.get("ticket"))
        if t not in pos_sl:
            continue                   # broker already flat: nothing to manage
        d = decisions.get(k)
        if d is None or d.get("qty", 0) <= 0:
            continue
        strat, sig_ms, direction = ast.literal_eval(k)
        if k in closes:
            # mechanically dead but broker still open (orphaned while no state
            # existed): hand the ticket over WITHOUT an open model trade — the
            # engines re-emit the closure on the first steps after restart,
            # which pops the ticket and closes the broker side automatically
            tickets.append([[strat, int(sig_ms), int(direction)], t])
            print(f"handover {k}: mech-closed, broker open -> runner will close ticket {t}")
            continue
        if pos_sl[t] <= 0:
            print(f"skip {k}: no server-side SL to derive stop_dist from")
            continue
        fill = float(op["fill"])
        # stop distance from the SL we attached at open: equals the mechanical
        # stop distance up to fill-vs-open drift (negligible for the risk/obs
        # arithmetic it feeds)
        open_trades.append(dict(trade_key=[strat, int(sig_ms), int(direction)],
                                strategy=strat, direction=int(direction),
                                qty=float(d["qty"]), stop_dist=abs(fill - pos_sl[t]),
                                entry_ms=int(sig_ms) + BARLEN[strat] * 60_000,
                                entry_price=fill, exit_ms=None, pc_pnl=None))
        tickets.append([[strat, int(sig_ms), int(direction)], t])
        print(f"adopt {k}: qty {d['qty']:.3f} @ {fill} "
              f"stop_dist {abs(fill - pos_sl[t]):.1f} ticket {t}")

    eq = float(hb["model_equity"]) if hb else cash
    st = dict(acct=dict(mode="live", e0=config.RL["initial_capital"], cash=cash,
                        open=open_trades, peak=max(config.RL["initial_capital"], eq),
                        prev_mark_eq=eq, mark_eq=eq, ret_hist=[],
                        n_decisions=n_dec, n_closed=n_closed),
              tickets=tickets, halted=False, last_bar_ms=0,
              ts=pd.Timestamp.utcnow().isoformat(), seeded=True)
    out = config.CACHE_DIR / "live_state.json"
    out.write_text(json.dumps(st))
    print(f"wrote {out}: {len(open_trades)} open trades, cash {cash:,.2f}, "
          f"mark eq {eq:,.2f}, {n_dec} decisions / {n_closed} closes carried")


if __name__ == "__main__":
    main()
