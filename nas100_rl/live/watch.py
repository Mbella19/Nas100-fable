"""Read-only pretty-printer for the live runner's journal stream.

Tails data_cache/live_journal.jsonl and renders each event as one readable
line (NY timestamps). Strictly read-only: safe to run while the runner is
live, and safe to Ctrl-C at any time.

    python3 -m nas100_rl.live.watch              # last 20 events, then follow
    python3 -m nas100_rl.live.watch -n 100       # more backlog first
    python3 -m nas100_rl.live.watch --hb         # heartbeats as lines too
    python3 -m nas100_rl.live.watch --no-follow  # print backlog and exit
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .. import config

A_NAME = {0: "SKIP", 1: "HALF 0.5x", 2: "FULL 1.0x"}

TTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if TTY else s
DIM = lambda s: _c("2", s)
BOLD = lambda s: _c("1", s)
GREEN = lambda s: _c("32", s)
RED = lambda s: _c("31", s)
YELLOW = lambda s: _c("33", s)
CYAN = lambda s: _c("36", s)


def _ny(ts) -> str:
    return pd.Timestamp(ts).tz_convert(config.NY_TZ).strftime("%m-%d %H:%M:%S")


def _ny_ms(ms: int) -> str:
    # always carry the date: catch-up events routinely reference prior days
    return (pd.Timestamp(int(ms), unit="ms", tz="UTC")
            .tz_convert(config.NY_TZ).strftime("%m-%d %H:%M"))


def _key(e) -> str:
    try:
        strat, ms, d = ast.literal_eval(e["key"])
        side = "LONG" if int(d) == 1 else "SHORT"
        return f"{strat.upper():4s} {side:5s} sig {_ny_ms(ms)} NY"
    except Exception:
        return str(e.get("key", "?"))


def render(e: dict) -> str | None:
    ev, t = e.get("ev"), _ny(e.get("ts"))
    if ev == "decision":
        name = A_NAME.get(e.get("action"), "?")
        col = {0: DIM, 1: YELLOW, 2: GREEN}.get(e.get("action"), str)
        extra = " HALTED" if e.get("halted") else ""
        return (f"{t}  {col(BOLD('DECIDE '))} {_key(e)} -> {col(name)}"
                f"  qty {e.get('qty', 0):.2f} @100k  model eq {e.get('equity', 0):,.0f}{extra}")
    if ev == "broker_open":
        if e.get("ok"):
            return (f"{t}  {GREEN('OPEN   ')} {_key(e)} -> {e.get('lots')} lots"
                    f" @ {e.get('fill')}  ticket {e.get('ticket')}")
        return f"{t}  {RED('OPEN-FAIL')} {_key(e)}  msg={e.get('msg')}"
    if ev == "mech_close":
        pnl = float(e.get("pc_pnl", 0.0))
        col = GREEN if pnl >= 0 else RED
        return (f"{t}  {CYAN('CLOSE  ')} {_key(e)} exit {_ny_ms(e['exit_ms'])} NY"
                f"  pnl/contract {col(f'{pnl:+,.1f}')}")
    if ev == "broker_close":
        ok = GREEN("ok") if e.get("ok") else RED("FAILED")
        return (f"{t}  {CYAN('BR-CLOSE')} {_key(e)} ticket {e.get('ticket')}"
                f" @ {e.get('px')} {ok}")
    if ev == "stale_signal_no_order":
        return (f"{t}  {YELLOW('STALE  ')} {_key(e)} age {e.get('age_min', 0):.0f}m"
                f" -> model only, no broker order")
    if ev == "skip_minlot":
        return f"{t}  {YELLOW('TOO-SML')} {_key(e)} {e.get('lots')} lots < min lot, no order"
    if ev == "KILL_SWITCH":
        return f"{t}  {RED(BOLD('KILL SWITCH'))} drawdown {e.get('dd', 0):.1%} — trading halted"
    if ev == "start":
        return f"{t}  {CYAN('START  ')} runner up, live from {_ny_ms(e['live_from_ms'])} NY"
    if ev == "gap_heal":
        return f"{t}  {GREEN('HEALED ')} {e.get('bars')} missing bars filled from gateway backlog"
    if ev == "gap_warning":
        return f"{t}  {RED('GAP!   ')} unhealed hole {e.get('span')}"
    if ev == "state_restored":
        return (f"{t}  {CYAN('RESTORE')} {e.get('open')} open trades, "
                f"{e.get('tickets')} tickets, eq {e.get('equity', 0):,.0f}"
                + (RED("  HALTED") if e.get("halted") else ""))
    if ev == "state_restore_failed":
        return f"{t}  {RED('RESTORE FAILED')} {e.get('err')} — started fresh"
    if ev in ("paper_start", "replay_done"):
        return f"{t}  {CYAN(ev.upper())} {json.dumps({k: v for k, v in e.items() if k not in ('ev', 'ts')}, default=str)}"
    if ev == "heartbeat":
        return (f"{t}  {DIM('beat')}   bar {e.get('last_bar')} srv"
                f"  model eq {e.get('model_equity', 0):,.0f}"
                f"  open {e.get('open_trades')} closed {e.get('closed_trades')}"
                + (RED("  HALTED") if e.get("halted") else ""))
    return f"{t}  {ev}  {json.dumps({k: v for k, v in e.items() if k not in ('ev', 'ts')}, default=str)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=20, help="backlog events to show first")
    ap.add_argument("--all", action="store_true", help="replay the whole journal")
    ap.add_argument("--hb", action="store_true", help="print heartbeats as lines")
    ap.add_argument("--no-follow", action="store_true", help="print backlog and exit")
    ap.add_argument("--journal", default=None, help="journal path override")
    args = ap.parse_args()

    path = Path(args.journal) if args.journal else config.CACHE_DIR / config.LIVE["journal"]
    while not path.exists():
        print(DIM(f"waiting for {path} ..."))
        time.sleep(2.0)

    f = open(path, "r")
    backlog = f.readlines()
    if not args.all and len(backlog) > 5000:
        backlog = backlog[-5000:]

    def emit(line: str, ticker: bool) -> None:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            return
        if e.get("ev") == "heartbeat" and not args.hb:
            if ticker and TTY:  # live one-line ticker instead of scroll spam
                sys.stdout.write("\r\033[K" + render(e))
                sys.stdout.flush()
            return
        if TTY:
            sys.stdout.write("\r\033[K")
        print(render(e))

    shown = [l for l in backlog
             if args.hb or args.all or '"ev": "heartbeat"' not in l]
    if not args.all:
        shown = shown[-args.n:]
    for line in shown:
        emit(line, ticker=False)
    if args.no_follow:
        return

    print(DIM(f"-- following {path.name} (Ctrl-C to stop) --"))
    try:
        while True:
            line = f.readline()
            if line:
                emit(line, ticker=True)
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
