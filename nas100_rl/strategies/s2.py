"""S2: 5-Bar Momentum Burst with Declining Risk — 1:1 port of the Pine v6 script.

Chart TF: 5 minutes. Parameters frozen in config.S2 (as run on TradingView:
lookback=4, threshold=4.5, riskPct=0.006).

Pine quirks replicated exactly:
- ATR(14) Wilder, shifted one bar (value at t uses data through t-1).
- dailyCount increments on EVERY gated signal, even while a position is open.
- 1st signal of the NY day -> 3x ATR stop; 2nd -> 9x ATR stop.
- Entry at next 5m bar open; static stop active from the entry bar onward,
  filled on the 1-min path (bar magnifier); gap-through fills at sub-bar open.
- Force exit (market at next bar open) when NY hour >= 16, or Friday (UTC) and
  UTC hour >= 18.
- qty = clamp(equity * riskPct / stopDist, 1, 200), fractional, sized on realized
  equity at the signal bar close. Commission $1.25 per contract per order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, indicators
from ..engine import M1Map, Trade, trades_to_frame, walk_stop_over_range

P = config.S2


def run(bars5: pd.DataFrame, m1: pd.DataFrame, initial_equity: float | None = None,
        collect_signals: bool = False) -> dict:
    """Run S2 over 5-min bars with 1-min fills.

    Returns dict(trades=DataFrame, equity_end=float, signals=list|None).
    `collect_signals` additionally records every would-be entry (for the RL layer),
    including ones skipped because a position was open.
    """
    o = bars5["open"].to_numpy()
    h = bars5["high"].to_numpy()
    l = bars5["low"].to_numpy()
    c = bars5["close"].to_numpy()
    n = len(bars5)

    atr_raw = indicators.atr(h, l, c, P["atr_period"])
    atr_val = np.concatenate(([np.nan], atr_raw[:-1]))          # [1] shift
    lb = P["lookback"]
    move = np.full(n, np.nan)
    move[lb:] = c[lb:] - c[:-lb]

    ny = bars5["ny"]
    ny_hour = ny.dt.hour.to_numpy()
    ny_day = ny.dt.day.to_numpy()
    ny_month = ny.dt.month.to_numpy()
    utc_ms = bars5["utc_ms"].to_numpy()
    utc_hour = ((utc_ms // 3_600_000) % 24).astype(np.int64)
    utc_wd = (((utc_ms // 86_400_000) + 3) % 7).astype(np.int64)  # Mon=0 .. Fri=4

    is_us = (ny_hour >= 9) & (ny_hour < 16)
    force_exit = (ny_hour >= 16) | ((utc_wd == 4) & (utc_hour >= P["friday_exit_utc_hour"]))
    valid_atr = ~np.isnan(atr_val) & (atr_val > P["atr_valid_min"]) & (atr_val < P["atr_valid_max"])
    valid = is_us & valid_atr & ~force_exit
    thr = P["threshold"] * atr_val
    long_cond = valid & (move > thr)
    short_cond = valid & (move < -thr)

    m1map = M1Map(m1, bars5["srv"], 5)
    comm = P["commission_per_contract_per_order"]

    equity = float(initial_equity if initial_equity is not None else P["initial_capital"])
    trades: list[Trade] = []
    signals: list[dict] = [] if collect_signals else None

    pos: Trade | None = None
    pos_stop = 0.0
    pending_entry: dict | None = None
    pending_close: str | None = None
    daily_count = 0
    prev_day, prev_month = -1, -1

    for i in range(n):
        # --- fills at bar open ------------------------------------------------
        if pending_close is not None:
            if pos is not None:
                pos.close(utc_ms[i], o[i], pending_close, comm * pos.qty)
                equity += pos.pnl
                trades.append(pos)
                pos = None
            pending_close = None
        if pending_entry is not None:
            e = pending_entry
            pos = Trade(strategy="s2", direction=e["dir"], qty=e["qty"],
                        entry_ms=int(utc_ms[i]), entry_price=float(o[i]),
                        commission=comm * e["qty"],
                        meta=dict(signal_i=e["i"], stop=e["sl"], nth=e["nth"]))
            pos_stop = e["sl"]
            pending_entry = None

        # --- intrabar stop on the 1-min path ----------------------------------
        if pos is not None:
            i0, i1 = m1map.range(i)
            hit = walk_stop_over_range(m1map, i0, i1, pos.direction, pos_stop)
            if hit is not None:
                j, px = hit
                # TV reports fills with the chart bar's open time; keep the exact
                # minute separately for the RL portfolio simulation.
                pos.meta["exit_ms_exact"] = int(m1map.ms[j])
                pos.close(utc_ms[i], px, "stop", comm * pos.qty)
                equity += pos.pnl
                trades.append(pos)
                pos = None

        # --- bar close logic ---------------------------------------------------
        if ny_day[i] != prev_day or ny_month[i] != prev_month:
            daily_count = 0
            prev_day, prev_month = int(ny_day[i]), int(ny_month[i])

        is_long = bool(long_cond[i])
        is_short = bool(short_cond[i])
        if (is_long or is_short) and daily_count < P["max_per_day"]:
            sm = P["stop_mult"] if daily_count == 0 else P["stop_mult_2nd"]
            daily_count += 1
            direction = 1 if is_long else -1
            stop_dist = sm * atr_val[i]
            qty = max(P["qty_min"], min(P["qty_max"], equity * P["risk_pct"] / stop_dist))
            sl = c[i] - direction * stop_dist
            if collect_signals:
                signals.append(dict(strategy="s2", signal_i=i, signal_ms=int(utc_ms[i]),
                                    direction=direction, stop_dist=float(stop_dist),
                                    sl=float(sl), nth=daily_count, atr=float(atr_val[i]),
                                    taken=pos is None))
            if pos is None:
                pending_entry = dict(i=i, dir=direction, qty=qty, sl=sl, nth=daily_count)

        if force_exit[i] and pos is not None:
            pending_close = "session_close"

    if pos is not None:
        pos.close(utc_ms[n - 1], c[n - 1], "open_at_end", comm * pos.qty)
        equity += pos.pnl
        trades.append(pos)

    return dict(trades=trades_to_frame(trades), equity_end=equity, signals=signals)
