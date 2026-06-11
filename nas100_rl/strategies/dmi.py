"""PROJECT 1.8 DMI — 1:1 port of the Pine v6 script, run on the 1-minute chart
(as in the TV export). Parameters frozen in config.DMI; useFirstCandle=Off mode.

Pine quirks replicated exactly (all verified in the source):
- Signal detection (NY 09:30-15:58, in-session, before NYC close):
  long = +DI>-DI & ADX>=30 & close>open & close>close[1] & volume>SMA14(volume);
  sets a sticky flag with potentialEntryPrice/initialStopLevel — the two directions
  SHARE potentialEntryPrice/initialStopLevel (later signal overwrites them).
- Arming (same or later bar, long checked first with `else if` for short):
  long fires when high >= potential + 1.5*ATR(5) and (high - 5*ATR) > initialStop;
  market entry at NEXT 1-min bar open; TP = signal-bar close +/- 15*ATR (never
  recomputed); stop starts at high -/+ 5*ATR.
- NO exit order is active during the entry bar (the first strategy.exit call
  happens at the entry bar's close, after a trail update using the entry bar).
- Trail update each in-management bar: stop = max(stop, high - 5*ATR) for longs
  (min/low+5*ATR shorts); exit order replaced at bar close, active next bar.
- Day reset at NY midnight wipes ALL script vars (entryTaken, stops, flags) even
  with a position open -> the last placed exit order stays active but FROZEN
  (no more trailing) until a later signal overwrites it or the position exits.
- entryTaken resets only at day change -> max ONE arming event per NY day.
- An arming while a position is still open queues a strategy.entry: opposite
  direction REVERSES the position; same direction is rejected by the broker
  (pyramiding) BUT the script state (stop/TP/entryPrice/trailActive) is still
  overwritten and the frozen order is replaced -> the open position is re-managed
  with the new levels.
- Sizing: qty = max(0.01, round(200/stopDistance, 2)) (round half away from zero).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import config, indicators
from ..engine import Trade, trades_to_frame, walk_exit

P = config.DMI


def _round2_half_away(x: float) -> float:
    return math.floor(x * 100 + 0.5) / 100 if x >= 0 else -math.floor(-x * 100 + 0.5) / 100


def run(m1: pd.DataFrame, initial_equity: float | None = None,
        collect_signals: bool = False, risk_dollars: float | None = None) -> dict:
    o = m1["open"].to_numpy()
    h = m1["high"].to_numpy()
    l = m1["low"].to_numpy()
    c = m1["close"].to_numpy()
    vol = m1["volume"].to_numpy()
    utc_ms = m1["utc_ms"].to_numpy()
    n = len(m1)

    atr = indicators.atr(h, l, c, P["atr_length"])
    di_p, di_m, adx = indicators.dmi(h, l, c, P["dmi_length"], P["adx_smoothing"])
    avg_vol = indicators.sma(vol, P["volume_sma_len"])

    ny = m1["ny"]
    ny_min = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
    ny_dow = ny.dt.dayofweek.to_numpy()           # Pine dayofweek change == NY date change
    s0 = P["session_start_ny"][0] * 60 + P["session_start_ny"][1]
    s1 = P["session_end_ny"][0] * 60 + P["session_end_ny"][1]
    in_session = (ny_min >= s0) & (ny_min < s1)
    # isAfterNYCClose uses time_close (bar open + 1min): blocks bars from 15:59 on
    after_close = (ny_min + 1) >= s1

    with np.errstate(invalid="ignore"):
        dmi_long_ok = (di_p > di_m) & (adx >= P["min_adx"])
        dmi_short_ok = (di_m > di_p) & (adx >= P["min_adx"])
        vol_ok = vol > avg_vol * P["volume_multiplier"]
        up_bar = np.concatenate(([False], (c[1:] > o[1:]) & (c[1:] > c[:-1])))
        dn_bar = np.concatenate(([False], (c[1:] < o[1:]) & (c[1:] < c[:-1])))
    base_long = dmi_long_ok & up_bar & vol_ok
    base_short = dmi_short_ok & dn_bar & vol_ok
    eligible = in_session & ~after_close

    risk = float(risk_dollars if risk_dollars is not None else P["risk_per_trade"])
    atr_m = P["atr_multiplier"]

    trades: list[Trade] = []
    signals: list[dict] = [] if collect_signals else None
    equity = float(initial_equity if initial_equity is not None else P["initial_capital"])

    pos: Trade | None = None
    order_stop = math.nan        # persistent exit order (frozen across day resets)
    order_limit = math.nan
    order_active = False
    pending: dict | None = None  # market order queued at prev bar close

    # script vars
    entry_taken = False
    long_sig = False
    short_sig = False
    potential = math.nan
    initial_stop = math.nan
    cur_stop = math.nan
    cur_tp = math.nan
    trade_count = 0
    prev_dow = -1

    for i in range(n):
        # --- 1) market order fills at open ------------------------------------
        if pending is not None:
            d = pending["dir"]
            if pos is None:
                pos = Trade(strategy="dmi", direction=d, qty=pending["qty"],
                            entry_ms=int(utc_ms[i]), entry_price=float(o[i]),
                            meta=dict(signal_i=pending["i"]))
            elif pos.direction != d:
                pos.close(utc_ms[i], o[i], "Long" if d > 0 else "Short", 0.0)
                equity += pos.pnl
                trades.append(pos)
                pos = Trade(strategy="dmi", direction=d, qty=pending["qty"],
                            entry_ms=int(utc_ms[i]), entry_price=float(o[i]),
                            meta=dict(signal_i=pending["i"], reversal=True))
                order_active = False  # old exit order dies with the reversal
            # same direction: rejected (pyramiding); state was already overwritten
            pending = None

        # --- 2) intrabar exit fills (active order from previous bars) ---------
        if pos is not None and order_active:
            st = None if math.isnan(order_stop) else order_stop
            li = None if math.isnan(order_limit) else order_limit
            res = walk_exit(o[i], h[i], l[i], c[i], pos.direction, st, li)
            if res is not None:
                px, why = res
                pos.close(utc_ms[i], px, "SL Hit" if why == "stop" else "TP Hit", 0.0)
                equity += pos.pnl
                trades.append(pos)
                pos = None
                order_active = False

        # --- 3) bar-close script logic -----------------------------------------
        if ny_dow[i] != prev_dow:
            prev_dow = int(ny_dow[i])
            entry_taken = False
            long_sig = False
            short_sig = False
            potential = math.nan
            initial_stop = math.nan
            cur_stop = math.nan
            cur_tp = math.nan
            trade_count = 0

        # detection block
        if (not entry_taken) and eligible[i] and trade_count < P["max_trades_per_day"]:
            if base_long[i] and not long_sig:
                long_sig = True
                potential = c[i]
                initial_stop = potential - atr_m * atr[i]
            if base_short[i] and not short_sig:
                short_sig = True
                potential = c[i]
                initial_stop = potential + atr_m * atr[i]

        # arming block (long first; Pine `else if` for short)
        if (long_sig or short_sig) and (not entry_taken) and trade_count < P["max_trades_per_day"] \
                and eligible[i]:
            if long_sig and dmi_long_ok[i]:
                if h[i] >= potential + P["trail_start_multiplier"] * atr[i]:
                    new_stop = h[i] - atr_m * atr[i]
                    if new_stop > initial_stop:
                        entry_price = c[i]
                        cur_stop = new_stop
                        dist = abs(entry_price - cur_stop)
                        qty = max(P["min_contracts"],
                                  _round2_half_away((risk / dist) / P["point_value"])) if dist > 0 \
                            else P["min_contracts"]
                        pending = dict(dir=1, qty=qty, i=i)
                        entry_taken = True
                        cur_tp = entry_price + P["tp_atr_multiplier"] * atr[i]
                        trade_count += 1
                        long_sig = False
                        if collect_signals:
                            signals.append(dict(strategy="dmi", signal_i=i,
                                                signal_ms=int(utc_ms[i]), direction=1,
                                                stop=cur_stop, tp=cur_tp, qty=qty,
                                                stop_dist=float(dist),
                                                adx=float(adx[i]), di_spread=float(di_p[i] - di_m[i]),
                                                atr=float(atr[i]),
                                                kind="entry" if pos is None else
                                                ("reversal" if pos.direction < 0 else "rejected")))
            elif short_sig and dmi_short_ok[i]:
                if l[i] <= potential - P["trail_start_multiplier"] * atr[i]:
                    new_stop = l[i] + atr_m * atr[i]
                    if new_stop < initial_stop:
                        entry_price = c[i]
                        cur_stop = new_stop
                        dist = abs(entry_price - cur_stop)
                        qty = max(P["min_contracts"],
                                  _round2_half_away((risk / dist) / P["point_value"])) if dist > 0 \
                            else P["min_contracts"]
                        pending = dict(dir=-1, qty=qty, i=i)
                        entry_taken = True
                        cur_tp = entry_price - P["tp_atr_multiplier"] * atr[i]
                        trade_count += 1
                        short_sig = False
                        if collect_signals:
                            signals.append(dict(strategy="dmi", signal_i=i,
                                                signal_ms=int(utc_ms[i]), direction=-1,
                                                stop=cur_stop, tp=cur_tp, qty=qty,
                                                stop_dist=float(dist),
                                                adx=float(adx[i]), di_spread=float(di_p[i] - di_m[i]),
                                                atr=float(atr[i]),
                                                kind="entry" if pos is None else
                                                ("reversal" if pos.direction > 0 else "rejected")))

        # management block (only when entryTaken survived the day reset)
        if entry_taken and pos is not None:
            if pos.direction > 0:
                trail = h[i] - atr_m * atr[i]
                if not math.isnan(cur_stop) and trail > cur_stop:
                    cur_stop = trail
            else:
                trail = l[i] + atr_m * atr[i]
                if not math.isnan(cur_stop) and trail < cur_stop:
                    cur_stop = trail
            order_stop = cur_stop
            order_limit = cur_tp
            order_active = not (math.isnan(order_stop) and math.isnan(order_limit))

    if pos is not None:
        pos.close(utc_ms[n - 1], c[n - 1], "open_at_end", 0.0)
        equity += pos.pnl
        trades.append(pos)

    return dict(trades=trades_to_frame(trades), equity_end=equity, signals=signals)
