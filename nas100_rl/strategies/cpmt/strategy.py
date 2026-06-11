"""CPMT v12 — 30-minute chart strategy layer over the pattern engines.

Replicates the Pine script in netted "Both (single position)" mode as exported:
- Six streams (4h, 3h, 2h, 6h, 1h, 30m), each running 3-4 pattern engines with
  different pivot lengths; events surface on the 30m bar whose close time equals
  the HTF bar close (request.security lookahead-off semantics).
- Candidate score = pivot_len*100 + stream_order, lower wins; long preferred on
  ties. Gated streams (all but 4h) require the daily soft gate; blackout hours
  {23,0,1} fixed GMT+3 block all entries.
- Stop distance = clamp(|line - invalidation|, 1.0, 2.5) * stream ATR(14), measured
  from the broken LINE price (not the fill); trail unit = 2.5 * stream ATR fixed at
  signal; trail ratchets on prior 30m-bar extremes; time stop = signal bar +
  round(3 * width * chart_bars_per_htf_bar); exits fill on the 1-min path.
- qty = floor(realized_equity * 2% / dist / 0.01) * 0.01, clamped [0.01, 100].
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ... import config, indicators
from ...engine import M1Map, Trade, trades_to_frame, walk_stop_over_range
from .patterns import Engine

P = config.CPMT


def _round_half_away(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def build_stream_events(htf: pd.DataFrame, piv_lens: tuple[int, ...]) -> dict[int, dict]:
    """Run the engines of one stream over its HTF bars.

    Returns {close_ms: {"L": (px, inv, w, piv_len, name) | None, "S": ..., "atr": a14}}
    with engine-priority merge (first engine in piv_lens order wins each side).
    Only bars that emitted at least one break event get an entry.
    """
    h = htf["high"].to_numpy()
    l = htf["low"].to_numpy()
    c = htf["close"].to_numpy()
    close_ms = htf["close_ms"].to_numpy()
    a14 = indicators.atr(h, l, c, 14)
    engines = [Engine(P["tol"], P["max_width"], P["max_pat"]) for _ in piv_lens]
    pivs = [(indicators.pivot_high(h, p, p), indicators.pivot_low(l, p, p)) for p in piv_lens]
    out: dict[int, dict] = {}
    n = len(htf)
    for t in range(n):
        long_ev = None
        short_ev = None
        for eng, p, (ph, pl) in zip(engines, piv_lens, pivs):
            ev = eng.step(t, h[t], l[t], c[t], ph[t], pl[t], p)
            if long_ev is None and ev.up_px == ev.up_px:
                long_ev = (float(ev.up_px), float(ev.up_inv), float(ev.up_w), p, ev.up_name)
            if short_ev is None and ev.dn_px == ev.dn_px:
                short_ev = (float(ev.dn_px), float(ev.dn_inv), float(ev.dn_w), p, ev.dn_name)
        if long_ev is not None or short_ev is not None:
            out[int(close_ms[t])] = {"L": long_ev, "S": short_ev, "atr": float(a14[t])}
    return out


def run(bars30: pd.DataFrame, m1: pd.DataFrame, frames: dict[str, pd.DataFrame],
        daily: pd.DataFrame, initial_equity: float | None = None,
        collect_signals: bool = False, qty_max: float | None = None,
        trade_from_ms: int = 0) -> dict:
    """`trade_from_ms`: engines and state run from the data start (warmup), but
    entries are only placed on bars at/after this UTC-ms timestamp — so
    `initial_equity` is the equity AT that boundary (TV re-basing)."""
    """frames: {"240": htf_bars, "180": ..., "120": ..., "360": ..., "60": ..., "30": bars30}"""
    o = bars30["open"].to_numpy()
    h = bars30["high"].to_numpy()
    l = bars30["low"].to_numpy()
    c = bars30["close"].to_numpy()
    close_ms = bars30["close_ms"].to_numpy()
    utc_ms = bars30["utc_ms"].to_numpy()
    n = len(bars30)

    # stream events keyed by 30m close_ms
    streams = []
    for (tf, piv_lens, k), gated in zip(P["streams"], P["gated"]):
        streams.append(dict(events=build_stream_events(frames[tf], piv_lens),
                            k=k, gated=gated))

    # daily soft gate: last CLOSED daily bar vs EMA(10) +/- 0.25*ATR(14)
    dc = daily["close"].to_numpy()
    d_ema = indicators.ema(dc, P["gate_len"])
    d_atr = indicators.atr(daily["high"].to_numpy(), daily["low"].to_numpy(), dc, 14)
    day_key = daily["srv_day"].to_numpy()
    bar_day = bars30["srv"].dt.normalize().to_numpy()
    d_idx = np.searchsorted(day_key, bar_day, side="left")  # index of current day
    g_trend = np.zeros(n, dtype=np.int64)
    prev = d_idx - 1
    okp = prev >= 0
    gc = np.where(okp, dc[np.clip(prev, 0, None)], np.nan)
    ge = np.where(okp, d_ema[np.clip(prev, 0, None)], np.nan)
    ga = np.where(okp, d_atr[np.clip(prev, 0, None)], np.nan)
    with np.errstate(invalid="ignore"):
        up = gc > ge + P["gate_band"] * ga
        dn = gc < ge - P["gate_band"] * ga
    g_trend[up & ~np.isnan(gc)] = 1
    g_trend[dn & ~np.isnan(gc)] = -1

    # blackout on the 30m bar's close hour in fixed GMT+3
    close_h_gmt3 = ((close_ms // 3_600_000 + 3) % 24).astype(np.int64)
    blackout = np.isin(close_h_gmt3, P["blackout_hours_gmt3"])

    m1map = M1Map(m1, bars30["srv"], 30)
    q_max = float(qty_max if qty_max is not None else P["qty_max"])

    equity = float(initial_equity if initial_equity is not None else P["initial_capital"])
    trades: list[Trade] = []
    signals: list[dict] = [] if collect_signals else None

    pos: Trade | None = None
    active_stop = math.nan
    slot_best = math.nan
    slot_unit = math.nan
    slot_dead: int | None = None
    pending_entry: dict | None = None
    pending_close = False

    for i in range(n):
        # --- 1) market fills at open -------------------------------------------
        if pending_close:
            if pos is not None:
                pos.close(utc_ms[i], o[i], "time_stop", 0.0)
                equity += pos.pnl
                trades.append(pos)
                pos = None
            pending_close = False
        if pending_entry is not None:
            e = pending_entry
            pos = Trade(strategy="cpmt", direction=e["dir"], qty=e["qty"],
                        entry_ms=int(utc_ms[i]), entry_price=float(o[i]),
                        meta=dict(signal_i=e["i"], pattern=e["name"], stream=e["stream"],
                                  piv_len=e["piv_len"], stop0=e["stop"]))
            active_stop = e["stop"]
            slot_unit = e["unit"]
            slot_best = math.nan
            slot_dead = e["dead"]
            pending_entry = None

        # --- 2) intrabar stop on the 1-min path --------------------------------
        if pos is not None and active_stop == active_stop:
            i0, i1 = m1map.range(i)
            hit = walk_stop_over_range(m1map, i0, i1, pos.direction, active_stop)
            if hit is not None:
                j, px = hit
                pos.meta["exit_ms_exact"] = int(m1map.ms[j])
                pos.close(utc_ms[i], px, "stop", 0.0)
                equity += pos.pnl
                trades.append(pos)
                pos = None

        # --- 3) bar-close logic --------------------------------------------------
        # manage first (mirrors script order: trail update then time stop)
        if pos is not None:
            if pos.direction > 0:
                slot_best = max(pos.entry_price, h[i]) if slot_best != slot_best \
                    else max(slot_best, h[i])
                active_stop = max(active_stop, slot_best - slot_unit)
            else:
                slot_best = min(pos.entry_price, l[i]) if slot_best != slot_best \
                    else min(slot_best, l[i])
                active_stop = min(active_stop, slot_best + slot_unit)
            if slot_dead is not None and i >= slot_dead:
                pending_close = True

        # entry selection when flat
        if pos is None and pending_entry is None and utc_ms[i] >= trade_from_ms:
            best_l = None  # (score, px, inv, w, atr, k, name, stream, piv_len)
            best_s = None
            if not blackout[i]:
                cm = int(close_ms[i])
                for order, st in enumerate(streams):
                    ev = st["events"].get(cm)
                    if ev is None:
                        continue
                    a = ev["atr"]
                    if not (a == a and a > 0):
                        continue
                    for side in ("L", "S"):
                        e = ev[side]
                        if e is None:
                            continue
                        is_long = side == "L"
                        if st["gated"] and not (g_trend[i] == 0
                                                or g_trend[i] == (1 if is_long else -1)):
                            continue
                        px, inv, w, piv_len, name = e
                        score = piv_len * 100 + order
                        cand = (score, px, inv, w, a, st["k"], name, order, piv_len)
                        if is_long:
                            if best_l is None or score < best_l[0]:
                                best_l = cand
                        else:
                            if best_s is None or score < best_s[0]:
                                best_s = cand
            if best_l is not None or best_s is not None:
                go_long = best_l is not None and (best_s is None or best_l[0] <= best_s[0])
                score, lvl, inv, w, a_s, k, name, order, piv_len = best_l if go_long else best_s
                raw = (lvl - inv) if go_long else (inv - lvl)
                if raw != raw:
                    raw = P["stop_cap"] * a_s
                dist = min(max(raw, P["stop_floor"] * a_s), P["stop_cap"] * a_s)
                stop = lvl - dist if go_long else lvl + dist
                qty = math.floor(equity * P["risk_pct"] / 100.0
                                 / (dist * P["point_value"]) / P["qty_step"]) * P["qty_step"]
                qty = min(qty, q_max)
                if collect_signals:
                    signals.append(dict(strategy="cpmt", signal_i=i, signal_ms=int(utc_ms[i]),
                                        direction=1 if go_long else -1, line=lvl, inv=inv,
                                        stop=stop, dist=float(dist), width=float(w),
                                        atr=float(a_s), pattern=name, stream=order,
                                        piv_len=int(piv_len),
                                        unit=P["trail_mult"] * a_s,
                                        dead_bars=_round_half_away(P["ts_mult"] * w * k)
                                        if P["ts_mult"] > 0 else -1,
                                        taken=qty >= P["qty_min"] and dist > 0))
                if qty >= P["qty_min"] and dist > 0 and i + 1 < n:
                    pending_entry = dict(
                        i=i, dir=1 if go_long else -1, qty=qty, stop=stop,
                        unit=P["trail_mult"] * a_s,
                        dead=(i + _round_half_away(P["ts_mult"] * w * k))
                        if P["ts_mult"] > 0 else None,
                        name=name, stream=order, piv_len=piv_len)

    if pos is not None:
        pos.close(utc_ms[n - 1], c[n - 1], "open_at_end", 0.0)
        equity += pos.pnl
        trades.append(pos)

    return dict(trades=trades_to_frame(trades), equity_end=equity, signals=signals)
