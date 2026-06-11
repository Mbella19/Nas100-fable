"""TV-semantics execution primitives shared by all three strategy ports.

Conventions replicated (process_orders_on_close = false, bar magnifier = on):
- Market orders placed at chart-bar close fill at the NEXT chart bar's open.
- Stop/limit exit orders are walked over the 1-minute sub-bars of each chart bar
  (we have real 1-min data = TV bar magnifier on intraday charts).
- A stop fills at the stop price, or at the sub-bar's open when it gaps through.
- When stop AND limit could both fill inside one 1-min sub-bar, the documented TV
  broker-emulator path is used: open -> nearest extreme -> other extreme -> close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    strategy: str
    direction: int                  # +1 long, -1 short
    qty: float
    entry_ms: int                   # UTC epoch ms of fill
    entry_price: float
    exit_ms: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    commission: float = 0.0
    pnl: float = 0.0                # net of commission
    meta: dict = field(default_factory=dict)

    def close(self, exit_ms: int, exit_price: float, reason: str, commission: float) -> None:
        self.exit_ms = int(exit_ms)
        self.exit_price = float(exit_price)
        self.exit_reason = reason
        self.commission += commission
        self.pnl = (self.exit_price - self.entry_price) * self.direction * self.qty - self.commission


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append(dict(
            strategy=t.strategy, direction=t.direction, qty=t.qty,
            entry_ms=t.entry_ms, entry_price=t.entry_price,
            exit_ms=t.exit_ms, exit_price=t.exit_price, exit_reason=t.exit_reason,
            commission=t.commission, pnl=t.pnl, **t.meta,
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df["entry_time"] = pd.to_datetime(df["entry_ms"], unit="ms", utc=True)
        df["exit_time"] = pd.to_datetime(df["exit_ms"], unit="ms", utc=True)
    return df


class M1Map:
    """Maps chart bars to their 1-minute sub-bar index ranges."""

    def __init__(self, m1: pd.DataFrame, chart_srv: pd.Series, chart_minutes: int):
        self.o = m1["open"].to_numpy()
        self.h = m1["high"].to_numpy()
        self.l = m1["low"].to_numpy()
        self.c = m1["close"].to_numpy()
        self.ms = m1["utc_ms"].to_numpy()
        srv = m1["srv"].to_numpy()
        self.start = np.searchsorted(srv, chart_srv.to_numpy(), side="left")
        end_times = (chart_srv + pd.Timedelta(minutes=chart_minutes)).to_numpy()
        self.end = np.searchsorted(srv, end_times, side="left")

    def range(self, bar_idx: int) -> tuple[int, int]:
        return int(self.start[bar_idx]), int(self.end[bar_idx])


def path_order_high_first(o: float, h: float, low: float) -> bool:
    """TV broker emulator: if the high is closer to the open, price visits high first."""
    return (h - o) <= (o - low)


def walk_exit(o: float, h: float, low: float, c: float, direction: int,
              stop: float | None, limit: float | None) -> tuple[float, str] | None:
    """Resolve stop/limit fills inside ONE sub-bar following the TV path assumption.

    Returns (price, reason) or None. Gap-through: if the open is already beyond the
    level, fills at the open.
    """
    stop_hit = stop is not None and (low <= stop if direction > 0 else h >= stop)
    lim_hit = limit is not None and (h >= limit if direction > 0 else low <= limit)
    if not stop_hit and not lim_hit:
        return None

    def stop_fill() -> tuple[float, str]:
        if direction > 0:
            return (o if o <= stop else stop), "stop"
        return (o if o >= stop else stop), "stop"

    def limit_fill() -> tuple[float, str]:
        if direction > 0:
            return (o if o >= limit else limit), "limit"
        return (o if o <= limit else limit), "limit"

    if stop_hit and not lim_hit:
        return stop_fill()
    if lim_hit and not stop_hit:
        return limit_fill()
    # both inside one sub-bar: gap-through at open wins outright, else path order
    if direction > 0 and o <= stop:
        return o, "stop"
    if direction < 0 and o >= stop:
        return o, "stop"
    if direction > 0 and o >= limit:
        return o, "limit"
    if direction < 0 and o <= limit:
        return o, "limit"
    high_first = path_order_high_first(o, h, low)
    if direction > 0:
        # long: limit above (at high side), stop below (at low side)
        return limit_fill() if high_first else stop_fill()
    return stop_fill() if high_first else limit_fill()


def walk_stop_over_range(m1: M1Map, i0: int, i1: int, direction: int,
                         stop: float) -> tuple[int, float] | None:
    """First 1-min bar in [i0, i1) where a static stop fills. Returns (m1_idx, price)."""
    if direction > 0:
        for j in range(i0, i1):
            if m1.l[j] <= stop:
                return j, (m1.o[j] if m1.o[j] <= stop else stop)
    else:
        for j in range(i0, i1):
            if m1.h[j] >= stop:
                return j, (m1.o[j] if m1.o[j] >= stop else stop)
    return None
