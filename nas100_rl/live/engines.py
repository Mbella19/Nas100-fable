"""Windowed-replay engines: run the FROZEN strategy code incrementally.

No strategy file is modified and no logic is re-implemented. Each step re-runs
the verified batch strategy over a trailing window of 1-min bars that is long
enough for (a) bit-convergent indicators (RMA/EMA seed influence < float eps),
(b) all path-dependent state (open position, per-day state, CPMT pattern/pivot
buffers) to coincide with the full-history run. Equivalence with the frozen
signal stream is asserted empirically by `runner --selftest` — if a window is
ever too short, the selftest fails loudly rather than trading on divergence.

Outputs per step:
  emissions — mechanically-taken signals (the RL decision points), with the
              signal-time context fields the feature builder needs and the
              absolute stop/tp levels the executor needs;
  closed    — newly CLOSED mechanical trades as unified-stream rows (same
              columns/economics as rl/signals.py, basis="live").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, data


def _spread_at(m1: pd.DataFrame, t_ms: int) -> float:
    ms = m1["utc_ms"].to_numpy()
    i = int(np.clip(np.searchsorted(ms, t_ms, side="right") - 1, 0, len(ms) - 1))
    return float(m1["spread"].iloc[i]) * 0.1          # MT5 points -> index units


def _live_pc_pnl(row: dict, m1: pd.DataFrame, basis: str) -> tuple[float, float]:
    pc = (row["exit_price"] - row["entry_price"]) * row["direction"] - row["commission_pc"]
    cost = 0.0
    if basis == "live":
        cost = _spread_at(m1, row["entry_ms"] if row["direction"] > 0 else row["exit_ms"])
        pc -= cost
    return pc, cost


class StrategyEngine:
    """One frozen strategy, replayed over a window. Cadence-independent: step()
    may be called per chart bar (live) or per day (selftest) — emissions and
    closures depend only on the data, not on call frequency."""

    def __init__(self, name: str, basis: str = "live", live_from_ms: int = 0,
                 seen_signals: set | None = None, seen_entries: set | None = None):
        assert name in ("s2", "dmi", "cpmt")
        self.name = name
        self.basis = basis
        self.live_from_ms = int(live_from_ms)   # no decisions before this time
        L = config.LIVE
        self.barlen_ms = {"s2": 300_000, "dmi": 60_000, "cpmt": 1_800_000}[name]
        self.window_days = {"s2": L["s2_window_days"], "dmi": L["dmi_window_days"],
                            "cpmt": None}[name]
        self.anchor = None if self.window_days else pd.Timestamp(L["cpmt_anchor"])
        # watermarks: what the stream already contains (seeded from the frozen parquet)
        self.seen_signals = seen_signals if seen_signals is not None else set()
        self.seen_entries = seen_entries if seen_entries is not None else set()

    # ------------------------------------------------------------------ window
    def _window(self, m1: pd.DataFrame) -> pd.DataFrame:
        if self.anchor is not None:
            w = m1[m1["srv"] >= self.anchor]
        else:
            last_day = m1["srv"].iloc[-1].normalize()
            w = m1[m1["srv"] >= last_day - pd.Timedelta(days=self.window_days)]
        return w.reset_index(drop=True)

    # ------------------------------------------------------------------ run
    def _run(self, w: pd.DataFrame) -> dict:
        if self.name == "s2":
            from ..strategies import s2
            bars5 = data.resample(w, 5)
            return s2.run(bars5, w, collect_signals=True)
        if self.name == "dmi":
            from ..strategies import dmi
            return dmi.run(w, collect_signals=True)
        from ..strategies.cpmt import strategy as cpmt
        frames = {tf: data.resample(w, int(tf))
                  for tf in ("240", "180", "120", "360", "60", "30")}
        d = data.daily(w)
        return cpmt.run(frames["30"], w, frames, d, collect_signals=True)

    # ------------------------------------------------------------------ step
    def step(self, m1: pd.DataFrame) -> dict:
        w = self._window(m1)
        res = self._run(w)
        emissions = self._new_emissions(res["signals"])
        tr = merge_signal_context(res["trades"], res["signals"], self.name)
        closed = self._new_closed(tr, w)
        return dict(emissions=emissions, closed=closed)

    def _mech_taken(self, s: dict) -> bool:
        if self.name == "dmi":
            return s.get("kind") != "rejected"
        return bool(s.get("taken"))

    def _new_emissions(self, signals: list[dict]) -> list[dict]:
        out = []
        for s in signals:
            if not self._mech_taken(s):
                continue
            if int(s["signal_ms"]) < self.live_from_ms:
                continue
            key = (int(s["signal_ms"]), int(s["direction"]))
            if key in self.seen_signals:
                continue
            self.seen_signals.add(key)
            e = dict(strategy=self.name, signal_ms=int(s["signal_ms"]),
                     direction=int(s["direction"]),
                     stop_dist=float(s.get("stop_dist", s.get("dist"))),
                     atr_sig=float(s["atr"]),
                     nth_of_day=int(s.get("nth", 1)),
                     adx_sig=float(s.get("adx", np.nan)),
                     di_spread=float(s.get("di_spread", np.nan)),
                     pattern=str(s.get("pattern", "")),
                     stream=int(s.get("stream", -1)),
                     width=float(s.get("width", np.nan)),
                     commission_pc=2 * config.S2["commission_per_contract_per_order"]
                     if self.name == "s2" else 0.0)
            # absolute protective levels for the broker safety net + exec context
            e["sl_level"] = float(s.get("sl", s.get("stop", np.nan)))
            e["tp_level"] = float(s.get("tp", np.nan))
            e["kind"] = str(s.get("kind", "entry"))
            e["unit"] = float(s.get("unit", np.nan))
            e["dead_bars"] = int(s.get("dead_bars", -1))
            out.append(e)
        return out

    def _new_closed(self, tr: pd.DataFrame, w: pd.DataFrame) -> list[dict]:
        """Map newly closed trades to unified-stream rows (mirrors rl/signals.py
        column construction; equality with the frozen parquet is selftested)."""
        if tr is None or not len(tr):
            return []
        sigless = []
        for r in tr.itertuples():
            entry_ms = int(r.entry_ms)
            if entry_ms in self.seen_entries:
                continue
            # the strategy reports a STILL-OPEN position at the window's last
            # bar as a pseudo-trade (exit_reason="open_at_end"): not a closure —
            # leave it unseen so the real exit is emitted by a later step
            if str(r.exit_reason) == "open_at_end":
                continue
            # unseen entries far older than live start are window-warmup
            # artifacts (e.g. the anchor's first weeks, where full-history state
            # differed); real pre-live trades are seeded from the frozen parquet
            if entry_ms < self.live_from_ms - 30 * 86_400_000:
                self.seen_entries.add(entry_ms)
                continue
            exit_ms = int(getattr(r, "exit_ms_exact", float("nan"))
                          if not pd.isna(getattr(r, "exit_ms_exact", float("nan")))
                          else r.exit_ms)
            self.seen_entries.add(entry_ms)
            row = dict(strategy=self.name,
                       signal_ms=entry_ms - self.barlen_ms,
                       direction=int(r.direction),
                       entry_ms=entry_ms,
                       entry_price=float(r.entry_price),
                       exit_ms=exit_ms,
                       exit_price=float(r.exit_price),
                       exit_reason=str(r.exit_reason),
                       commission_pc=2 * config.S2["commission_per_contract_per_order"]
                       if self.name == "s2" else 0.0)
            if self.name == "s2":
                row.update(stop_dist=float(r.stop_dist), atr_sig=float(r.atr),
                           nth_of_day=int(r.nth), adx_sig=np.nan, di_spread=np.nan,
                           pattern="", stream=-1, width=np.nan)
            elif self.name == "dmi":
                row.update(stop_dist=float(r.stop_dist), atr_sig=float(r.atr),
                           nth_of_day=1, adx_sig=float(r.adx),
                           di_spread=float(r.di_spread),
                           pattern="", stream=-1, width=np.nan)
            else:
                row.update(stop_dist=float(r.dist), atr_sig=float(r.atr),
                           nth_of_day=1, adx_sig=np.nan, di_spread=np.nan,
                           pattern=str(r.sig_pattern) if hasattr(r, "sig_pattern")
                           else str(getattr(r, "pattern", "")),
                           stream=int(getattr(r, "sig_stream", getattr(r, "stream", -1))),
                           width=float(r.width))
            pc, cost = _live_pc_pnl(row, w, self.basis)
            row["pc_pnl"] = pc
            row["spread_cost_pc"] = cost
            row["r_mult"] = pc / row["stop_dist"]
            row["hold_ms"] = row["exit_ms"] - row["entry_ms"]
            sigless.append(row)
        return sigless


def merge_signal_context(tr: pd.DataFrame, signals: list[dict], name: str) -> pd.DataFrame:
    """Attach signal-time context (stop_dist/atr/...) to raw trades, exactly as
    rl/signals.py does, so _new_closed sees merged columns."""
    sig = pd.DataFrame([s for s in signals])
    if not len(sig) or not len(tr):
        return tr
    if name == "s2":
        sig = sig[sig["taken"]]
        return tr.merge(sig[["signal_i", "stop_dist", "atr", "nth"]].rename(
            columns={"nth": "sig_nth"}), on="signal_i", how="left")
    if name == "dmi":
        sig = sig[sig["kind"] != "rejected"]
        return tr.merge(sig[["signal_i", "stop_dist", "atr", "adx", "di_spread"]],
                        on="signal_i", how="left")
    sig = sig[sig["taken"]]
    return tr.merge(sig[["signal_i", "dist", "atr", "pattern", "stream", "width"]]
                    .rename(columns={"pattern": "sig_pattern", "stream": "sig_stream"}),
                    on="signal_i", how="left")
