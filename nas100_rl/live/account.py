"""Internal model account: a 100k-scale mirror of rl/env.SignalEnv.

The policy's portfolio-state inputs come from THIS account, never from raw
broker equity (swap/slippage/rounding would push the observations out of the
training distribution). Broker reality is journaled separately and reconciled
nightly against this account.

mode="parity"  — selftest: trades carry their precomputed lifecycle (entry
                 price known at decision time, settle by exit_ms), bit-matching
                 SignalEnv so decisions can be asserted identical.
mode="live"    — entries are pending until the fill is known (price unknown ->
                 marks 0 until filled; risk is still reserved, like an order in
                 flight), settlement happens on mechanical-closure events.
"""
from __future__ import annotations

import math

import numpy as np

from .. import config
from ..rl.env import native_qty

R = config.RL


class ModelAccount:
    def __init__(self, mode: str = "live", equity0: float | None = None):
        assert mode in ("parity", "live")
        self.mode = mode
        self.e0 = float(equity0 if equity0 is not None else R["initial_capital"])
        self.cash = self.e0
        self.open: list[dict] = []
        self.peak = self.e0
        self.prev_mark_eq = self.e0
        self.mark_eq = self.e0
        self.ret_hist: list[float] = []
        self.ledger: list[dict] = []
        self.n_decisions = 0

    # ------------------------------------------------------------- state io
    def to_state(self) -> dict:
        return dict(mode=self.mode, e0=self.e0, cash=self.cash, open=self.open,
                    peak=self.peak, prev_mark_eq=self.prev_mark_eq,
                    ret_hist=self.ret_hist[-200:], n_decisions=self.n_decisions)

    @classmethod
    def from_state(cls, st: dict) -> "ModelAccount":
        a = cls(mode=st["mode"], equity0=st["e0"])
        a.cash = st["cash"]; a.open = st["open"]; a.peak = st["peak"]
        a.prev_mark_eq = st["prev_mark_eq"]; a.ret_hist = list(st["ret_hist"])
        a.n_decisions = st["n_decisions"]
        return a

    # ------------------------------------------------------------- mechanics
    def _settle_due(self, now_ms: int):
        """parity mode: settle trades whose precomputed exit_ms <= now."""
        still = []
        for t in self.open:
            if t.get("exit_ms") is not None and t["exit_ms"] <= now_ms:
                self.cash += t["qty"] * t["pc_pnl"]
                self.ledger.append(dict(trade_key=t["trade_key"], strategy=t["strategy"],
                                        qty=t["qty"], pnl=t["qty"] * t["pc_pnl"],
                                        entry_ms=t["entry_ms"], exit_ms=t["exit_ms"]))
            else:
                still.append(t)
        self.open = still

    def on_fill(self, trade_key, entry_price: float):
        for t in self.open:
            if t["trade_key"] == trade_key:
                t["entry_price"] = float(entry_price)

    def on_close(self, trade_key, pc_pnl: float, exit_ms: int):
        """live mode: mechanical closure event from the engines."""
        still = []
        for t in self.open:
            if t["trade_key"] == trade_key:
                self.cash += t["qty"] * pc_pnl
                self.ledger.append(dict(trade_key=trade_key, strategy=t["strategy"],
                                        qty=t["qty"], pnl=t["qty"] * pc_pnl,
                                        entry_ms=t["entry_ms"], exit_ms=exit_ms))
            else:
                still.append(t)
        self.open = still

    def _mark(self, px: float) -> float:
        eq = self.cash
        for t in self.open:
            if t["entry_price"] is not None and not math.isnan(t["entry_price"]):
                eq += t["qty"] * (px - t["entry_price"]) * t["direction"]
        self.mark_eq = eq
        self.peak = max(self.peak, eq)
        return eq

    def obs_port(self, n_feat_total: int = None) -> np.ndarray:
        eq = self.mark_eq
        open_risk = sum(t["qty"] * t["stop_dist"] for t in self.open) / max(eq, 1e-9)
        slope = float(np.mean(self.ret_hist[-10:])) * 100 if self.ret_hist else 0.0
        return np.array([len(self.open) / 3.0,
                         open_risk / R["total_open_risk_cap"],
                         (1.0 - eq / self.peak) * 10.0,
                         np.clip(slope, -5, 5)], dtype=np.float32)

    # ------------------------------------------------------------- decision
    def on_signal(self, sig_ms: int, mark_price: float) -> None:
        """Bring the account to the decision moment (SignalEnv step-tail order:
        settle exits <= now, mark, append inter-decision log return)."""
        if self.mode == "parity":
            self._settle_due(sig_ms)
        eq = self._mark(mark_price)
        if self.n_decisions > 0:
            r = math.log(max(eq, 1e-9) / max(self.prev_mark_eq, 1e-9))
            self.ret_hist.append(r)
        self.prev_mark_eq = eq
        self.n_decisions += 1

    def size(self, strategy: str, stop_dist: float, mult: float) -> float:
        """Native sizing x action multiplier under the hard caps (exact
        SignalEnv.step arithmetic, including pending trades reserving risk)."""
        eq = self.mark_eq
        if mult <= 0 or eq <= 0:
            return 0.0
        qty = native_qty(strategy, eq, stop_dist) * mult
        qty = min(qty, R["per_trade_risk_cap"] * eq / stop_dist)
        open_risk = sum(t["qty"] * t["stop_dist"] for t in self.open)
        room = R["total_open_risk_cap"] * eq - open_risk
        if room <= 0:
            return 0.0
        qty = min(qty, room / stop_dist)
        return qty if qty > 1e-9 else 0.0

    def register(self, trade_key, strategy: str, direction: int, qty: float,
                 stop_dist: float, entry_ms: int, entry_price: float | None,
                 exit_ms: int | None = None, pc_pnl: float | None = None):
        self.open.append(dict(trade_key=trade_key, strategy=strategy,
                              direction=int(direction), qty=float(qty),
                              stop_dist=float(stop_dist), entry_ms=int(entry_ms),
                              entry_price=entry_price,
                              exit_ms=exit_ms, pc_pnl=pc_pnl))
