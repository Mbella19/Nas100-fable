"""Portfolio environment over the unified signal stream.

One shared account. Decision points = signal events (time-ordered). Actions are
size multipliers {0, 0.5, 1.0, 1.5} applied to each strategy's NATIVE sizing rule
evaluated at current portfolio equity, under hard risk caps (per-trade 3%, total
open 6%). Entries/exits/stops are the verified mechanical lifecycles — the agent
only chooses participation and scale.

Native sizing at portfolio capital (100k):
  S2  : clamp(E * 0.006 / stop_dist, 1, 200)            (native, designed at 100k)
  CPMT: floor(E * 0.02 / stop_dist / 0.01)*0.01, <=1000 (clamps scaled by capital ratio)
  DMI : max(0.01, round2(2000 / stop_dist))             (fixed-$ risk, scaled 10k->100k)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config

R = config.RL
N_PORT_FEATS = 4


def native_qty(strategy: str, equity: float, stop_dist: float) -> float:
    if strategy == "s2":
        return max(config.S2["qty_min"], min(config.S2["qty_max"],
                                             equity * config.S2["risk_pct"] / stop_dist))
    if strategy == "cpmt":
        q = math.floor(equity * config.CPMT["risk_pct"] / 100.0 / stop_dist
                       / config.CPMT["qty_step"]) * config.CPMT["qty_step"]
        return min(max(q, 0.0), R["cpmt_qty_max"])
    # dmi
    q = R["dmi_risk_dollars"] / stop_dist
    q = math.floor(q * 100 + 0.5) / 100
    return max(config.DMI["min_contracts"], q)


@dataclass
class OpenTrade:
    trade_id: int
    strategy: str
    direction: int
    qty: float
    entry_price: float
    stop_dist: float
    exit_ms: int
    exit_price: float
    pc_pnl: float
    entry_ms: int


class SignalEnv:
    """Steps over signals[i0:i1]. Policy-agnostic; used for RL training, baselines
    and evaluation (same code path for fairness)."""

    def __init__(self, sig: pd.DataFrame, feats: np.ndarray, mark_price: np.ndarray,
                 equity0: float = None, reward_scale: float = 100.0,
                 dd_penalty: float = 0.0, var_penalty: float = 0.0,
                 credit: str = "trade"):
        """credit="trade": contextual-bandit reward — each decision is immediately
        rewarded with its own trade's equity contribution (mean-variance utility);
        sharp credit assignment for sparse decisions. credit="portfolio": reward is
        the marked portfolio log-return between decisions (needs gamma>0)."""
        self.sig = sig.reset_index(drop=True)
        self.feats = feats.astype(np.float32)
        self.mark = mark_price            # per-signal mark price (last closed 30m close)
        self.e0 = float(equity0 if equity0 is not None else R["initial_capital"])
        self.reward_scale = reward_scale
        self.dd_penalty = dd_penalty
        self.var_penalty = var_penalty    # kappa: reward = r - (kappa/2) r^2 (log-utility)
        self.credit = credit
        self.s_ms = self.sig["signal_ms"].to_numpy()
        self.s_strat = self.sig["strategy"].to_numpy()
        self.s_dir = self.sig["direction"].to_numpy()
        self.s_sd = self.sig["stop_dist"].to_numpy()
        self.s_entry_ms = self.sig["entry_ms"].to_numpy()
        self.s_entry_px = self.sig["entry_price"].to_numpy()
        self.s_exit_ms = self.sig["exit_ms"].to_numpy()
        self.s_exit_px = self.sig["exit_price"].to_numpy()
        self.s_pc = self.sig["pc_pnl"].to_numpy()
        self.s_tid = self.sig["trade_id"].to_numpy()
        self.obs_dim = self.feats.shape[1] + N_PORT_FEATS
        self.n_actions = len(R["action_multipliers"])

    # ------------------------------------------------------------------ core
    def reset(self, i0: int, i1: int) -> np.ndarray:
        self.i0, self.i1 = i0, i1
        self.i = i0
        self.cash = self.e0
        self.open: list[OpenTrade] = []
        self.peak = self.e0
        self.prev_mark_eq = self.e0
        self.ret_hist: list[float] = []
        self.ledger: list[dict] = []
        self._settle_and_mark(self.s_ms[self.i])
        return self._obs()

    def _settle_and_mark(self, now_ms: int) -> float:
        still = []
        for t in self.open:
            if t.exit_ms <= now_ms:
                self.cash += t.qty * t.pc_pnl
                self.ledger.append(dict(trade_id=t.trade_id, strategy=t.strategy,
                                        qty=t.qty, pnl=t.qty * t.pc_pnl,
                                        entry_ms=t.entry_ms, exit_ms=t.exit_ms))
            else:
                still.append(t)
        self.open = still
        px = self.mark[self.i] if self.i < self.i1 else self.mark[self.i1 - 1]
        eq = self.cash + sum(t.qty * (px - t.entry_price) * t.direction for t in self.open)
        self.mark_eq = eq
        self.peak = max(self.peak, eq)
        return eq

    def _obs(self) -> np.ndarray:
        i = min(self.i, self.i1 - 1)
        eq = self.mark_eq
        open_risk = sum(t.qty * t.stop_dist for t in self.open) / max(eq, 1e-9)
        slope = float(np.mean(self.ret_hist[-10:])) * 100 if self.ret_hist else 0.0
        port = np.array([len(self.open) / 3.0,
                         open_risk / R["total_open_risk_cap"],
                         (1.0 - eq / self.peak) * 10.0,
                         np.clip(slope, -5, 5)], dtype=np.float32)
        return np.concatenate([self.feats[i], port])

    def step(self, action: int):
        i = self.i
        mult = R["action_multipliers"][int(action)]
        eq = self.mark_eq
        took = False
        qty = 0.0
        if mult > 0 and eq > 0:
            qty = native_qty(self.s_strat[i], eq, self.s_sd[i]) * mult
            # hard caps: per-trade and total open risk (downgrade, never upgrade)
            qty = min(qty, R["per_trade_risk_cap"] * eq / self.s_sd[i])
            open_risk = sum(t.qty * t.stop_dist for t in self.open)
            room = R["total_open_risk_cap"] * eq - open_risk
            if room <= 0:
                qty = 0.0
            else:
                qty = min(qty, room / self.s_sd[i])
            if qty > 1e-9:
                took = True
                self.open.append(OpenTrade(
                    trade_id=int(self.s_tid[i]), strategy=self.s_strat[i],
                    direction=int(self.s_dir[i]), qty=float(qty),
                    entry_price=float(self.s_entry_px[i]), stop_dist=float(self.s_sd[i]),
                    exit_ms=int(self.s_exit_ms[i]), exit_price=float(self.s_exit_px[i]),
                    pc_pnl=float(self.s_pc[i]), entry_ms=int(self.s_entry_ms[i])))

        trade_ret = (qty * self.s_pc[i]) / max(eq, 1e-9) if took else 0.0

        self.i += 1
        done = self.i >= self.i1
        if done:
            # settle everything at the final signal's mark
            last_ms = int(self.s_ms[self.i1 - 1])
            eq_new = self._final_equity(last_ms)
        else:
            eq_new = self._settle_and_mark(self.s_ms[self.i])

        r = math.log(max(eq_new, 1e-9) / max(self.prev_mark_eq, 1e-9))
        self.ret_hist.append(r)
        dd_now = 1.0 - eq_new / self.peak
        if self.credit == "trade":
            rs = trade_ret * self.reward_scale
        else:
            rs = r * self.reward_scale
        # mean-variance utility on the scaled return: kappa penalizes big swings
        # quadratically -> Sharpe-aligned behavior
        reward = rs - 0.5 * self.var_penalty * rs * rs \
            - self.dd_penalty * max(0.0, dd_now) * abs(rs)
        self.prev_mark_eq = eq_new
        info = dict(equity=eq_new, took=took, qty=qty)
        return (None if done else self._obs()), float(reward), done, info

    def _final_equity(self, now_ms: int) -> float:
        # settle exits known by now; mark the rest at their eventual exit price
        # (episode-end convention, identical for all policies)
        for t in self.open:
            self.cash += t.qty * t.pc_pnl
            self.ledger.append(dict(trade_id=t.trade_id, strategy=t.strategy,
                                    qty=t.qty, pnl=t.qty * t.pc_pnl,
                                    entry_ms=t.entry_ms, exit_ms=t.exit_ms))
        self.open = []
        self.mark_eq = self.cash
        self.peak = max(self.peak, self.cash)
        return self.cash


def run_fixed_policy(env: SignalEnv, i0: int, i1: int, policy) -> dict:
    """policy: callable(obs, signal_row_index) -> action int. Returns summary."""
    obs = env.reset(i0, i1)
    done = False
    while not done:
        a = policy(obs, env.i)
        obs, r, done, info = env.step(a)
    return dict(equity_end=env.mark_eq, ledger=pd.DataFrame(env.ledger))
