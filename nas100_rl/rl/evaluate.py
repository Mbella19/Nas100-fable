"""Evaluation: daily mark-to-market equity curves, metrics, baselines, success gate.

All policies (RL, always-take, per-strategy solo, random-gate) run through the SAME
SignalEnv code path and the same daily-curve construction, so comparisons are
apples-to-apples: shared $100k account per period, identical costs and caps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, data
from . import features as featmod
from . import signals as sigmod
from .env import SignalEnv, run_fixed_policy

A_SKIP, A_HALF, A_FULL, A_LARGE = 0, 1, 2, 3


def load_all(basis: str = "tv"):
    sig = sigmod.build(basis=basis)
    fdf, names = featmod.build(basis=basis)
    assert (fdf["trade_id"].to_numpy() == sig["trade_id"].to_numpy()).all()
    feats = fdf[names].to_numpy(dtype=np.float32)
    mark = fdf["mark_price"].to_numpy(dtype=np.float64)
    df1m = data.load_1min()
    g = df1m.groupby(df1m["srv"].dt.normalize())
    days = pd.DataFrame({"day_end_ms": g["utc_ms"].last(), "day_close": g["close"].last(),
                         "period": g["period"].first()})
    days.index.name = "srv_day"
    days = days.reset_index()
    periods = {}
    per = sig["period"].to_numpy()
    for p in ("train", "val", "oos"):
        idx = np.where(per == p)[0]
        periods[p] = (int(idx[0]), int(idx[-1]) + 1)
    return dict(sig=sig, feats=feats, mark=mark, days=days, periods=periods, names=names)


def daily_curve_range(ledger: pd.DataFrame, sig: pd.DataFrame, days: pd.DataFrame,
                      ms_lo: int, ms_hi: int, equity0: float) -> pd.DataFrame:
    dd = days[(days["day_end_ms"] >= ms_lo) & (days["day_end_ms"] <= ms_hi)].reset_index(drop=True)
    return _curve(ledger, sig, dd, equity0)


def daily_curve(ledger: pd.DataFrame, sig: pd.DataFrame, days: pd.DataFrame,
                period: str, equity0: float) -> pd.DataFrame:
    """Exact daily equity: realized PnL on exit days + mark-to-market of open trades
    at each server-day close."""
    dd = days[days["period"] == period].reset_index(drop=True)
    return _curve(ledger, sig, dd, equity0)


def _curve(ledger: pd.DataFrame, sig: pd.DataFrame, dd: pd.DataFrame,
           equity0: float) -> pd.DataFrame:
    ends = dd["day_end_ms"].to_numpy()
    closes = dd["day_close"].to_numpy()
    nd = len(dd)
    cash_delta = np.zeros(nd)
    mark_delta = np.zeros((nd,))
    eq_mark = np.zeros(nd)
    if len(ledger):
        led = ledger.merge(sig[["trade_id", "direction", "entry_price", "stop_dist"]],
                           on="trade_id", how="left")
        for r in led.itertuples():
            k_in = int(np.searchsorted(ends, r.entry_ms, side="left"))
            k_out = int(np.searchsorted(ends, r.exit_ms, side="left"))
            if k_out < nd:
                cash_delta[k_out] += r.pnl
            elif nd > 0:
                cash_delta[nd - 1] += r.pnl
            for k in range(k_in, min(k_out, nd)):
                eq_mark[k] += r.qty * (closes[k] - r.entry_price) * r.direction
    equity = equity0 + np.cumsum(cash_delta) + eq_mark
    return pd.DataFrame({"srv_day": dd["srv_day"], "equity": equity})


def metrics(curve: pd.DataFrame, equity0: float) -> dict:
    eq = curve["equity"].to_numpy()
    if len(eq) < 2:
        return dict(net=0.0, sharpe=0.0, maxdd=0.0, calmar=0.0, cagr=0.0)
    r = np.diff(np.log(np.maximum(eq, 1e-9)))
    sd = r.std()
    sharpe = float(r.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    maxdd = float(dd.max())
    years = len(eq) / 252.0
    cagr = float((eq[-1] / equity0) ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else -1.0
    calmar = cagr / maxdd if maxdd > 1e-9 else 0.0
    return dict(net=float(eq[-1] - equity0), sharpe=sharpe, maxdd=maxdd,
                calmar=float(calmar), cagr=cagr)


def eval_policy(world: dict, period: str, policy, equity0: float = None,
                dd_penalty: float = 0.0) -> dict:
    i0, i1 = world["periods"][period]
    env = SignalEnv(world["sig"], world["feats"], world["mark"], equity0=equity0,
                    dd_penalty=dd_penalty)
    res = run_fixed_policy(env, i0, i1, policy)
    e0 = env.e0
    curve = daily_curve(res["ledger"], world["sig"], world["days"], period, e0)
    m = metrics(curve, e0)
    m["n_trades"] = len(res["ledger"])
    if len(res["ledger"]):
        m["win_rate"] = float((res["ledger"]["pnl"] > 0).mean())
    else:
        m["win_rate"] = 0.0
    m["curve"] = curve
    m["ledger"] = res["ledger"]
    return m


def eval_range(world: dict, i0: int, i1: int, policy) -> dict:
    """Evaluate a policy on an arbitrary signal index range (walk-forward folds)."""
    env = SignalEnv(world["sig"], world["feats"], world["mark"])
    res = run_fixed_policy(env, i0, i1, policy)
    ms_lo = int(world["sig"]["signal_ms"].iloc[i0])
    ms_hi = int(world["sig"]["signal_ms"].iloc[i1 - 1])
    curve = daily_curve_range(res["ledger"], world["sig"], world["days"], ms_lo, ms_hi, env.e0)
    m = metrics(curve, env.e0)
    m["n_trades"] = len(res["ledger"])
    return m


def baseline_policies(sig: pd.DataFrame) -> dict:
    strat = sig["strategy"].to_numpy()

    def always(obs, i):
        return A_FULL

    def solo(name):
        def p(obs, i):
            return A_FULL if strat[i] == name else A_SKIP
        return p

    def random_gate(p_take: float, seed: int):
        rng = np.random.default_rng(seed)

        def p(obs, i):
            return A_FULL if rng.random() < p_take else A_SKIP
        return p

    return dict(always=always, solo_s2=solo("s2"), solo_dmi=solo("dmi"),
                solo_cpmt=solo("cpmt"), random_gate=random_gate)


def gate_table(world: dict, policies: dict[str, object]) -> pd.DataFrame:
    rows = []
    for pname, pol in policies.items():
        for period in ("train", "val", "oos"):
            m = eval_policy(world, period, pol)
            rows.append(dict(policy=pname, period=period, net=m["net"], sharpe=m["sharpe"],
                             maxdd=m["maxdd"], calmar=m["calmar"], trades=m["n_trades"],
                             win=m["win_rate"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    world = load_all()
    pols = baseline_policies(world["sig"])
    t = gate_table(world, {k: pols[k] for k in ("always", "solo_s2", "solo_dmi", "solo_cpmt")})
    pd.set_option("display.width", 160)
    for period in ("train", "val", "oos"):
        sub = t[t["period"] == period].copy()
        sub["net"] = sub["net"].round(0)
        print(f"\n== {period} ==")
        print(sub[["policy", "net", "sharpe", "maxdd", "calmar", "trades", "win"]]
              .to_string(index=False))
