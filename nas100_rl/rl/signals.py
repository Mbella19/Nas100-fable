"""Unified signal/trade dataset for the RL layer.

Runs the three VERIFIED strategy ports over the full timeline in their mechanical
(ungated) form and emits one row per actual trade entry, with the trade's full
lifecycle (fill, exit, per-contract economics) plus signal-time context.

Design: the strategies remain exactly their verified selves (frozen logic and
state evolution). The RL agent overlays take/skip/scale on this fixed stream —
it never alters entries, exits, stops, or internal state. A skipped signal
therefore does not change later signals (copy-trading semantics).

Per-contract economics let the portfolio env size positions at any equity:
  pnl($) = per_contract_pnl * qty(action, equity, stop_dist)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, data


def _s2_dataset(df1m: pd.DataFrame) -> pd.DataFrame:
    from ..strategies import s2
    bars5 = data.resample(df1m, 5)
    res = s2.run(bars5, df1m, collect_signals=True)
    tr = res["trades"]
    sig = pd.DataFrame([s for s in res["signals"] if s["taken"]])
    tr = tr.merge(sig[["signal_i", "stop_dist", "atr", "nth"]].rename(
        columns={"nth": "sig_nth"}), left_on="signal_i", right_on="signal_i", how="left")
    out = pd.DataFrame({
        "strategy": "s2",
        "signal_ms": (tr["entry_ms"] - 300_000).astype("int64"),  # decision = prior bar close
        "direction": tr["direction"],
        "entry_ms": tr["entry_ms"],
        "entry_price": tr["entry_price"],
        "exit_ms": tr.get("exit_ms_exact", tr["exit_ms"]).fillna(tr["exit_ms"]).astype("int64")
        if "exit_ms_exact" in tr else tr["exit_ms"],
        "exit_price": tr["exit_price"],
        "exit_reason": tr["exit_reason"],
        "stop_dist": tr["stop_dist"],
        "atr_sig": tr["atr"],
        "commission_pc": 2 * config.S2["commission_per_contract_per_order"],
        "nth_of_day": tr["nth"],
        "adx_sig": np.nan, "di_spread": np.nan,
        "pattern": "", "stream": -1, "width": np.nan,
    })
    return out


def _dmi_dataset(df1m: pd.DataFrame) -> pd.DataFrame:
    from ..strategies import dmi
    res = dmi.run(df1m, collect_signals=True)
    tr = res["trades"]
    sig = pd.DataFrame(res["signals"])
    sig = sig[sig["kind"] != "rejected"]
    tr = tr.merge(sig[["signal_i", "stop_dist", "atr", "adx", "di_spread"]],
                  on="signal_i", how="left")
    out = pd.DataFrame({
        "strategy": "dmi",
        "signal_ms": (tr["entry_ms"] - 60_000).astype("int64"),
        "direction": tr["direction"],
        "entry_ms": tr["entry_ms"],
        "entry_price": tr["entry_price"],
        "exit_ms": tr["exit_ms"],
        "exit_price": tr["exit_price"],
        "exit_reason": tr["exit_reason"],
        "stop_dist": tr["stop_dist"],
        "atr_sig": tr["atr"],
        "commission_pc": 0.0,
        "nth_of_day": 1,
        "adx_sig": tr["adx"], "di_spread": tr["di_spread"],
        "pattern": "", "stream": -1, "width": np.nan,
    })
    return out


def _cpmt_dataset(df1m: pd.DataFrame) -> pd.DataFrame:
    from ..strategies.cpmt import strategy as cpmt
    frames = {tf: data.resample(df1m, int(tf)) for tf in ("240", "180", "120", "360", "60", "30")}
    d = data.daily(df1m)
    res = cpmt.run(frames["30"], df1m, frames, d, collect_signals=True)
    tr = res["trades"]
    sig = pd.DataFrame([s for s in res["signals"] if s["taken"]])
    tr = tr.merge(sig[["signal_i", "dist", "atr", "pattern", "stream", "width"]]
                  .rename(columns={"pattern": "sig_pattern", "stream": "sig_stream"}),
                  on="signal_i", how="left")
    exit_ms = tr["exit_ms"].copy()
    if "exit_ms_exact" in tr:
        exact = tr["exit_ms_exact"]
        exit_ms = exact.fillna(exit_ms)
    out = pd.DataFrame({
        "strategy": "cpmt",
        "signal_ms": (tr["entry_ms"] - 1_800_000).astype("int64"),
        "direction": tr["direction"],
        "entry_ms": tr["entry_ms"],
        "entry_price": tr["entry_price"],
        "exit_ms": exit_ms.astype("int64"),
        "exit_price": tr["exit_price"],
        "exit_reason": tr["exit_reason"],
        "stop_dist": tr["dist"],
        "atr_sig": tr["atr"],
        "commission_pc": 0.0,
        "nth_of_day": 1,
        "adx_sig": np.nan, "di_spread": np.nan,
        "pattern": tr["sig_pattern"], "stream": tr["sig_stream"], "width": tr["width"],
    })
    return out


def build(refresh: bool = False, basis: str = "tv") -> pd.DataFrame:
    """basis="tv": the TV-verified cost basis (zero spread; S2 commission only) used
    for verification parity. basis="live": subtracts the measured bid/ask spread per
    trade (longs pay at entry, shorts at exit; MT5 bid-feed accounting)."""
    if basis == "live":
        return _build_live(refresh)
    cache = config.CACHE_DIR / "signals.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    df1m = data.load_1min()
    parts = [_s2_dataset(df1m), _dmi_dataset(df1m), _cpmt_dataset(df1m)]
    sig = pd.concat(parts, ignore_index=True)
    sig = sig.sort_values(["signal_ms", "strategy"], kind="stable").reset_index(drop=True)

    # per-contract pnl and R-multiple of the mechanical trade
    sig["pc_pnl"] = (sig["exit_price"] - sig["entry_price"]) * sig["direction"] - sig["commission_pc"]
    sig["r_mult"] = sig["pc_pnl"] / sig["stop_dist"]
    sig["hold_ms"] = sig["exit_ms"] - sig["entry_ms"]

    # period labels by signal time
    bounds = {}
    for p in ("train", "val", "oos"):
        d = df1m[df1m["period"] == p]
        bounds[p] = (int(d["utc_ms"].iloc[0]), int(d["utc_ms"].iloc[-1]))
    sig["period"] = "train"
    sig.loc[sig["signal_ms"] >= bounds["val"][0], "period"] = "val"
    sig.loc[sig["signal_ms"] >= bounds["oos"][0], "period"] = "oos"

    sig["trade_id"] = np.arange(len(sig))
    sig.to_parquet(cache, index=False)
    return sig


def _build_live(refresh: bool = False) -> pd.DataFrame:
    cache = config.CACHE_DIR / "signals_live.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    sig = build(basis="tv").copy()
    m1 = data.load_1min()
    ms = m1["utc_ms"].to_numpy()
    spread = m1["spread"].to_numpy() * 0.1          # MT5 points -> index units

    def at(t_ms):
        i = np.clip(np.searchsorted(ms, t_ms, side="right") - 1, 0, len(ms) - 1)
        return spread[i]

    dirs = sig["direction"].to_numpy()
    cost = np.where(dirs > 0, at(sig["entry_ms"].to_numpy()), at(sig["exit_ms"].to_numpy()))
    sig["spread_cost_pc"] = cost
    sig["pc_pnl"] = sig["pc_pnl"] - cost
    sig["r_mult"] = sig["pc_pnl"] / sig["stop_dist"]
    sig.to_parquet(cache, index=False)
    return sig


if __name__ == "__main__":
    sig = build(refresh=True)
    print(f"{len(sig)} signals")
    print(sig.groupby(["period", "strategy"])["r_mult"].agg(["size", "mean"]).round(3))
    print("\nNaN check:", sig[["stop_dist", "pc_pnl", "r_mult"]].isna().sum().to_dict())
