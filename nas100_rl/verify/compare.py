"""Trade-by-trade comparison between a Python port and the TradingView export."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebase_equity(tv: pd.DataFrame, initial_capital: float, window_start) -> float:
    """Equity at `window_start` implied by the TV export's cumulative PnL column."""
    prior = tv[tv["exit_time"] < window_start]
    if len(prior) == 0:
        return initial_capital
    return initial_capital + float(prior["cum_pnl"].iloc[-1])


def align(tv: pd.DataFrame, port: pd.DataFrame, tol_ms: int) -> pd.DataFrame:
    """Greedy one-to-one alignment on (direction, entry time within tol)."""
    tv = tv.reset_index(drop=True)
    port = port.reset_index(drop=True)
    tv_ms = tv["entry_time"].astype("int64").to_numpy() // 1_000_000
    pt_ms = port["entry_ms"].to_numpy()
    used = np.zeros(len(port), dtype=bool)
    match_idx = np.full(len(tv), -1)
    for i in range(len(tv)):
        cand = np.where(
            (~used)
            & (port["direction"].to_numpy() == tv["direction"].iloc[i])
            & (np.abs(pt_ms - tv_ms[i]) <= tol_ms)
        )[0]
        if len(cand):
            j = cand[np.argmin(np.abs(pt_ms[cand] - tv_ms[i]))]
            match_idx[i] = j
            used[j] = True
    out = tv.copy()
    out["port_idx"] = match_idx
    return out


def report(name: str, tv: pd.DataFrame, port: pd.DataFrame, tol_ms: int,
           window_start, window_end) -> dict:
    tv_w = tv[(tv["entry_time"] >= window_start) & (tv["entry_time"] <= window_end)
              & ~tv["open_at_export"]].reset_index(drop=True)
    pt_w = port[(port["entry_time"] >= window_start) & (port["entry_time"] <= window_end)
                & (port["exit_reason"] != "open_at_end")].reset_index(drop=True)
    al = align(tv_w, pt_w, tol_ms)
    m = al["port_idx"] >= 0
    matched = al[m]
    pj = pt_w.iloc[matched["port_idx"].to_numpy()].reset_index(drop=True)
    mt = matched.reset_index(drop=True)

    entry_diff = (pj["entry_price"].to_numpy() - mt["entry_price"].to_numpy())
    exit_dt = (pj["exit_ms"].to_numpy() - mt["exit_time"].astype("int64").to_numpy() // 1_000_000)
    exit_diff = (pj["exit_price"].to_numpy() - mt["exit_price"].to_numpy())
    qty_ratio = pj["qty"].to_numpy() / mt["qty"].to_numpy()
    pnl_diff = pj["pnl"].to_numpy() - mt["pnl"].to_numpy()

    res = dict(
        name=name,
        window=(str(window_start), str(window_end)),
        tv_trades=len(tv_w),
        port_trades=len(pt_w),
        matched=len(matched),
        match_rate_tv=len(matched) / max(1, len(tv_w)),
        match_rate_port=len(matched) / max(1, len(pt_w)),
        pnl_tv_window_sum=float(tv_w["pnl"].sum()),
        entry_px_mean_abs=float(np.mean(np.abs(entry_diff))) if len(mt) else np.nan,
        entry_px_p95_abs=float(np.percentile(np.abs(entry_diff), 95)) if len(mt) else np.nan,
        exit_time_same=float(np.mean(exit_dt == 0)) if len(mt) else np.nan,
        exit_px_mean_abs=float(np.mean(np.abs(exit_diff))) if len(mt) else np.nan,
        qty_ratio_med=float(np.median(qty_ratio)) if len(mt) else np.nan,
        pnl_tv_sum=float(mt["pnl"].sum()),
        pnl_port_sum=float(pj["pnl"].sum()),
        pnl_port_matched_only=float(pj["pnl"].sum()),
        pnl_port_window_sum=float(pt_w["pnl"].sum()),
        pnl_diff_mean=float(np.mean(pnl_diff)) if len(mt) else np.nan,
    )
    res["unmatched_tv"] = al[~m][["entry_time", "direction", "entry_price", "entry_signal",
                                  "exit_time", "exit_signal", "pnl"]]
    used = set(matched["port_idx"].to_numpy().tolist())
    res["unmatched_port"] = pt_w[~pt_w.index.isin(used)]
    res["matched_pairs"] = (mt, pj)
    return res


def print_report(res: dict, show_unmatched: int = 8) -> None:
    print(f"=== {res['name']} | window {res['window'][0]} .. {res['window'][1]}")
    print(f"  TV trades: {res['tv_trades']}   port trades: {res['port_trades']}   "
          f"matched: {res['matched']}  ({res['match_rate_tv']:.1%} of TV, "
          f"{res['match_rate_port']:.1%} of port)")
    print(f"  entry px |diff|: mean {res['entry_px_mean_abs']:.3f}  p95 {res['entry_px_p95_abs']:.3f}")
    print(f"  exit: same-bar-time {res['exit_time_same']:.1%}  px |diff| mean {res['exit_px_mean_abs']:.3f}")
    print(f"  qty ratio median: {res['qty_ratio_med']:.4f}")
    print(f"  PnL matched:  TV {res['pnl_tv_sum']:,.0f}   port {res['pnl_port_matched_only']:,.0f}")
    print(f"  PnL window:   TV {res['pnl_tv_window_sum']:,.0f}   port {res['pnl_port_window_sum']:,.0f}   "
          f"(diff {res['pnl_port_window_sum'] - res['pnl_tv_window_sum']:+,.0f} = "
          f"{(res['pnl_port_window_sum'] - res['pnl_tv_window_sum']) / max(1e-9, abs(res['pnl_tv_window_sum'])):+.1%})")
    if len(res["unmatched_tv"]):
        print(f"  -- unmatched TV trades ({len(res['unmatched_tv'])}), first {show_unmatched}:")
        print(res["unmatched_tv"].head(show_unmatched).to_string(index=False))
    if len(res["unmatched_port"]):
        cols = ["entry_time", "direction", "entry_price", "exit_time", "exit_reason", "pnl"]
        print(f"  -- unmatched port trades ({len(res['unmatched_port'])}), first {show_unmatched}:")
        print(res["unmatched_port"][cols].head(show_unmatched).to_string(index=False))
