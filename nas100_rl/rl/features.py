"""Causal per-signal features for the RL agent.

Every feature is computed strictly from information available at the signal's
decision time (the signal bar's close):
- market features come from the LAST CLOSED 30m bar at/before signal_ms and the
  LAST CLOSED server-day before the signal's day;
- per-strategy rolling performance uses mechanical trades with exit_ms <= signal_ms;
- normalization stats (z-score) are fit on TRAIN-period rows only.

Strategy-level truncation tests live in tests/test_core.py (S2/CPMT); feature-level
causality is additionally cross-checked by the live runner's parity selftest, which
recomputes features incrementally and compares to this module's cached matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, data, indicators
from . import signals as sigmod

REVERSAL_PATTERNS = {"Double Top", "Double Bottom", "Triple Top", "Triple Bottom",
                     "Head and Shoulders", "Inv. Head and Shoulders"}
CONTINUATION_PATTERNS = {"Bullish Flag", "Bearish Flag", "Bullish Pennant", "Bearish Pennant"}

FEATURES: list[str] = []  # populated by build(); persisted alongside the matrix


def _roll_z(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win, min_periods=win // 4).mean()
    s = x.rolling(win, min_periods=win // 4).std()
    return (x - m) / s.replace(0.0, np.nan)


def _bar30_features(bars30: pd.DataFrame) -> pd.DataFrame:
    c = bars30["close"]
    r = np.log(c / c.shift(1))
    atr14 = pd.Series(indicators.atr(bars30["high"].to_numpy(), bars30["low"].to_numpy(),
                                     c.to_numpy(), 14), index=bars30.index)
    ema20 = pd.Series(indicators.ema(c.to_numpy(), 20), index=bars30.index)
    ema100 = pd.Series(indicators.ema(c.to_numpy(), 100), index=bars30.index)
    f = pd.DataFrame(index=bars30.index)
    f["close_ms"] = bars30["close_ms"]
    f["mark_price"] = c
    f["rv_8h"] = r.rolling(16, min_periods=8).std() * 100
    f["rv_1d"] = r.rolling(48, min_periods=24).std() * 100
    f["rv_5d"] = r.rolling(240, min_periods=120).std() * 100
    f["vol_ratio"] = (f["rv_8h"] / f["rv_5d"]).clip(0, 5)
    f["atr_regime_z"] = _roll_z(np.log(atr14), 240 * 52 // 10).clip(-4, 4)  # ~1y of 30m bars
    f["ema20_dist"] = ((c - ema20) / atr14).clip(-10, 10)
    f["ema100_dist"] = ((c - ema100) / atr14).clip(-15, 15)
    f["autocorr1"] = r.rolling(240, min_periods=120).corr(r.shift(1)).clip(-1, 1)
    dn = (r.where(r < 0, 0.0) ** 2).rolling(240, min_periods=120).sum()
    tot = (r ** 2).rolling(240, min_periods=120).sum()
    f["downside_share"] = (dn / tot.replace(0.0, np.nan)).clip(0, 1)
    day = bars30["srv"].dt.normalize()
    g = bars30.groupby(day)
    day_open = g["open"].transform("first")
    day_hi = g["high"].cummax()
    day_lo = g["low"].cummin()
    rng = (day_hi - day_lo)
    f["range_pos_day"] = ((c - day_lo) / rng.replace(0.0, np.nan)).clip(0, 1)
    f["day_ret_atr"] = ((c - day_open) / atr14).clip(-15, 15)
    f["bars_into_day"] = g.cumcount() / 46.0
    f["srv_day"] = day
    return f


def _daily_features(daily: pd.DataFrame, bars30: pd.DataFrame) -> pd.DataFrame:
    dc = daily["close"].to_numpy()
    dh = daily["high"].to_numpy()
    dl = daily["low"].to_numpy()
    _, _, adx = indicators.dmi(dh, dl, dc, 14, 14)
    ema10 = indicators.ema(dc, config.CPMT["gate_len"])
    atr14 = indicators.atr(dh, dl, dc, 14)
    f = pd.DataFrame(index=daily.index)
    f["srv_day"] = daily["srv_day"]
    f["d_adx"] = adx / 100.0
    f["d_ema10_dist"] = np.clip((dc - ema10) / atr14, -8, 8)
    f["d_ret5"] = pd.Series(dc).pct_change(5).to_numpy() * 100
    f["d_ret20"] = pd.Series(dc).pct_change(20).to_numpy() * 100
    f["d_atr_z"] = _roll_z(pd.Series(np.log(atr14)), 252).clip(-4, 4).to_numpy()
    f["d_close"] = dc
    return f


def build(refresh: bool = False, basis: str = "tv") -> tuple[pd.DataFrame, list[str]]:
    suffix = "" if basis == "tv" else f"_{basis}"
    cache = config.CACHE_DIR / f"features{suffix}.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        names = [c for c in df.columns if c.startswith("f_")]
        return df, names

    sig = sigmod.build(basis=basis)
    df1m = data.load_1min()
    bars30 = data.resample(df1m, 30)
    daily = data.daily(df1m)

    b30 = _bar30_features(bars30)
    dfe = _daily_features(daily, bars30)

    # map each signal to last closed 30m bar and last closed daily bar
    cms = b30["close_ms"].to_numpy()
    i30 = np.searchsorted(cms, sig["signal_ms"].to_numpy(), side="right") - 1
    assert (i30 >= 0).all()
    b = b30.iloc[i30].reset_index(drop=True)

    b_day = b["srv_day"].to_numpy()
    dkey = dfe["srv_day"].to_numpy()
    di = np.searchsorted(dkey, b_day, side="left") - 1     # last CLOSED day strictly before
    di = np.clip(di, 0, len(dfe) - 1)
    d = dfe.iloc[di].reset_index(drop=True)

    # today's gap: first 30m open of the signal's day vs prev daily close (causal: day open
    # is known before any in-day signal)
    day_open = bars30.groupby(bars30["srv"].dt.normalize())["open"].first()
    g_open = day_open.reindex(pd.DatetimeIndex(b_day)).to_numpy()
    d_atr = indicators.atr(daily["high"].to_numpy(), daily["low"].to_numpy(),
                           daily["close"].to_numpy(), 14)
    gap = (g_open - d["d_close"].to_numpy()) / d_atr[di]

    # per-strategy rolling mechanical performance (exit_ms <= signal_ms, shifted)
    n = len(sig)
    roll10_r = np.zeros(n)
    roll10_wr = np.full(n, 0.5)
    roll30_r = np.zeros(n)
    days_since_win = np.full(n, 10.0)
    all_roll10 = np.zeros(n)
    by_strat = {}
    for s in ("s2", "dmi", "cpmt"):
        t = sig[sig["strategy"] == s].sort_values("exit_ms")
        by_strat[s] = (t["exit_ms"].to_numpy(), t["r_mult"].to_numpy(), t["pc_pnl"].to_numpy())
    all_sorted = sig.sort_values("exit_ms")
    all_exits = all_sorted["exit_ms"].to_numpy()
    all_r = all_sorted["r_mult"].to_numpy()
    sms = sig["signal_ms"].to_numpy()
    strats = sig["strategy"].to_numpy()
    for i in range(n):
        ex, rm, pc = by_strat[strats[i]]
        k = np.searchsorted(ex, sms[i], side="right")
        if k > 0:
            w10 = rm[max(0, k - 10):k]
            roll10_r[i] = w10.mean()
            roll10_wr[i] = (w10 > 0).mean()
            roll30_r[i] = rm[max(0, k - 30):k].mean()
            wins = np.where(pc[:k] > 0)[0]
            if len(wins):
                days_since_win[i] = min(20.0, (sms[i] - ex[wins[-1]]) / 86_400_000)
        ka = np.searchsorted(all_exits, sms[i], side="right")
        if ka > 0:
            all_roll10[i] = all_r[max(0, ka - 10):ka].mean()

    ny_t = pd.to_datetime(sig["signal_ms"], unit="ms", utc=True).dt.tz_convert(config.NY_TZ)
    hour_frac = (ny_t.dt.hour + ny_t.dt.minute / 60.0).to_numpy()
    dow = ny_t.dt.dayofweek.to_numpy()

    pat = sig["pattern"].fillna("").to_numpy()
    out = pd.DataFrame({"trade_id": sig["trade_id"], "period": sig["period"],
                        "strategy": sig["strategy"], "signal_ms": sig["signal_ms"],
                        "mark_price": b["mark_price"].to_numpy()})
    F = {
        "f_is_s2": (strats == "s2").astype(float),
        "f_is_dmi": (strats == "dmi").astype(float),
        "f_is_cpmt": (strats == "cpmt").astype(float),
        "f_direction": sig["direction"].to_numpy().astype(float),
        "f_stop_atr": np.clip(sig["stop_dist"].to_numpy() / sig["atr_sig"].to_numpy(), 0, 12),
        "f_log_stop": np.log(np.clip(sig["stop_dist"].to_numpy(), 1e-9, None)),
        "f_nth2": (sig["nth_of_day"].to_numpy() == 2).astype(float),
        "f_adx_sig": np.nan_to_num(sig["adx_sig"].to_numpy() / 100.0),
        "f_di_spread": np.clip(np.nan_to_num(sig["di_spread"].to_numpy() / 50.0), -2, 2),
        "f_pat_rev": np.isin(pat, list(REVERSAL_PATTERNS)).astype(float),
        "f_pat_cont": np.isin(pat, list(CONTINUATION_PATTERNS)).astype(float),
        "f_pat_bilat": ((pat != "") & ~np.isin(pat, list(REVERSAL_PATTERNS | CONTINUATION_PATTERNS))).astype(float),
        "f_width": np.nan_to_num(sig["width"].to_numpy()) / 100.0,
        "f_stream": np.where(sig["stream"].to_numpy() >= 0, sig["stream"].to_numpy(), 0) / 5.0,
        "f_hour_sin": np.sin(2 * np.pi * hour_frac / 24),
        "f_hour_cos": np.cos(2 * np.pi * hour_frac / 24),
        "f_dow_mon": (dow == 0).astype(float),
        "f_dow_tue": (dow == 1).astype(float),
        "f_dow_wed": (dow == 2).astype(float),
        "f_dow_thu": (dow == 3).astype(float),
        "f_dow_fri": (dow == 4).astype(float),
        "f_rv_8h": b["rv_8h"].to_numpy(),
        "f_rv_1d": b["rv_1d"].to_numpy(),
        "f_rv_5d": b["rv_5d"].to_numpy(),
        "f_vol_ratio": b["vol_ratio"].to_numpy(),
        "f_atr_regime": b["atr_regime_z"].to_numpy(),
        "f_ema20_dist": b["ema20_dist"].to_numpy(),
        "f_ema100_dist": b["ema100_dist"].to_numpy(),
        "f_autocorr1": b["autocorr1"].to_numpy(),
        "f_downside_share": b["downside_share"].to_numpy(),
        "f_range_pos": b["range_pos_day"].to_numpy(),
        "f_day_ret_atr": b["day_ret_atr"].to_numpy(),
        "f_bars_into_day": b["bars_into_day"].to_numpy(),
        "f_d_adx": d["d_adx"].to_numpy(),
        "f_d_ema10_dist": d["d_ema10_dist"].to_numpy(),
        "f_d_ret5": np.clip(d["d_ret5"].to_numpy(), -25, 25),
        "f_d_ret20": np.clip(d["d_ret20"].to_numpy(), -40, 40),
        "f_d_atr_z": d["d_atr_z"].to_numpy(),
        "f_gap_atr": np.clip(np.nan_to_num(gap), -5, 5),
        "f_trend_align": sig["direction"].to_numpy() * d["d_ema10_dist"].to_numpy(),
        "f_roll10_r": np.clip(roll10_r, -3, 3),
        "f_roll10_wr": roll10_wr,
        "f_roll30_r": np.clip(roll30_r, -3, 3),
        "f_days_since_win": days_since_win / 20.0,
        "f_all_roll10_r": np.clip(all_roll10, -3, 3),
    }
    for k, v in F.items():
        out[k] = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0)

    names = list(F.keys())
    # z-score on TRAIN rows only
    tr_mask = (out["period"] == "train").to_numpy()
    stats = {}
    for k in names:
        mu = float(out.loc[tr_mask, k].mean())
        sd = float(out.loc[tr_mask, k].std())
        sd = sd if sd > 1e-9 else 1.0
        out[k] = np.clip((out[k] - mu) / sd, -6, 6)
        stats[k] = (mu, sd)
    pd.DataFrame(stats, index=["mu", "sd"]).to_parquet(config.CACHE_DIR / f"feature_stats{suffix}.parquet")
    out.to_parquet(cache, index=False)
    return out, names


if __name__ == "__main__":
    df, names = build(refresh=True)
    print(f"{len(df)} rows, {len(names)} features")
    print(df.groupby("period").size().to_dict())
    desc = df[names].describe().T[["mean", "std", "min", "max"]]
    print(desc.round(2).to_string())
