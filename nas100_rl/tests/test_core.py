"""Unit tests: indicator exactness, engine fill semantics, no-lookahead guarantees."""
import numpy as np
import pandas as pd
import pytest

from nas100_rl import indicators as ind
from nas100_rl.engine import walk_exit, path_order_high_first


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def test_rma_seed_and_recursion():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r = ind.rma(x, 3)
    assert np.isnan(r[0]) and np.isnan(r[1])
    assert r[2] == pytest.approx(2.0)                      # SMA seed
    assert r[3] == pytest.approx((4.0 + 2 * 2.0) / 3)      # Wilder recursion
    assert r[4] == pytest.approx((5.0 + 2 * r[3]) / 3)


def test_rma_with_leading_nan():
    x = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
    r = ind.rma(x, 3)
    assert np.isnan(r[3])
    assert r[4] == pytest.approx(2.0)


def test_atr_first_bar_tr():
    h = np.array([10.0, 11.0, 12.0])
    l = np.array([9.0, 10.0, 11.0])
    c = np.array([9.5, 10.5, 11.5])
    tr = ind.true_range(h, l, c)
    assert tr[0] == pytest.approx(1.0)                     # h-l on first bar
    assert tr[1] == pytest.approx(max(1.0, abs(11 - 9.5), abs(10 - 9.5)))


def test_ema_seed():
    x = np.arange(1.0, 11.0)
    e = ind.ema(x, 5)
    assert np.isnan(e[3])
    assert e[4] == pytest.approx(3.0)
    assert e[5] == pytest.approx(3.0 + (2 / 6) * (6.0 - 3.0))


def test_dmi_warmup_and_range():
    rng = np.random.default_rng(0)
    n = 500
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    h = c + rng.uniform(0.1, 1, n)
    l = c - rng.uniform(0.1, 1, n)
    plus, minus, adx = ind.dmi(h, l, c, 14, 14)
    assert np.all(np.isnan(adx[:27]))
    ok = ~np.isnan(adx)
    assert ok.sum() > 400
    assert np.nanmin(plus) >= 0 and np.nanmin(minus) >= 0
    assert np.nanmax(adx[ok]) <= 100


def test_pivot_high_strict_and_alignment():
    h = np.array([1.0, 2.0, 5.0, 2.0, 1.0, 1.0, 5.0, 5.0, 1.0, 0.0])
    p = ind.pivot_high(h, 2, 2)
    assert p[4] == pytest.approx(5.0)          # value at i-2, confirmed at i=4
    assert np.isnan(p[8])                      # equal highs (5,5) -> no strict pivot
    # left side equality also voids: build one
    h2 = np.array([5.0, 1.0, 5.0, 1.0, 0.0])
    p2 = ind.pivot_high(h2, 2, 2)
    assert np.isnan(p2[4])


# ---------------------------------------------------------------------------
# engine fill semantics
# ---------------------------------------------------------------------------
def test_path_order_rule():
    # (h - o) <= (o - low) -> price visits the high first
    assert path_order_high_first(10, 11, 8) is True    # 1 <= 2
    assert path_order_high_first(10, 13, 9) is False   # 3 > 1


def test_walk_exit_stop_gap_through():
    # long stopped on gap: open below stop
    px, why = walk_exit(o=95.0, h=96.0, low=94.0, c=95.5, direction=1, stop=98.0, limit=None)
    assert why == "stop" and px == 95.0
    # normal stop touch fills at stop
    px, why = walk_exit(o=100.0, h=101.0, low=97.0, c=98.0, direction=1, stop=98.5, limit=None)
    assert why == "stop" and px == 98.5


def test_walk_exit_stop_limit_same_bar_path():
    # long, stop 98 limit 102, bar o=100 h=103 l=97: o->? h-o=3, o-l=3 -> high first (<=)
    px, why = walk_exit(100.0, 103.0, 97.0, 99.0, 1, 98.0, 102.0)
    assert why == "limit" and px == 102.0
    # same bar but open closer to low: o=98.5 -> h-o=4.5 > o-l=1.5 -> low first -> stop
    px, why = walk_exit(98.5, 103.0, 97.0, 99.0, 1, 98.0, 102.0)
    assert why == "stop" and px == 98.0


def test_walk_exit_short_sides():
    # short: stop above, limit below
    px, why = walk_exit(100.0, 102.0, 95.0, 96.0, -1, 101.0, 96.0)
    # h-o=2, o-l=5 -> high first -> stop
    assert why == "stop" and px == 101.0


# ---------------------------------------------------------------------------
# no-lookahead: S2 signals on truncated data equal signals on full data
# ---------------------------------------------------------------------------
def test_s2_no_lookahead():
    from nas100_rl import data, config
    from nas100_rl.strategies import s2
    df1m = data.load_1min()
    # small slice: 30 server-days
    days = df1m["srv"].dt.normalize().unique()
    sl = df1m[df1m["srv"] < days[30]].reset_index(drop=True)
    bars5 = data.resample(sl, 5).iloc[:]  # resample of slice
    full = s2.run(bars5, sl, collect_signals=True)["signals"]
    # truncate at 70% of bars: signals before the cut must be identical
    cut = int(len(bars5) * 0.7)
    bars5_t = bars5.iloc[:cut].reset_index(drop=True)
    cut_srv = bars5.iloc[cut]["srv"]
    sl_t = sl[sl["srv"] < cut_srv].reset_index(drop=True)
    trunc = s2.run(bars5_t, sl_t, collect_signals=True)["signals"]
    full_pre = [s for s in full if s["signal_i"] < cut - 1]  # last bar excluded (close-time parity)
    trunc_pre = [s for s in trunc if s["signal_i"] < cut - 1]
    assert len(full_pre) == len(trunc_pre)
    for a, b in zip(full_pre, trunc_pre):
        assert a["signal_ms"] == b["signal_ms"]
        assert a["direction"] == b["direction"]
        assert a["sl"] == pytest.approx(b["sl"])


def test_cpmt_no_lookahead():
    from nas100_rl import data
    from nas100_rl.strategies.cpmt import strategy as cpmt
    df1m = data.load_1min()
    days = df1m["srv"].dt.normalize().unique()
    sl = df1m[df1m["srv"] < days[120]].reset_index(drop=True)
    frames = {tf: data.resample(df1m, int(tf)) for tf in ("240", "180", "120", "360", "60", "30")}
    fr = {tf: f[f["srv"] < days[120]].reset_index(drop=True) for tf, f in frames.items()}
    d = data.daily(df1m)
    d_sl = d[d["srv_day"] < days[120]].reset_index(drop=True)
    full = cpmt.run(fr["30"], sl, fr, d_sl, collect_signals=True)["signals"]

    cut_day = days[90]
    sl_t = sl[sl["srv"] < cut_day].reset_index(drop=True)
    fr_t = {tf: f[f["srv"] < cut_day].reset_index(drop=True) for tf, f in fr.items()}
    d_t = d_sl[d_sl["srv_day"] < cut_day].reset_index(drop=True)
    trunc = cpmt.run(fr_t["30"], sl_t, fr_t, d_t, collect_signals=True)["signals"]

    cut_ms = int(pd.Timestamp(cut_day).tz_localize("UTC").value // 1e6)
    # generous margin: drop signals near the cut whose HTF bar would be truncated
    margin = 86_400_000
    full_pre = [s for s in full if s["signal_ms"] < cut_ms - margin]
    trunc_pre = [s for s in trunc if s["signal_ms"] < cut_ms - margin]
    assert len(full_pre) == len(trunc_pre) and len(full_pre) > 0
    for a, b in zip(full_pre, trunc_pre):
        assert a["signal_ms"] == b["signal_ms"] and a["direction"] == b["direction"]
        assert a["stop"] == pytest.approx(b["stop"])
