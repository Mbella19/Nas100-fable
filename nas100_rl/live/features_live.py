"""Live feature builder: the same 46 features as rl/features.py, computed
incrementally for one signal at a time.

Reuses the frozen formula helpers (`_bar30_features`, `_daily_features`) and the
frozen TRAIN normalization stats — nothing is refit. Windows are sized so every
rolling/recursive indicator is numerically converged at the right edge; parity
with the cached feature matrix is asserted by `runner --selftest`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, data, indicators
from ..rl import features as featmod


class FeatureBuilder:
    def __init__(self, basis: str = "live"):
        suffix = "" if basis == "tv" else f"_{basis}"
        st = pd.read_parquet(config.CACHE_DIR / f"feature_stats{suffix}.parquet")
        self.names: list[str] = list(st.columns)
        self.mu = st.loc["mu"].to_numpy(dtype=np.float64)
        self.sd = st.loc["sd"].to_numpy(dtype=np.float64)
        self.anchor = pd.Timestamp(config.LIVE["cpmt_anchor"])
        self.win30_days = int(config.LIVE["feature_window_days"])
        self._cache_key = None
        self._b30 = None
        self._dfe = None
        self._d_atr = None
        self._day_open = None
        # rolling-performance source: closed mechanical trades, sorted on demand
        self._hist_dirty = True
        self._by_strat: dict = {}
        self._all = None

    # ------------------------------------------------------------- history
    def set_history(self, sig_hist: pd.DataFrame):
        """sig_hist: unified rows (frozen parquet + live closures appended)."""
        self.sig_hist = sig_hist
        self._hist_dirty = True

    def append_history(self, rows: list[dict]):
        if rows:
            self.sig_hist = pd.concat([self.sig_hist, pd.DataFrame(rows)],
                                      ignore_index=True)
            self._hist_dirty = True

    def _perf_arrays(self):
        if not self._hist_dirty:
            return
        h = self.sig_hist
        h = h[~h["exit_ms"].isna()]
        for s in ("s2", "dmi", "cpmt"):
            t = h[h["strategy"] == s].sort_values("exit_ms")
            self._by_strat[s] = (t["exit_ms"].to_numpy(dtype=np.int64),
                                 t["r_mult"].to_numpy(dtype=np.float64),
                                 t["pc_pnl"].to_numpy(dtype=np.float64))
        a = h.sort_values("exit_ms")
        self._all = (a["exit_ms"].to_numpy(dtype=np.int64),
                     a["r_mult"].to_numpy(dtype=np.float64))
        self._hist_dirty = False

    # ------------------------------------------------------------- frames
    def _frames(self, m1: pd.DataFrame):
        """30m/daily feature frames over converged windows; cached until a new
        bar arrives (values at CLOSED bars never change retroactively)."""
        key = (len(m1), int(m1["utc_ms"].iloc[-1]))
        if key == self._cache_key:
            return
        last_day = m1["srv"].iloc[-1].normalize()
        w30 = m1[m1["srv"] >= last_day - pd.Timedelta(days=self.win30_days)].reset_index(drop=True)
        wd = m1[m1["srv"] >= self.anchor].reset_index(drop=True)
        bars30 = data.resample(w30, 30)
        daily = data.daily(wd)
        self._b30 = featmod._bar30_features(bars30)
        self._dfe = featmod._daily_features(daily, bars30)
        self._d_atr = indicators.atr(daily["high"].to_numpy(), daily["low"].to_numpy(),
                                     daily["close"].to_numpy(), 14)
        self._day_open = bars30.groupby(bars30["srv"].dt.normalize())["open"].first()
        self._cache_key = key

    # ------------------------------------------------------------- one row
    def row(self, e: dict, m1: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Feature vector (z-scored, clipped, ordered like training) and the
        mark price for a single emission dict."""
        self._frames(m1)
        self._perf_arrays()
        sms = int(e["signal_ms"])
        strat = e["strategy"]

        cms = self._b30["close_ms"].to_numpy()
        i30 = int(np.searchsorted(cms, sms, side="right") - 1)
        assert i30 >= 0, "no closed 30m bar before signal"
        b = self._b30.iloc[i30]

        b_day = b["srv_day"]
        dkey = self._dfe["srv_day"].to_numpy()
        di = int(np.searchsorted(dkey, np.datetime64(b_day), side="left") - 1)
        di = int(np.clip(di, 0, len(self._dfe) - 1))
        d = self._dfe.iloc[di]

        g_open = self._day_open.get(b_day, np.nan)
        gap = (g_open - d["d_close"]) / self._d_atr[di]

        ex, rm, pc = self._by_strat[strat]
        k = int(np.searchsorted(ex, sms, side="right"))
        roll10_r = roll30_r = 0.0
        roll10_wr = 0.5
        days_since_win = 10.0
        if k > 0:
            w10 = rm[max(0, k - 10):k]
            roll10_r = float(w10.mean())
            roll10_wr = float((w10 > 0).mean())
            roll30_r = float(rm[max(0, k - 30):k].mean())
            wins = np.where(pc[:k] > 0)[0]
            if len(wins):
                days_since_win = min(20.0, (sms - ex[wins[-1]]) / 86_400_000)
        a_ex, a_rm = self._all
        ka = int(np.searchsorted(a_ex, sms, side="right"))
        all_roll10 = float(a_rm[max(0, ka - 10):ka].mean()) if ka > 0 else 0.0

        ny_t = pd.Timestamp(sms, unit="ms", tz="UTC").tz_convert(config.NY_TZ)
        hour_frac = ny_t.hour + ny_t.minute / 60.0
        dow = ny_t.dayofweek
        pat = e.get("pattern", "") or ""

        F = {
            "f_is_s2": float(strat == "s2"),
            "f_is_dmi": float(strat == "dmi"),
            "f_is_cpmt": float(strat == "cpmt"),
            "f_direction": float(e["direction"]),
            "f_stop_atr": float(np.clip(e["stop_dist"] / e["atr_sig"], 0, 12)),
            "f_log_stop": float(np.log(max(e["stop_dist"], 1e-9))),
            "f_nth2": float(e.get("nth_of_day", 1) == 2),
            "f_adx_sig": float(np.nan_to_num(e.get("adx_sig", np.nan) / 100.0)),
            "f_di_spread": float(np.clip(np.nan_to_num(e.get("di_spread", np.nan) / 50.0), -2, 2)),
            "f_pat_rev": float(pat in featmod.REVERSAL_PATTERNS),
            "f_pat_cont": float(pat in featmod.CONTINUATION_PATTERNS),
            "f_pat_bilat": float(pat != "" and pat not in
                                 (featmod.REVERSAL_PATTERNS | featmod.CONTINUATION_PATTERNS)),
            "f_width": float(np.nan_to_num(e.get("width", np.nan))) / 100.0,
            "f_stream": float(max(e.get("stream", -1), 0)) / 5.0,
            "f_hour_sin": float(np.sin(2 * np.pi * hour_frac / 24)),
            "f_hour_cos": float(np.cos(2 * np.pi * hour_frac / 24)),
            "f_dow_mon": float(dow == 0),
            "f_dow_tue": float(dow == 1),
            "f_dow_wed": float(dow == 2),
            "f_dow_thu": float(dow == 3),
            "f_dow_fri": float(dow == 4),
            "f_rv_8h": float(b["rv_8h"]),
            "f_rv_1d": float(b["rv_1d"]),
            "f_rv_5d": float(b["rv_5d"]),
            "f_vol_ratio": float(b["vol_ratio"]),
            "f_atr_regime": float(b["atr_regime_z"]),
            "f_ema20_dist": float(b["ema20_dist"]),
            "f_ema100_dist": float(b["ema100_dist"]),
            "f_autocorr1": float(b["autocorr1"]),
            "f_downside_share": float(b["downside_share"]),
            "f_range_pos": float(b["range_pos_day"]),
            "f_day_ret_atr": float(b["day_ret_atr"]),
            "f_bars_into_day": float(b["bars_into_day"]),
            "f_d_adx": float(d["d_adx"]),
            "f_d_ema10_dist": float(d["d_ema10_dist"]),
            "f_d_ret5": float(np.clip(d["d_ret5"], -25, 25)),
            "f_d_ret20": float(np.clip(d["d_ret20"], -40, 40)),
            "f_d_atr_z": float(d["d_atr_z"]),
            "f_gap_atr": float(np.clip(np.nan_to_num(gap), -5, 5)),
            "f_trend_align": float(e["direction"] * d["d_ema10_dist"]),
            "f_roll10_r": float(np.clip(roll10_r, -3, 3)),
            "f_roll10_wr": roll10_wr,
            "f_roll30_r": float(np.clip(roll30_r, -3, 3)),
            "f_days_since_win": days_since_win / 20.0,
            "f_all_roll10_r": float(np.clip(all_roll10, -3, 3)),
        }
        raw = np.array([np.nan_to_num(F[k2], nan=0.0) for k2 in self.names],
                       dtype=np.float64)
        z = np.clip((raw - self.mu) / self.sd, -6, 6).astype(np.float32)
        return z, float(b["mark_price"])
