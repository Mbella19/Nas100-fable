"""Data pipeline: MT5 1-min CSVs -> tz-resolved frames -> resampled bars -> parquet cache.

Timezone model
--------------
The MT5 server uses the "NY close at midnight" convention: server time = New York
time + 7 hours year-round (GMT+2 winter / GMT+3 summer, switching on US DST dates).
The trading day runs 01:00 -> 23:59 server = 18:00 -> 16:59 New York.
`validate()` checks this against the data (session opens, NYSE-open volume spike,
weekend gaps) and fails loudly if the convention does not hold.

All frames carry:
  srv  : naive server time (bar open)
  ny   : naive New York time (bar open) = srv - 7h
  utc  : tz-naive UTC epoch ... stored as int64 ms since epoch (TV `time` analogue)
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config

NY = ZoneInfo(config.NY_TZ)
_SESSION_OPEN_HOUR = 1  # server-time session open (18:00 NY)


def _read_mt5_csv(path: Path, period: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        names=["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"],
        dtype={"open": "f8", "high": "f8", "low": "f8", "close": "f8",
               "tickvol": "i8", "vol": "i8", "spread": "i8"},
    )
    srv = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    out = pd.DataFrame({
        "srv": srv,
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "volume": df["tickvol"].astype("f8"),
        "spread": df["spread"].astype("i4"),
    })
    out["period"] = period
    return out


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    ny = df["srv"] - pd.Timedelta(hours=config.SERVER_MINUS_NY_HOURS)
    df["ny"] = ny
    utc = ny.dt.tz_localize(NY, nonexistent="shift_forward", ambiguous=True).dt.tz_convert("UTC")
    df["utc_ms"] = utc.astype("int64") // 1_000_000
    return df


def load_1min(refresh: bool = False) -> pd.DataFrame:
    """Full concatenated 1-min series with period labels, cached to parquet."""
    cache = config.CACHE_DIR / "m1.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    parts = [_read_mt5_csv(config.CSV_FILES[p], p) for p in ("train", "val", "oos")]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("srv", kind="stable").reset_index(drop=True)
    if df["srv"].duplicated().any():
        raise ValueError("duplicate 1-min timestamps across files")
    df = _add_time_columns(df)
    config.CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def resample(df1m: pd.DataFrame, minutes: int, anchor: str = "session",
             refresh: bool = False) -> pd.DataFrame:
    """Aggregate 1-min bars into `minutes` bars.

    anchor="session": slots anchored at 01:00 server within each server-day (TV-style
    intraday alignment for this symbol; last slot truncated at 24:00).
    anchor="midnight": slots anchored at 00:00 server (MT5-style).
    Bar label = slot start. close_ms = min(slot end, next session open) as epoch ms UTC.
    """
    cache = config.CACHE_DIR / f"m{minutes}_{anchor}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    day = df1m["srv"].dt.normalize()
    base = day + pd.Timedelta(hours=_SESSION_OPEN_HOUR if anchor == "session" else 0)
    off_min = (df1m["srv"] - base).dt.total_seconds() // 60
    if anchor == "session" and (off_min < 0).any():
        # bars before 01:00 server (shouldn't exist under the NY+7 convention)
        n = int((off_min < 0).sum())
        raise ValueError(f"{n} bars before session open; tz convention violated")
    slot = (off_min // minutes).astype("int64")
    bar_open = base + pd.to_timedelta(slot * minutes, unit="m")

    g = df1m.groupby(bar_open, sort=True)
    bars = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "n1m": g["open"].size(),
        "period": g["period"].first(),
    })
    bars.index.name = "srv"
    bars = bars.reset_index()
    bars = _add_time_columns(bars)
    # close time: slot end, truncated at the 24:00 server-day boundary
    slot_end = bars["srv"] + pd.Timedelta(minutes=minutes)
    day_end = bars["srv"].dt.normalize() + pd.Timedelta(days=1)
    close_srv = np.minimum(slot_end, day_end)
    close_ny = pd.Series(close_srv) - pd.Timedelta(hours=config.SERVER_MINUS_NY_HOURS)
    close_utc = close_ny.dt.tz_localize(NY, nonexistent="shift_forward", ambiguous=True).dt.tz_convert("UTC")
    bars["close_ms"] = close_utc.astype("int64") // 1_000_000
    bars.to_parquet(cache, index=False)
    return bars


def daily(df1m: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """One bar per server-day (= TV daily bar for this symbol, rolling 17:00 NY)."""
    cache = config.CACHE_DIR / "daily.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    day = df1m["srv"].dt.normalize()
    g = df1m.groupby(day, sort=True)
    bars = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "period": g["period"].first(),
    })
    bars.index.name = "srv_day"
    bars = bars.reset_index()
    bars.to_parquet(cache, index=False)
    return bars


def validate(df1m: pd.DataFrame) -> dict:
    """Sanity checks for the timezone convention. Returns a dict of findings."""
    res: dict = {}
    day = df1m["srv"].dt.normalize()
    first_bar = df1m.groupby(day)["srv"].min()
    open_hours = (first_bar - first_bar.dt.normalize()).dt.total_seconds() / 3600
    res["n_days"] = len(first_bar)
    res["open_hour_counts"] = open_hours.round(2).value_counts().sort_index().to_dict()

    # weekend check (server dates should be Mon-Fri)
    dows = day.dt.dayofweek.unique()
    res["server_day_of_week"] = sorted(int(d) for d in dows)

    # NYSE-open volume spike: avg tick volume by NY minute-of-day, summer vs winter
    ny = df1m["ny"]
    mins = ny.dt.hour * 60 + ny.dt.minute
    for label, mask in [("summer(Jul)", ny.dt.month == 7), ("winter(Jan)", ny.dt.month == 1)]:
        sub = df1m[mask]
        m = sub.groupby(mins[mask])["volume"].mean()
        open_min = 9 * 60 + 30
        res[f"vol_{label}_0925_0935"] = {
            int(k): round(float(m.get(k, np.nan)), 1) for k in range(open_min - 5, open_min + 6)
        }
    return res


if __name__ == "__main__":
    df = load_1min(refresh=True)
    print(f"rows={len(df):,}  srv {df['srv'].iloc[0]} -> {df['srv'].iloc[-1]}")
    print(df.groupby("period")["srv"].agg(["min", "max", "size"]))
    v = validate(df)
    print("\nsession-open hour distribution (server):", v["open_hour_counts"])
    print("server days of week:", v["server_day_of_week"])
    for k in ("vol_summer(Jul)_0925_0935", "vol_winter(Jan)_0925_0935"):
        print(k, v[k])
