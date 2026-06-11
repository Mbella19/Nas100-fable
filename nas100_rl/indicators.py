"""TradingView-exact indicator implementations (vectorized numpy, float64).

All functions replicate Pine v6 `ta.*` semantics including warmup/seeding:
- rma (Wilder): seeded with the SMA of the first `length` values, na before that.
- atr: true range with first-bar tr = high-low.
- dmi: Pine's ta.dmi (rma-smoothed DM/TR, fixnan, adx of |+DI - -DI|/sum).
"""
from __future__ import annotations

import numpy as np


def sma(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= length:
        c = np.cumsum(np.where(np.isnan(x), 0.0, x))
        # Pine ta.sma propagates na inside the window; inputs here are na-free
        out[length - 1:] = (c[length - 1:] - np.concatenate(([0.0], c[:-length]))) / length
    return out


def rma(x: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.rma: na until first window of non-na values completes; SMA seed.

    Handles a leading na-run in `x` (e.g. series derived from `change()`):
    seeding starts after the last leading na.
    """
    n = len(x)
    out = np.full(n, np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) == 0:
        return out
    start = valid[0]
    if n - start < length:
        return out
    seed_idx = start + length - 1
    out[seed_idx] = np.nanmean(x[start:seed_idx + 1])
    alpha = 1.0 / length
    acc = out[seed_idx]
    for i in range(seed_idx + 1, n):
        acc = acc + alpha * (x[i] - acc)
        out[i] = acc
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    return rma(true_range(high, low, close), length)


def ema(x: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.ema: SMA seed, alpha = 2/(length+1)."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < length:
        return out
    out[length - 1] = np.mean(x[:length])
    alpha = 2.0 / (length + 1)
    acc = out[length - 1]
    for i in range(length, n):
        acc = acc + alpha * (x[i] - acc)
        out[i] = acc
    return out


def dmi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        di_length: int, adx_smoothing: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pine ta.dmi -> (+DI, -DI, ADX)."""
    n = len(high)
    up = np.concatenate(([np.nan], np.diff(high)))
    down = np.concatenate(([np.nan], -np.diff(low)))
    plus_dm = np.where(np.isnan(up), np.nan, np.where((up > down) & (up > 0), up, 0.0))
    minus_dm = np.where(np.isnan(down), np.nan, np.where((down > up) & (down > 0), down, 0.0))
    trur = rma(true_range(high, low, close), di_length)
    plus = 100.0 * rma(plus_dm, di_length) / trur
    minus = 100.0 * rma(minus_dm, di_length) / trur
    # Pine fixnan: carry last non-na (division na only during warmup here)
    s = plus + minus
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = np.abs(plus - minus) / np.where(s == 0, 1.0, s)
    adx = 100.0 * rma(dx, adx_smoothing)
    return plus, minus, adx


def pivot_high(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """Pine ta.pivothigh: value at bar i-right confirmed at bar i, if it is strictly
    greater than the `left` bars before it and the `right` bars after it (equal
    neighbors void the pivot). Aligned to the CONFIRMATION bar i (na elsewhere).
    """
    n = len(high)
    out = np.full(n, np.nan)
    w = left + right + 1
    if n < w:
        return out
    win = np.lib.stride_tricks.sliding_window_view(high, w)
    center = win[:, left]
    lmax = win[:, :left].max(axis=1) if left > 0 else np.full(len(win), -np.inf)
    rmax = win[:, left + 1:].max(axis=1) if right > 0 else np.full(len(win), -np.inf)
    ok = (center > lmax) & (center > rmax)
    out[w - 1:] = np.where(ok, center, np.nan)
    return out


def pivot_low(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(low)
    out = np.full(n, np.nan)
    w = left + right + 1
    if n < w:
        return out
    win = np.lib.stride_tricks.sliding_window_view(low, w)
    center = win[:, left]
    lmin = win[:, :left].min(axis=1) if left > 0 else np.full(len(win), np.inf)
    rmin = win[:, left + 1:].min(axis=1) if right > 0 else np.full(len(win), np.inf)
    ok = (center < lmin) & (center < rmin)
    out[w - 1:] = np.where(ok, center, np.nan)
    return out
