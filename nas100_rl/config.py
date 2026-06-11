"""Central configuration: paths, frozen strategy parameters (as actually used in the
TradingView exports' Properties sheets — NOT script defaults), cost model, RL settings.

Strategy parameters are FROZEN. They define the verified 1:1 ports and must never be
tuned by the RL layer.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "NAS100 DATA"
TV_RESULTS_DIR = PROJECT_ROOT / "Tradingview Results"
CACHE_DIR = PROJECT_ROOT / "data_cache"
REPORTS_DIR = PROJECT_ROOT / "nas100_rl" / "reports"
CHECKPOINT_DIR = PROJECT_ROOT / "nas100_rl" / "checkpoints"

CSV_FILES = {
    "train": DATA_DIR / "NAS100 TRAINING.csv",
    "val": DATA_DIR / "NAS100 VALIDATION.csv",
    "oos": DATA_DIR / "NAS100 LOCKED OOS.csv",
}

TV_XLSX = {
    "s2": TV_RESULTS_DIR / "S2__5b_Momentum_Burst_-_Declining_Risk_PEPPERSTONE_NAS100_2026-06-11.xlsx",
    "dmi": TV_RESULTS_DIR / "PROJECT_1.8_DMI_PEPPERSTONE_NAS100_2026-06-11.xlsx",
    "cpmt": TV_RESULTS_DIR / "CPMT_v12_PEPPERSTONE_NAS100_2026-06-11.xlsx",
}

# MT5 server time = New York time + 7 hours, year-round (NY-close-at-midnight broker
# convention; validated against the data by data.validate(): session-open hours,
# weekday set, NYSE-open volume spike at 09:30 NY summer AND winter).
SERVER_MINUS_NY_HOURS = 7
NY_TZ = "America/New_York"

# ---------------------------------------------------------------------------
# S2: 5-Bar Momentum Burst — Declining Risk (5-minute chart)
# Values from S2 xlsx Properties (lookback/risk differ from script defaults!).
# ---------------------------------------------------------------------------
S2 = dict(
    timeframe="5min",
    lookback=4,
    threshold=4.5,
    atr_period=14,
    stop_mult=3.0,
    stop_mult_2nd=9.0,
    max_per_day=2,
    risk_pct=0.006,
    initial_capital=100_000.0,
    commission_per_contract_per_order=1.25,
    qty_min=1.0,
    qty_max=200.0,
    atr_valid_min=1.0,
    atr_valid_max=100.0,
    friday_exit_utc_hour=18,
)

# ---------------------------------------------------------------------------
# PROJECT 1.8 DMI (1-minute chart)
# ---------------------------------------------------------------------------
DMI = dict(
    timeframe="1min",
    use_first_candle=False,
    use_volume_spike=True,
    volume_multiplier=1.0,
    volume_sma_len=14,
    use_dmi_filter=True,
    dmi_length=14,
    adx_smoothing=14,
    min_adx=30.0,
    atr_length=5,
    atr_multiplier=5.0,        # trail distance & initial stop
    trail_start_multiplier=1.5,
    tp_atr_multiplier=15.0,
    risk_per_trade=200.0,      # fixed dollars
    point_value=1.0,
    min_contracts=0.01,
    max_trades_per_day=2,      # nominal; entryTaken jank caps at 1 effective
    session_start_ny=(9, 30),  # NYSE 09:30-16:00, exchange tz = America/New_York
    session_end_ny=(16, 0),
    initial_capital=10_000.0,
)

# ---------------------------------------------------------------------------
# CPMT v12 (30-minute chart, netted "Both (single position)" mode)
# ---------------------------------------------------------------------------
CPMT = dict(
    timeframe="30min",
    direction="both",          # netted single position, as run on TV
    risk_pct=2.0,              # % of realized equity
    qty_step=0.01,
    qty_min=0.01,
    qty_max=100.0,
    trail_mult=2.5,            # × stream ATR(14)
    stop_floor=1.0,
    stop_cap=2.5,
    ts_mult=3.0,               # time stop × pattern width
    gate_len=10,               # daily EMA length
    gate_band=0.25,            # × daily ATR(14)
    blackout_hours_gmt3=(23, 0, 1),
    tol=0.25,
    max_width=200,
    max_pat=20,
    point_value=1.0,
    initial_capital=10_000.0,
    # streams: (tf_minutes, pivot_lengths, chart_bars_per_htf_bar k)
    streams=[
        ("240", (6, 7, 8), 8.0),
        ("180", (10, 12, 14), 6.0),
        ("120", (10, 12, 14), 4.0),
        ("360", (4, 5, 6), 12.0),
        ("60", (8, 10, 12, 14), 2.0),
        ("30", (20, 24, 28), 1.0),
    ],
    gated=[False, True, True, True, True, True],  # only the 4h core is ungated
)

# ---------------------------------------------------------------------------
# RL portfolio settings
# ---------------------------------------------------------------------------
RL = dict(
    initial_capital=100_000.0,
    # Native sizing scaled to portfolio capital (clamps are broker artifacts and
    # scale with the capital ratio; signal/exit logic untouched):
    # S2 native at 100k -> unchanged. CPMT native at 10k -> qty clamps x10.
    # DMI fixed $200 risk at 10k -> $2000 at 100k (fixed, non-compounding).
    dmi_risk_dollars=2_000.0,
    cpmt_qty_max=1_000.0,
    per_trade_risk_cap=0.03,   # of current equity, after action multiplier
    total_open_risk_cap=0.06,
    # de-risk-only action space: upsizing proved Sharpe-neutral but drawdown-positive
    # (Sharpe is leverage-invariant), and Sharpe is the binding gate metric
    action_multipliers=(0.0, 0.5, 1.0),
)

# ---------------------------------------------------------------------------
# Live runner (MT5 execution; the verified Python stack stays the brain)
# ---------------------------------------------------------------------------
LIVE = dict(
    symbol="NAS100",
    ensemble_tag="deploy_live",          # 9-seed mean-logits ensemble, never one seed
    # model account runs at 100k scale (training distribution); broker lots =
    # model_qty / usd_per_point_per_lot * risk_scale (0.1 => trade a 10k-like account)
    risk_scale=0.10,
    usd_per_point_per_lot=1.0,           # fallback; overridden by broker symbol spec
    lot_step=0.01,
    min_lot=0.01,
    magic={"s2": 771001, "dmi": 771002, "cpmt": 771003},
    # windowed-replay spans: long enough for bit-convergent indicators + all
    # path-dependent state (positions, day-state, pattern buffers); parity is
    # asserted against the frozen signal stream by `runner --selftest`
    s2_window_days=12,                   # holds <1 session; ATR14@5m converges <2d
    dmi_window_days=7,                   # holds p99 0.2d; RMA14@1m converges <1d
    cpmt_anchor="2023-01-03",            # fixed anchor: daily ATR14 RMA needs ~490 daily bars
    feature_window_days=420,             # 30m features: ema100/atr-regime convergence
    bridge_dir=("/Users/gervaciusjr/Library/Application Support/"
                "net.metaquotes.wine.metatrader5/drive_c/Program Files/"
                "MetaTrader 5/MQL5/Files"),  # FileBridgeGateway dir (MT5-under-Wine)
    live_store="live_m1.parquet",        # collector append target in CACHE_DIR
    journal="live_journal.jsonl",        # every decision/order/fill, append-only
    poll_seconds=2.0,
)
