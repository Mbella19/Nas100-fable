"""Parse TradingView strategy-tester xlsx exports into tidy per-trade frames.

Export structure: 'Trades' sheet has two rows per trade (Exit row first, then
Entry row), sharing the trade number. Timestamps are UTC (verified against
session-close fills and NY-session entries).
"""
from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd

from .. import config


def load_trades(which: str) -> pd.DataFrame:
    path = Path(config.TV_XLSX[which])
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Trades"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header = [str(x) for x in rows[0]]
    df = pd.DataFrame(rows[1:], columns=header)

    ent = df[df["Type"].str.startswith("Entry")].set_index("Trade number")
    ext = df[df["Type"].str.startswith("Exit")].set_index("Trade number")
    out = pd.DataFrame({
        "trade_no": ent.index,
        "direction": ent["Type"].map(lambda s: 1 if "long" in s else -1).to_numpy(),
        "entry_time": pd.to_datetime(ent["Date and time"], utc=True),
        "entry_price": ent["Price USD"].astype(float).to_numpy(),
        "qty": ent["Size (qty)"].astype(float).to_numpy(),
        "entry_signal": ent["Signal"].to_numpy(),
        "exit_time": pd.to_datetime(ext.loc[ent.index, "Date and time"], utc=True).to_numpy(),
        "exit_price": ext.loc[ent.index, "Price USD"].astype(float).to_numpy(),
        "exit_signal": ext.loc[ent.index, "Signal"].to_numpy(),
        "pnl": ent["Net PnL USD"].astype(float).to_numpy(),
        "cum_pnl": ent["Cumulative PnL USD"].astype(float).to_numpy(),
    }).reset_index(drop=True)
    out = out.sort_values("entry_time").reset_index(drop=True)
    out["open_at_export"] = out["exit_signal"] == "Open"
    return out
