"""Broker gateways: the brain (verified Python stack) is gateway-agnostic.

Three implementations of one small surface:
  ReplayGateway     — historical/paper: streams bars from a frame, simulates fills
                      at the next 1-min open (the engine's own fill convention).
  FileBridgeGateway — MT5 under Wine (Mac testing): exchanges CSV files with the
                      NasBridge.mq5 EA inside the terminal's MQL5/Files directory.
  Mt5Gateway        — native MetaTrader5 python package (Windows VPS production).

All prices/levels are index units. `side` is +1 long / -1 short. Lots are broker
lots; the runner converts model qty -> lots.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class OrderResult:
    ok: bool
    ticket: str = ""
    fill_price: float = float("nan")
    comment: str = ""


class BaseGateway:
    """Minimal broker surface. poll_bars() must return only CLOSED M1 bars,
    strictly after the last returned bar, as a frame with columns
    (srv, open, high, low, close, volume, spread)."""

    def poll_bars(self) -> pd.DataFrame: raise NotImplementedError
    def backlog(self) -> pd.DataFrame:
        """One-shot historical bars available at startup (e.g. the EA's bars.csv
        backfill), used by the runner to heal gaps in its stored history BEFORE
        the live loop starts. Must leave the steady-state poll cursor at the
        live edge. Default: nothing available."""
        return pd.DataFrame()
    def account_equity(self) -> float | None: return None
    def symbol_spec(self) -> dict: return {}
    def market_order(self, side: int, lots: float, sl: float | None,
                     tp: float | None, comment: str) -> OrderResult:
        raise NotImplementedError
    def modify(self, ticket: str, sl: float | None = None,
               tp: float | None = None) -> bool: raise NotImplementedError
    def close(self, ticket: str) -> OrderResult: raise NotImplementedError
    def positions(self) -> list[dict]: return []


# ---------------------------------------------------------------- replay/paper
class ReplayGateway(BaseGateway):
    """Streams a historical m1 frame as if live; paper-fills orders at the next
    bar's open (matches the verified engine's market-order semantics)."""

    def __init__(self, m1: pd.DataFrame, start: pd.Timestamp, bars_per_poll: int = 1):
        self.m1 = m1.reset_index(drop=True)
        self.i = int(np.searchsorted(self.m1["srv"].to_numpy(), np.datetime64(start)))
        self.bars_per_poll = bars_per_poll
        self.tickets = 0
        self.open: dict[str, dict] = {}
        self.ledger: list[dict] = []
        self._pending_fills: list[dict] = []

    def poll_bars(self) -> pd.DataFrame:
        j = min(self.i + self.bars_per_poll, len(self.m1))
        out = self.m1.iloc[self.i:j][["srv", "open", "high", "low", "close",
                                      "volume", "spread"]].copy()
        self.i = j
        # paper fills queued by market_order/close land at this poll's first open
        if len(out) and self._pending_fills:
            px = float(out["open"].iloc[0])
            for f in self._pending_fills:
                f["cb"](px)
            self._pending_fills.clear()
        return out

    def exhausted(self) -> bool:
        return self.i >= len(self.m1)

    def market_order(self, side, lots, sl, tp, comment) -> OrderResult:
        self.tickets += 1
        t = f"P{self.tickets}"
        res = OrderResult(ok=True, ticket=t)

        def fill(px):
            self.open[t] = dict(ticket=t, side=side, lots=lots, sl=sl, tp=tp,
                                price=px, comment=comment)
            self.ledger.append(dict(ev="open", ticket=t, side=side, lots=lots,
                                    price=px, sl=sl, tp=tp, comment=comment))
        self._pending_fills.append(dict(cb=fill))
        return res

    def modify(self, ticket, sl=None, tp=None) -> bool:
        if ticket in self.open:
            if sl is not None: self.open[ticket]["sl"] = sl
            if tp is not None: self.open[ticket]["tp"] = tp
            return True
        return False

    def close(self, ticket) -> OrderResult:
        res = OrderResult(ok=True, ticket=ticket)

        def fill(px):
            pos = self.open.pop(ticket, None)
            if pos is not None:
                self.ledger.append(dict(ev="close", ticket=ticket, price=px,
                                        pnl=(px - pos["price"]) * pos["side"] * pos["lots"]))
        self._pending_fills.append(dict(cb=fill))
        return res

    def positions(self) -> list[dict]:
        return list(self.open.values())


# ---------------------------------------------------------------- Wine bridge
class FileBridgeGateway(BaseGateway):
    """File exchange with NasBridge.mq5 running in the (Wine) MT5 terminal.

    EA writes:  bars.csv   (closed M1 bars, appended)
                acks.csv   (one line per executed command: id,ok,ticket,price,msg)
                account.csv(balance,equity), positions.csv, spec.csv
    We write:   commands.csv (id,action,side,lots,sl,tp,ticket,comment) — appended;
                the EA tails it and executes new ids."""

    def __init__(self, bridge_dir: str):
        self.dir = Path(bridge_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bars_path = self.dir / "bars.csv"
        self.cmd_path = self.dir / "commands.csv"
        self.ack_path = self.dir / "acks.csv"
        self._bars_offset = 0
        self._cmd_id = int(time.time())
        self._acks: dict[str, dict] = {}

    def backlog(self) -> pd.DataFrame:
        # first read of bars.csv from offset 0 = the EA's full backfill plus
        # everything appended since; poll cursor lands at the live edge
        return self.poll_bars()

    def poll_bars(self) -> pd.DataFrame:
        if not self.bars_path.exists():
            return pd.DataFrame()
        rows = []
        with open(self.bars_path, "r") as f:
            f.seek(self._bars_offset)
            for line in f:
                line = line.strip()
                if line:
                    rows.append(line.split(","))
            self._bars_offset = f.tell()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["srv", "open", "high", "low", "close",
                                         "volume", "spread"])
        df["srv"] = pd.to_datetime(df["srv"], format="%Y.%m.%d %H:%M")
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype("f8")
        df["spread"] = df["spread"].astype("f8").astype("i4")
        return df

    def _read_acks(self):
        if not self.ack_path.exists():
            return
        with open(self.ack_path) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) >= 4:
                    self._acks[p[0]] = dict(ok=p[1] == "1", ticket=p[2],
                                            price=float(p[3]) if p[3] else float("nan"),
                                            msg=",".join(p[4:]))

    def _command(self, action, side=0, lots=0.0, sl=None, tp=None,
                 ticket="", comment="") -> OrderResult:
        self._cmd_id += 1
        cid = str(self._cmd_id)
        with open(self.cmd_path, "a") as f:
            f.write(f"{cid},{action},{side},{lots:.2f},{'' if sl is None else f'{sl:.2f}'},"
                    f"{'' if tp is None else f'{tp:.2f}'},{ticket},{comment}\n")
        for _ in range(100):                       # wait up to ~10 s for the EA
            time.sleep(0.1)
            self._read_acks()
            if cid in self._acks:
                a = self._acks[cid]
                return OrderResult(ok=a["ok"], ticket=a["ticket"],
                                   fill_price=a["price"], comment=a["msg"])
        return OrderResult(ok=False, comment="bridge timeout")

    def market_order(self, side, lots, sl, tp, comment) -> OrderResult:
        return self._command("open", side, lots, sl, tp, comment=comment)

    def modify(self, ticket, sl=None, tp=None) -> bool:
        return self._command("modify", sl=sl, tp=tp, ticket=ticket).ok

    def close(self, ticket) -> OrderResult:
        return self._command("close", ticket=ticket)

    def account_equity(self) -> float | None:
        p = self.dir / "account.csv"
        if p.exists():
            try:
                bal, eq = p.read_text().strip().split(",")[:2]
                return float(eq)
            except (ValueError, OSError):
                return None
        return None

    def symbol_spec(self) -> dict:
        p = self.dir / "spec.csv"
        if p.exists():
            try:
                v = p.read_text().strip().split(",")
                return dict(lot_step=float(v[0]), min_lot=float(v[1]),
                            usd_per_point_per_lot=float(v[2]))
            except (ValueError, OSError):
                pass
        return {}


# ---------------------------------------------------------------- native MT5
class Mt5Gateway(BaseGateway):
    """Official MetaTrader5 package (Windows / Windows VPS only)."""

    def __init__(self, symbol: str, terminal_path: str | None = None):
        import MetaTrader5 as mt5            # noqa: import guarded by platform
        self.mt5 = mt5
        if not (mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        self.symbol = symbol
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select({symbol}) failed")
        self._last_srv: pd.Timestamp | None = None

    def backlog(self) -> pd.DataFrame:
        # deep history for startup gap-healing (~2.5 weeks of sessions)
        return self._closed_rates(20_000)

    def poll_bars(self) -> pd.DataFrame:
        return self._closed_rates(3_000)

    def _closed_rates(self, count: int) -> pd.DataFrame:
        mt5 = self.mt5
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, count)
        if rates is None or len(rates) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(rates[:-1])              # drop the forming bar
        df["srv"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})[
            ["srv", "open", "high", "low", "close", "volume", "spread"]]
        if self._last_srv is not None:
            df = df[df["srv"] > self._last_srv]
        if len(df):
            self._last_srv = df["srv"].iloc[-1]
        return df.reset_index(drop=True)

    def account_equity(self) -> float | None:
        info = self.mt5.account_info()
        return float(info.equity) if info else None

    def symbol_spec(self) -> dict:
        si = self.mt5.symbol_info(self.symbol)
        if si is None:
            return {}
        return dict(lot_step=si.volume_step, min_lot=si.volume_min,
                    usd_per_point_per_lot=si.trade_tick_value / max(si.trade_tick_size, 1e-9))

    def market_order(self, side, lots, sl, tp, comment) -> OrderResult:
        mt5 = self.mt5
        tick = mt5.symbol_info_tick(self.symbol)
        req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=self.symbol, volume=float(lots),
                   type=mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL,
                   price=tick.ask if side > 0 else tick.bid,
                   deviation=50, comment=comment[:26],
                   type_filling=mt5.ORDER_FILLING_IOC, type_time=mt5.ORDER_TIME_GTC)
        if sl is not None: req["sl"] = float(sl)
        if tp is not None: req["tp"] = float(tp)
        r = mt5.order_send(req)
        ok = r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(ok=ok, ticket=str(r.order) if ok else "",
                           fill_price=r.price if ok else float("nan"),
                           comment=str(r.retcode) if r else "send failed")

    def modify(self, ticket, sl=None, tp=None) -> bool:
        mt5 = self.mt5
        for p in mt5.positions_get(symbol=self.symbol) or []:
            if str(p.ticket) == str(ticket):
                req = dict(action=mt5.TRADE_ACTION_SLTP, symbol=self.symbol,
                           position=p.ticket,
                           sl=float(sl if sl is not None else p.sl),
                           tp=float(tp if tp is not None else p.tp))
                r = mt5.order_send(req)
                return r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
        return False

    def close(self, ticket) -> OrderResult:
        mt5 = self.mt5
        for p in mt5.positions_get(symbol=self.symbol) or []:
            if str(p.ticket) == str(ticket):
                tick = mt5.symbol_info_tick(self.symbol)
                side = -1 if p.type == mt5.POSITION_TYPE_BUY else 1
                req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=self.symbol,
                           volume=p.volume, position=p.ticket,
                           type=mt5.ORDER_TYPE_SELL if side < 0 else mt5.ORDER_TYPE_BUY,
                           price=tick.bid if side < 0 else tick.ask,
                           deviation=50, comment="bot close",
                           type_filling=mt5.ORDER_FILLING_IOC, type_time=mt5.ORDER_TIME_GTC)
                r = mt5.order_send(req)
                ok = r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
                return OrderResult(ok=ok, ticket=ticket,
                                   fill_price=r.price if ok else float("nan"))
        return OrderResult(ok=False, comment="position not found")

    def positions(self) -> list[dict]:
        out = []
        for p in self.mt5.positions_get(symbol=self.symbol) or []:
            out.append(dict(ticket=str(p.ticket), lots=p.volume,
                            side=1 if p.type == self.mt5.POSITION_TYPE_BUY else -1,
                            price=p.price_open, sl=p.sl, tp=p.tp, comment=p.comment))
        return out
