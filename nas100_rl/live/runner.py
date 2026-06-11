"""Live runner: the verified Python stack as the brain, MT5 as the hands.

Modes
-----
--selftest [--quick]   Parity proof on historical data (validation period, no
                       OOS touch): the windowed-replay engines must reproduce
                       the frozen signal stream, the live feature builder must
                       reproduce the cached feature matrix, and the model
                       account + ensemble must reproduce SignalEnv's decisions.
                       Run this after ANY change, and at every deployment site.
--paper N              Stream the last N days of history through the full live
                       loop with a paper gateway (orders simulated, no broker).
--live --gateway file|mt5
                       Trade. `file` = NasBridge.mq5 file bridge (MT5 under
                       Wine on the Mac); `mt5` = native package (Windows VPS).

The model account (100k scale) feeds the policy; broker orders are scaled by
LIVE['risk_scale']. Every decision/order/event is appended to the journal.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, data
from ..rl import signals as sigmod
from ..rl.train import load_ensemble
from .account import ModelAccount
from .engines import StrategyEngine
from .features_live import FeatureBuilder
from . import gateway as gw

L = config.LIVE
BARLEN = {"s2": 5, "dmi": 1, "cpmt": 30}
A_MULTS = config.RL["action_multipliers"]


def _slot(srv: pd.Timestamp, minutes: int) -> tuple:
    day = srv.normalize()
    off = (srv - day - pd.Timedelta(hours=1)).total_seconds() // 60
    return (day.value, int(off // minutes))


def _slot_start(srv: pd.Timestamp, minutes: int) -> pd.Timestamp:
    day, k = _slot(srv, minutes)
    return pd.Timestamp(day) + pd.Timedelta(hours=1) + pd.Timedelta(minutes=k * minutes)


def _mk_emission_from_row(r) -> dict:
    return dict(strategy=r.strategy, signal_ms=int(r.signal_ms),
                direction=int(r.direction), stop_dist=float(r.stop_dist),
                atr_sig=float(r.atr_sig), nth_of_day=int(r.nth_of_day),
                adx_sig=float(r.adx_sig), di_spread=float(r.di_spread),
                pattern=("" if pd.isna(r.pattern) else str(r.pattern)),
                stream=int(r.stream), width=float(r.width))


def _prep_live_bars(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["period"] = "live"
    return data._add_time_columns(bars)


def _merge_backlog(m1: pd.DataFrame, back: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Startup-only gap healing: insert backlog bars missing from held history
    (engines must see one consistent timeline from their first step). The
    steady-state loop stays strictly-append — never splice bars mid-run."""
    if back is None or not len(back):
        return m1, 0
    back = back.drop_duplicates("srv")
    new = back[~back["srv"].isin(m1["srv"]) & (back["srv"] > m1["srv"].iloc[0])]
    if not len(new):
        return m1, 0
    m1 = (pd.concat([m1, _prep_live_bars(new)], ignore_index=True)
          .sort_values("srv", kind="mergesort", ignore_index=True))
    return m1, len(new)


def _gap_spans(m1: pd.DataFrame, lookback_days: int = 45) -> list[tuple[str, str]]:
    """Missing-session detector over the recent timeline: jumps wider than a
    session boundary that aren't the weekend shape (holidays may false-positive
    — this only feeds a journal warning, never a behavior change)."""
    srv = m1.loc[m1["srv"] >= m1["srv"].iloc[-1] - pd.Timedelta(days=lookback_days),
                 "srv"].reset_index(drop=True)
    dt = srv.diff()
    out = []
    # one cleanly missing session = a 25h jump (23:59 -> 01:00 two days later);
    # the ordinary overnight jump is 1h — 20h splits them with margin
    for i in np.where(dt > pd.Timedelta(hours=20))[0]:
        a, b = srv.iloc[i - 1], srv.iloc[i]
        if a.dayofweek == 4 and b.dayofweek in (6, 0) \
                and dt.iloc[i] <= pd.Timedelta(hours=56):
            continue                                   # ordinary weekend
        out.append((str(a), str(b)))
    return out


class Journal:
    def __init__(self, path: Path):
        self.path = path

    def log(self, **kw):
        kw["ts"] = pd.Timestamp.utcnow().isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(kw, default=str) + "\n")


# ===================================================================== selftest
def selftest(quick: bool = False, cpmt_every: int = 1) -> bool:
    print("== live-runner parity selftest (validation period; no OOS touch) ==")
    t0 = time.time()
    m1 = data.load_1min()
    frozen = sigmod.build(basis="live")
    val = frozen[frozen["period"] == "val"].reset_index(drop=True)
    if quick:
        last_day = pd.Timestamp(val["signal_ms"].iloc[-1], unit="ms", tz="UTC") \
            .tz_convert(config.NY_TZ).tz_localize(None)
        cut_ms = int(val["signal_ms"].iloc[-1]) - 30 * 86_400_000
        val = val[val["signal_ms"] >= cut_ms].reset_index(drop=True)
    live_from_ms = int(val["signal_ms"].iloc[0])

    srv_ms = m1["utc_ms"].to_numpy()
    days = m1["srv"].dt.normalize()
    day_list = days[srv_ms >= live_from_ms].unique()

    ok_all = True

    # ---- 1) engine parity: emissions + closures vs the frozen stream --------
    engines = {}
    for name in ("s2", "dmi", "cpmt"):
        pre = frozen[(frozen["strategy"] == name)]
        seen_sig = {(int(s), int(d)) for s, d in
                    zip(pre.loc[pre["signal_ms"] < live_from_ms, "signal_ms"],
                        pre.loc[pre["signal_ms"] < live_from_ms, "direction"])}
        seen_ent = {int(x) for x in pre.loc[pre["signal_ms"] < live_from_ms, "entry_ms"]}
        engines[name] = StrategyEngine(name, basis="live", live_from_ms=live_from_ms,
                                       seen_signals=seen_sig, seen_entries=seen_ent)

    emissions, closures = [], []
    times = {k: [] for k in engines}
    for di, d in enumerate(day_list):
        d_end = d + pd.Timedelta(days=1)
        m1_thru = m1[m1["srv"] < d_end]
        for name, eng in engines.items():
            if name == "cpmt" and di % cpmt_every and d != day_list[-1]:
                continue
            tt = time.time()
            ev = eng.step(m1_thru)
            times[name].append(time.time() - tt)
            emissions += ev["emissions"]
            closures += ev["closed"]

    em = pd.DataFrame(emissions)
    # signal-STREAM parity is checked over every bar the engines saw (incl. 2026
    # data): no policy runs and no performance is computed here — this is
    # frozen-strategy replica verification, not an OOS model evaluation
    fz = frozen.loc[frozen["signal_ms"] >= live_from_ms,
                    ["strategy", "signal_ms", "direction", "stop_dist", "atr_sig"]]
    ekeys = set(zip(em["strategy"], em["signal_ms"], em["direction"])) if len(em) else set()
    fkeys = set(zip(fz["strategy"], fz["signal_ms"], fz["direction"]))
    missing = fkeys - ekeys
    extra = ekeys - fkeys
    print(f"[1] emissions: live {len(ekeys)} vs frozen {len(fkeys)} "
          f"| missing {len(missing)} extra {len(extra)}")
    if missing or extra:
        ok_all = False
        for k in sorted(missing)[:6]: print("    missing:", k[0],
                                            pd.Timestamp(k[1], unit="ms"), k[2])
        for k in sorted(extra)[:6]: print("    extra:  ", k[0],
                                          pd.Timestamp(k[1], unit="ms"), k[2])
    if len(em):
        j = em.merge(fz, on=["strategy", "signal_ms", "direction"], suffixes=("", "_f"))
        dd = (j["stop_dist"] - j["stop_dist_f"]).abs().max()
        da = (j["atr_sig"] - j["atr_sig_f"]).abs().max()
        print(f"    stop_dist max|d|={dd:.2e}  atr max|d|={da:.2e}")
        ok_all &= bool(dd < 1e-6 and da < 1e-6)

    cl = pd.DataFrame(closures)
    if len(cl):
        j = cl.merge(frozen, on=["strategy", "entry_ms"], suffixes=("", "_f"))
        bad = j[(j["exit_ms"] != j["exit_ms_f"])
                | ((j["exit_price"] - j["exit_price_f"]).abs() > 1e-6)]
        dpx = (j["exit_price"] - j["exit_price_f"]).abs().max()
        dpc = (j["pc_pnl"] - j["pc_pnl_f"]).abs().max()
        unmatched = cl.merge(frozen[["strategy", "entry_ms"]], on=["strategy", "entry_ms"],
                             how="left", indicator=True)
        unm = unmatched[unmatched["_merge"] == "left_only"]
        print(f"[1] closures: {len(cl)} live rows, {len(unm)} unmatched, "
              f"exit mismatches {len(bad)}, exit_px max|d|={dpx:.2e}, pc_pnl max|d|={dpc:.2e}")
        if len(bad):
            print("    mismatches by strategy:", bad.groupby("strategy").size().to_dict())
            for r in bad.head(6).itertuples():
                print(f"    {r.strategy} entry {pd.Timestamp(r.entry_ms, unit='ms')} "
                      f"exit {pd.Timestamp(r.exit_ms, unit='ms')} vs "
                      f"{pd.Timestamp(r.exit_ms_f, unit='ms')} "
                      f"px {r.exit_price:.1f} vs {r.exit_price_f:.1f} ({r.exit_reason} vs {r.exit_reason_f})")
        if len(unm):
            print("    unmatched by strategy:", unm.groupby("strategy").size().to_dict())
            for r in unm.head(6).itertuples():
                print(f"    unmatched {r.strategy} entry {pd.Timestamp(r.entry_ms, unit='ms')}")
        ok_all &= bool(len(bad) == 0 and len(unm) == 0)
    for name, ts in times.items():
        if ts:
            print(f"    {name}: {len(ts)} steps, mean {np.mean(ts):.2f}s, max {np.max(ts):.2f}s")

    # ---- 2) feature parity ---------------------------------------------------
    from ..rl import features as featmod
    fdf, names = featmod.build(basis="live")
    fb = FeatureBuilder(basis="live")
    fb.set_history(frozen)
    fmat = fdf[names].to_numpy(dtype=np.float32)
    val_idx = np.where(fdf["period"].to_numpy() == "val")[0]
    if quick:
        val_idx = val_idx[-len(val):]
    worst = 0.0
    worst_name = ""
    mark_dev = 0.0
    z_live: dict[int, np.ndarray] = {}     # live-built vectors, reused by [3b]
    for k, i in enumerate(val_idx):
        r = fdf.iloc[i]
        fr = frozen[frozen["trade_id"] == r["trade_id"]].iloc[0]
        sig_day_end = pd.Timestamp(int(fr["signal_ms"]), unit="ms", tz="UTC") \
            .tz_convert(config.NY_TZ).tz_localize(None) + pd.Timedelta(hours=7)
        m1_thru = m1[m1["srv"] < sig_day_end.normalize() + pd.Timedelta(days=1)]
        z, mark = fb.row(_mk_emission_from_row(fr), m1_thru)
        z_live[int(i)] = z
        d = np.abs(z - fmat[i])
        if d.max() > worst:
            worst = float(d.max()); worst_name = names[int(d.argmax())]
        mark_dev = max(mark_dev, abs(mark - float(r["mark_price"])))
    print(f"[2] features: {len(val_idx)} signals, max |dz| = {worst:.2e} ({worst_name}), "
          f"mark max|d| = {mark_dev:.2e}")
    ok_all &= bool(worst < 5e-3 and mark_dev < 1e-6)

    # ---- 3) decision parity vs SignalEnv ------------------------------------
    from ..rl import evaluate as ev
    world = ev.load_all(basis="live")
    pol = load_ensemble(L["ensemble_tag"])
    i0, i1 = world["periods"]["val"]
    env_actions = []
    from ..rl.env import SignalEnv
    env = SignalEnv(world["sig"], world["feats"], world["mark"])
    obs = env.reset(i0, i1)
    done = False
    while not done:
        a = pol(obs, env.i)
        env_actions.append(int(a))
        obs, _, done, _ = env.step(a)

    acct = ModelAccount(mode="parity")
    sig = world["sig"]
    mine = []
    port_dev = 0.0
    for i in range(i0, i1):
        r = sig.iloc[i]
        acct.on_signal(int(r["signal_ms"]), float(world["mark"][i]))
        port = acct.obs_port()
        o = np.concatenate([world["feats"][i], port])
        a = int(pol(o, i))
        mine.append(a)
        mult = A_MULTS[a]
        qty = acct.size(r["strategy"], float(r["stop_dist"]), mult)
        if qty > 0:
            acct.register((r["strategy"], int(r["signal_ms"]), int(r["direction"])),
                          r["strategy"], int(r["direction"]), qty, float(r["stop_dist"]),
                          int(r["entry_ms"]), float(r["entry_price"]),
                          exit_ms=int(r["exit_ms"]), pc_pnl=float(r["pc_pnl"]))
    agree = float(np.mean(np.array(mine) == np.array(env_actions)))
    print(f"[3] decisions vs SignalEnv: {agree:.1%} agreement "
          f"({int((1-agree)*len(mine))} of {len(mine)} differ)")
    ok_all &= bool(agree == 1.0)

    # ---- 3b) decisions again, on the LIVE-built feature vectors --------------
    # [2] proves live z ~ cached within 5e-3 and [3] proves cached z -> identical
    # actions; this closes the composition gap: the vectors the live builder
    # actually produces must yield the identical action sequence end-to-end.
    chk = list(val_idx[:5])
    assert all(np.allclose(world["feats"][i], fmat[i], atol=1e-6) for i in chk), \
        "feature matrix misaligned between features.build and evaluate.load_all"
    acct2 = ModelAccount(mode="parity")
    mine_live = []
    for i in range(i0, i1):
        r = sig.iloc[i]
        acct2.on_signal(int(r["signal_ms"]), float(world["mark"][i]))
        z = z_live.get(int(i), world["feats"][i])
        o = np.concatenate([z, acct2.obs_port()])
        a = int(pol(o, i))
        mine_live.append(a)
        qty = acct2.size(r["strategy"], float(r["stop_dist"]), A_MULTS[a])
        if qty > 0:
            acct2.register((r["strategy"], int(r["signal_ms"]), int(r["direction"])),
                           r["strategy"], int(r["direction"]), qty, float(r["stop_dist"]),
                           int(r["entry_ms"]), float(r["entry_price"]),
                           exit_ms=int(r["exit_ms"]), pc_pnl=float(r["pc_pnl"]))
    agree_b = float(np.mean(np.array(mine_live) == np.array(env_actions)))
    n_lz = sum(1 for i in range(i0, i1) if int(i) in z_live)
    print(f"[3b] decisions on live-built features: {agree_b:.1%} agreement "
          f"({n_lz}/{i1 - i0} decisions covered by live z)")
    ok_all &= bool(agree_b == 1.0)

    print(f"\nselftest {'PASS' if ok_all else 'FAIL'} in {time.time()-t0:.0f}s")
    return ok_all


# ===================================================================== live loop
class LiveRunner:
    def __init__(self, gateway: gw.BaseGateway, basis: str = "live",
                 m1: pd.DataFrame | None = None, frozen: pd.DataFrame | None = None,
                 journal_path: Path | None = None, store_path: Path | None = None):
        self.gw = gateway
        self.basis = basis
        self.journal = Journal(journal_path or config.CACHE_DIR / L["journal"])
        self.frozen = frozen if frozen is not None else sigmod.build(basis=basis)
        healed = 0
        if m1 is None:
            m1 = data.load_1min().copy()
            store = config.CACHE_DIR / L["live_store"]
            if store.exists():
                m1 = (pd.concat([m1, pd.read_parquet(store)], ignore_index=True)
                      .drop_duplicates("srv", keep="first")
                      .sort_values("srv", kind="mergesort", ignore_index=True))
            # heal holes from whatever history the gateway can serve (EA bars.csv
            # backfill / MT5 deep history) BEFORE the engines fix their timeline;
            # restarts used to shrink the store and leave such holes behind
            m1, healed = _merge_backlog(m1, self.gw.backlog())
            self.store_path = store
            self.state_path = config.CACHE_DIR / "live_state.json"
        else:
            self.store_path = store_path or (config.CACHE_DIR / "paper_m1.parquet")
            self.state_path = self.store_path.with_suffix(".state.json")
        self.m1 = m1
        self.live_from_ms = int(self.m1["utc_ms"].iloc[-1]) + 1
        if healed:
            self.journal.log(ev="gap_heal", bars=int(healed))
        for a, b in _gap_spans(self.m1):
            self.journal.log(ev="gap_warning", span=[a, b])
        self.engines = {}
        for name in ("s2", "dmi", "cpmt"):
            pre = self.frozen[self.frozen["strategy"] == name]
            self.engines[name] = StrategyEngine(
                name, basis=basis, live_from_ms=self.live_from_ms,
                seen_signals={(int(s), int(d)) for s, d in
                              zip(pre["signal_ms"], pre["direction"])},
                seen_entries={int(x) for x in pre["entry_ms"]})
        self.fb = FeatureBuilder(basis=basis)
        self.fb.set_history(self.frozen)
        self.acct = ModelAccount(mode="live")
        self.pol = load_ensemble(L["ensemble_tag"])
        self.last_slot = {k: None for k in self.engines}
        spec = {**dict(lot_step=L["lot_step"], min_lot=L["min_lot"],
                       usd_per_point_per_lot=L["usd_per_point_per_lot"]),
                **(self.gw.symbol_spec() or {})}
        self.spec = spec
        self.tickets: dict = {}            # trade_key -> broker ticket
        self.halted = False
        self._restore_state()              # re-adopt account/tickets/halt flag
        self.journal.log(ev="start", live_from_ms=self.live_from_ms, spec=spec)

    # ----------------------------------------------------------------- state
    def _save_state(self):
        """Persist what a restart must not lose: the model account (source of
        the policy's portfolio observations), open broker tickets, halt flag."""
        st = self.acct.to_state()
        st["open"] = [dict(t, trade_key=list(t["trade_key"])) for t in st["open"]]
        out = dict(acct=st,
                   tickets=[[list(k), v] for k, v in self.tickets.items()],
                   halted=self.halted,
                   last_bar_ms=int(self.m1["utc_ms"].iloc[-1]),
                   ts=pd.Timestamp.utcnow().isoformat())
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out))
        tmp.replace(self.state_path)

    def _restore_state(self):
        if not self.state_path.exists():
            return
        try:
            st = json.loads(self.state_path.read_text())
            acct = ModelAccount.from_state(st["acct"])
            fresh, stale = [], set()
            for t in acct.open:
                t["trade_key"] = tuple(t["trade_key"])
                # engines absorb (never re-emit) entries >30d before live start,
                # so a trade that old could never settle — drop it loudly
                if t["entry_ms"] < self.live_from_ms - 25 * 86_400_000:
                    stale.add(t["trade_key"])
                else:
                    fresh.append(t)
            acct.open = fresh
            self.acct = acct
            self.tickets = {tuple(k): v for k, v in st.get("tickets", [])
                            if tuple(k) not in stale}
            self.halted = bool(st.get("halted", False))
            self.journal.log(ev="state_restored", open=len(fresh),
                             tickets=len(self.tickets), halted=self.halted,
                             equity=round(acct.mark_eq, 2), saved=st.get("ts"),
                             dropped_stale=[str(k) for k in stale])
        except Exception as ex:            # corrupt state: start fresh, loudly
            self.journal.log(ev="state_restore_failed", err=repr(ex))

    # ----------------------------------------------------------------- bars
    def _append_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        # the EA's startup backfill (and any re-read after restart) overlaps
        # bars we already hold — keep strictly newer bars only, and return them
        # so the event loop never re-grinds engines over already-seen bars
        bars = bars[bars["srv"] > self.m1["srv"].iloc[-1]]
        if not len(bars):
            return bars
        bars = _prep_live_bars(bars)
        self.m1 = pd.concat([self.m1, bars], ignore_index=True)
        # persist EVERY live bar ever seen (atomic replace). This must stay a
        # union: rewriting only the current session's tail is what carved the
        # June-10 hole — each restart silently discarded its predecessors' bars
        tail = self.m1[self.m1["period"] == "live"]
        tmp = self.store_path.with_suffix(".tmp.parquet")
        tail.to_parquet(tmp, index=False)
        tmp.replace(self.store_path)
        return bars

    # ----------------------------------------------------------------- events
    def _process_events(self, events: list[dict]):
        events.sort(key=lambda e: (e["ms"], 0 if e["type"] == "close" else 1))
        for e in events:
            if e["type"] == "close":
                row = e["row"]
                key = (row["strategy"], int(row["signal_ms"]), int(row["direction"]))
                self.acct.on_close(key, float(row["pc_pnl"]), int(row["exit_ms"]))
                self.fb.append_history([row])
                t = self.tickets.pop(key, None)
                if t:
                    res = self.gw.close(t)
                    self.journal.log(ev="broker_close", key=str(key), ticket=t,
                                     ok=res.ok, px=res.fill_price)
                self.journal.log(ev="mech_close", key=str(key),
                                 exit_ms=int(row["exit_ms"]), pc_pnl=float(row["pc_pnl"]))
            else:
                self._decide(e["em"], e["now"])

    def _decide(self, em: dict, now_ms: int):
        z, mark = self.fb.row(em, self.m1)
        self.acct.on_signal(int(em["signal_ms"]), mark)
        obs = np.concatenate([z, self.acct.obs_port()])
        a = int(self.pol(obs, 0))
        mult = A_MULTS[a]
        qty = 0.0 if self.halted else self.acct.size(em["strategy"], em["stop_dist"], mult)
        key = (em["strategy"], int(em["signal_ms"]), int(em["direction"]))
        self.journal.log(ev="decision", key=str(key), action=a, mult=mult, qty=qty,
                         equity=self.acct.mark_eq, halted=self.halted)
        if qty <= 0:
            return
        entry_ms = int(em["signal_ms"]) + BARLEN[em["strategy"]] * 60_000
        self.acct.register(key, em["strategy"], int(em["direction"]), qty,
                           float(em["stop_dist"]), entry_ms, None)
        # stale signals (catch-up after downtime): keep the model account on its
        # idealized path, but never chase the market with a late broker order —
        # the mechanical closure event will settle the model trade regardless
        if now_ms - entry_ms > 2 * BARLEN[em["strategy"]] * 60_000:
            self.journal.log(ev="stale_signal_no_order", key=str(key),
                             age_min=(now_ms - entry_ms) / 60_000)
            return
        lots = qty * L["risk_scale"] / self.spec["usd_per_point_per_lot"]
        lots = max(round(lots / self.spec["lot_step"]) * self.spec["lot_step"], 0.0)
        if lots < self.spec["min_lot"]:
            self.journal.log(ev="skip_minlot", key=str(key), lots=lots)
            return
        sl = em.get("sl_level")
        tp = em.get("tp_level")
        res = self.gw.market_order(int(em["direction"]), lots,
                                   None if sl is None or np.isnan(sl) else float(sl),
                                   None if tp is None or np.isnan(tp) else float(tp),
                                   comment=f"{em['strategy']}|{em['signal_ms']}")
        if res.ok:
            self.tickets[key] = res.ticket
            if not np.isnan(res.fill_price):
                self.acct.on_fill(key, res.fill_price)
        self.journal.log(ev="broker_open", key=str(key), ok=res.ok, ticket=res.ticket,
                         lots=lots, fill=res.fill_price, msg=res.comment)

    def _sweep_fills(self):
        """Backfill entry prices the broker reports asynchronously (paper fills
        land one poll later) so open trades mark correctly in the model account."""
        if not self.tickets:
            return
        pos = {p.get("ticket"): p for p in (self.gw.positions() or [])}
        for key, t in list(self.tickets.items()):
            p = pos.get(t)
            if not p:
                continue
            for tr in self.acct.open:
                ep = tr["entry_price"]
                if tr["trade_key"] == key and (ep is None or
                                               (isinstance(ep, float) and np.isnan(ep))):
                    self.acct.on_fill(key, float(p["price"]))

    # ----------------------------------------------------------------- loop
    def run(self, poll_seconds: float | None = None, max_polls: int | None = None):
        polls = 0
        while True:
            bars = self.gw.poll_bars()
            if bars is not None and len(bars):
                bars = self._append_bars(bars)
            if bars is not None and len(bars):
                events = []
                for _, bar in bars.iterrows():
                    for name, eng in self.engines.items():
                        L_min = BARLEN[name]
                        s = _slot(bar["srv"], L_min)
                        if self.last_slot[name] is None:
                            self.last_slot[name] = s
                            continue
                        if s == self.last_slot[name]:
                            continue
                        self.last_slot[name] = s
                        # the engine must only ever see COMPLETED chart bars:
                        # dmi's chart bar is the m1 bar itself (include it);
                        # for s2/cpmt the arrival of a bar in a NEW slot
                        # completes the previous slot -> exclude the new slot's
                        # minutes, or a partial chart bar emits phantom signals
                        if L_min == 1:
                            cut = self.m1["srv"] <= bar["srv"]
                        else:
                            cut = self.m1["srv"] < _slot_start(bar["srv"], L_min)
                        seen = self.m1[cut]
                        ev = eng.step(seen)
                        step_now = int(seen["utc_ms"].iloc[-1])
                        for row in ev["closed"]:
                            events.append(dict(type="close", ms=row["exit_ms"], row=row))
                        for e in ev["emissions"]:
                            events.append(dict(type="emit", ms=e["signal_ms"], em=e,
                                               now=step_now))
                self._sweep_fills()
                self._process_events(events)
                dd = 1.0 - self.acct.mark_eq / self.acct.peak
                if dd > 0.35 and not self.halted:
                    self.halted = True
                    self.journal.log(ev="KILL_SWITCH", dd=dd)
                self._save_state()
            # proof-of-life heartbeat: the system is silent between signals,
            # so emit a status line every minute regardless of activity
            if time.time() - getattr(self, "_last_hb", 0.0) >= 60:
                self._last_hb = time.time()
                self.journal.log(ev="heartbeat",
                                 last_bar=str(self.m1["srv"].iloc[-1]),
                                 model_equity=round(self.acct.mark_eq, 2),
                                 open_trades=len(self.acct.open),
                                 closed_trades=len(self.acct.ledger),
                                 halted=self.halted)
                self._save_state()
            polls += 1
            if max_polls and polls >= max_polls:
                return
            if isinstance(self.gw, gw.ReplayGateway) and self.gw.exhausted():
                self.journal.log(ev="replay_done", equity=self.acct.mark_eq,
                                 trades=len(self.acct.ledger))
                print(f"paper run done: equity {self.acct.mark_eq:,.2f}, "
                      f"{len(self.acct.ledger)} closed trades, "
                      f"{len(self.acct.open)} open")
                return
            time.sleep(poll_seconds if poll_seconds is not None else
                       (0 if isinstance(self.gw, gw.ReplayGateway) else L["poll_seconds"]))


# ===================================================================== status
def status():
    """One-shot health view of the live deployment (process, feed, journal)."""
    import json as _json
    import subprocess
    out = subprocess.run(["pgrep", "-fl", "nas100_rl.live.runner"],
                         capture_output=True, text=True).stdout
    procs = [l for l in out.splitlines() if "--live" in l]
    print(f"runner process  : {'RUNNING  pid ' + procs[0].split()[0] if procs else 'NOT RUNNING'}")

    bdir = Path(L["bridge_dir"]).expanduser()
    bars = bdir / "bars.csv"
    if bars.exists():
        age = time.time() - bars.stat().st_mtime
        with open(bars, "rb") as f:
            f.seek(max(f.seek(0, 2) - 300, 0))
            last = f.read().decode(errors="ignore").strip().splitlines()[-1]
        print(f"EA bar feed     : last write {age:,.0f}s ago | latest bar {last.split(',')[0]} (server)")
        if age > 180:
            print("                  !! feed stale >3min — is MT5 open, EA attached, market open?")
    else:
        print(f"EA bar feed     : bars.csv MISSING in {bdir}")
    acc = bdir / "account.csv"
    if acc.exists():
        bal = acc.read_text().strip().split(",")
        print(f"broker account  : balance {float(bal[0]):,.2f}  equity {float(bal[1]):,.2f}")
    pos = (bdir / "positions.csv")
    ptxt = pos.read_text().strip() if pos.exists() else ""
    print(f"broker positions: {len(ptxt.splitlines()) if ptxt else 0} open"
          + (("\n  " + "\n  ".join(ptxt.splitlines())) if ptxt else ""))

    st_p = config.CACHE_DIR / "live_state.json"
    if st_p.exists():
        try:
            st = _json.loads(st_p.read_text())
            age = time.time() - st_p.stat().st_mtime
            print(f"saved state     : {len(st['acct']['open'])} open model trades, "
                  f"{len(st.get('tickets', []))} tickets"
                  + (", HALTED" if st.get("halted") else "")
                  + f" | written {age:,.0f}s ago")
        except (ValueError, KeyError, OSError):
            print("saved state     : unreadable")

    J = config.CACHE_DIR / L["journal"]
    if not J.exists():
        print("journal         : no journal yet")
        return
    counts: dict = {}
    last_hb = None
    tail_events: list = []
    with open(J) as f:
        for line in f:
            try:
                e = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            counts[e["ev"]] = counts.get(e["ev"], 0) + 1
            if e["ev"] == "heartbeat":
                last_hb = e
            else:
                tail_events.append(e)
    print(f"journal counts  : {counts}")
    if last_hb:
        print(f"last heartbeat  : {last_hb['ts']} | model equity {last_hb['model_equity']:,.2f} | "
              f"open {last_hb['open_trades']} closed {last_hb['closed_trades']} | "
              f"engine saw bar {last_hb['last_bar']}"
              + ("  !! HALTED" if last_hb.get("halted") else ""))
    for e in tail_events[-4:]:
        print(f"  recent: {e}")


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cpmt-every", type=int, default=1)
    ap.add_argument("--paper", type=int, default=0, metavar="DAYS")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--gateway", choices=["file", "mt5"], default="file")
    args = ap.parse_args()

    if args.status:
        status()
        return

    if args.selftest:
        ok = selftest(quick=args.quick, cpmt_every=args.cpmt_every)
        raise SystemExit(0 if ok else 1)

    if args.paper:
        # paper-replay over the VALIDATION tail (never the locked OOS window)
        m1 = data.load_1min()
        m1 = m1[m1["period"] != "oos"].reset_index(drop=True)
        days = m1["srv"].dt.normalize().unique()
        start = days[-args.paper]
        hist = m1[m1["srv"] < start].reset_index(drop=True)
        feed = m1[m1["srv"] >= start].reset_index(drop=True)
        start_ms = int(hist["utc_ms"].iloc[-1]) + 1
        frozen = sigmod.build(basis="live")
        frozen = frozen[frozen["signal_ms"] < start_ms].reset_index(drop=True)
        store = config.CACHE_DIR / "paper_m1.parquet"
        for p in (store, store.with_suffix(".state.json")):
            if p.exists():
                p.unlink()
        r = LiveRunner(gw.ReplayGateway(feed, start=start, bars_per_poll=30),
                       basis="live", m1=hist, frozen=frozen,
                       journal_path=config.CACHE_DIR / "paper_journal.jsonl",
                       store_path=store)
        r.journal.log(ev="paper_start", days=args.paper)
        r.run()
        return

    if args.live:
        # single-instance guard: two live runners = duplicate orders. The lock
        # is held for the process lifetime and released by the OS on any exit.
        try:
            import fcntl
            lock = open(config.CACHE_DIR / "live_runner.lock", "w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                print("REFUSED: another live runner is already running "
                      "(check: python3 -m nas100_rl.live.runner --status)")
                raise SystemExit(1)
            import os
            lock.write(str(os.getpid()))
            lock.flush()
        except ImportError:
            lock = None                      # Windows: no fcntl; rely on operator
        g = (gw.FileBridgeGateway(L["bridge_dir"]) if args.gateway == "file"
             else gw.Mt5Gateway(L["symbol"]))
        runner = LiveRunner(g)
        runner._instance_lock = lock
        runner.run()
        return

    ap.print_help()


if __name__ == "__main__":
    main()
