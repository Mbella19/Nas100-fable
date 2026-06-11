"""CPMT v12 pattern engine — direct port of the Pine v6 engine() / detectPat() /
updStatus() / addPivot() / newPat() functions. One Engine instance per
(timeframe, pivot-length) pair, stepped once per confirmed HTF bar.

All math, priority ordering, supersession, expiry (incl. apex truncation for
two-line patterns), invalidation and the first-up/first-dn event capture are
replicated 1:1. Indices are the engine's own bar indices (0-based positions in
its HTF series), mirroring Pine's bar_index.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ST_FORMING = 0
ST_AWAIT = 1
ST_REACHED = 2
ST_FAILED = 3

NAN = float("nan")


def _isna(x: float) -> bool:
    return x != x


def val_at(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
    return y2 if x2 == x1 else y1 + (y2 - y1) * (x - x1) / (x2 - x1)


@dataclass
class Pat:
    name: str
    dir: int
    two: bool
    xs: list[int]
    ys: list[float]
    nx1: int
    ny1: float
    nx2: int
    ny2: float
    mx1: int
    my1: float
    mx2: int
    my2: float
    tgt_size: float
    inv_level: float
    expiry_bar: int
    status: int = ST_FORMING
    target: float = NAN


@dataclass
class BreakEvent:
    """First up-break / dn-break info of one engine bar: price line, invalidation
    level, pattern width (HTF bars), pattern name."""
    up_px: float = NAN
    up_inv: float = NAN
    up_w: float = NAN
    up_name: str = ""
    dn_px: float = NAN
    dn_inv: float = NAN
    dn_w: float = NAN
    dn_name: str = ""


class Engine:
    def __init__(self, tol: float, max_width: int, max_pat: int, use: dict[str, bool] | None = None):
        self.tol = tol
        self.max_width = max_width
        self.max_pat = max_pat
        self.use = use or {}
        self.zzP: list[float] = []
        self.zzB: list[int] = []
        self.zzD: list[int] = []
        self.pats: list[Pat] = []

    # ---------------------------------------------------------------- zigzag
    def _add_pivot(self, d: int, price: float, b: int) -> bool:
        changed = False
        if not self.zzD:
            self.zzP.append(price); self.zzB.append(b); self.zzD.append(d)
            changed = True
        elif self.zzD[-1] == d:
            if (d > 0 and price > self.zzP[-1]) or (d < 0 and price < self.zzP[-1]):
                self.zzP[-1] = price; self.zzB[-1] = b
                changed = True
        else:
            self.zzP.append(price); self.zzB.append(b); self.zzD.append(d)
            changed = True
        if len(self.zzP) > 60:
            self.zzP.pop(0); self.zzB.pop(0); self.zzD.pop(0)
        return changed

    def _zp(self, i: int) -> float:
        return self.zzP[-1 - i]

    def _zb(self, i: int) -> int:
        return self.zzB[-1 - i]

    def _mk_xs(self, k: int) -> list[int]:
        return [self._zb(k - 1 - i) for i in range(k)]

    def _mk_ys(self, k: int) -> list[float]:
        return [self._zp(k - 1 - i) for i in range(k)]

    # ---------------------------------------------------------------- newPat
    def _new_pat(self, name: str, dir_: int, two: bool, xs: list[int], ys: list[float],
                 nx1: int, ny1: float, nx2: int, ny2: float,
                 mx1: int, my1: float, mx2: int, my2: float,
                 tgt_size: float, inv_lev: float) -> bool:
        w = xs[-1] - xs[0]
        if not (self.max_width == 0 or w <= self.max_width):
            return False
        expiry = xs[-1] + max(20, 2 * w)
        if two:
            s_u = (my2 - my1) / max(1, mx2 - mx1)
            s_l = (ny2 - ny1) / max(1, nx2 - nx1)
            if abs(s_u - s_l) > 1e-10:
                apex_x = ((ny1 - s_l * nx1) - (my1 - s_u * mx1)) / (s_u - s_l)
                if not _isna(apex_x) and apex_x > xs[-1]:
                    expiry = min(expiry, int(apex_x))
        p = Pat(name, dir_, two, xs, ys, nx1, ny1, nx2, ny2, mx1, my1, mx2, my2,
                tgt_size, inv_lev, expiry)
        # a new pattern supersedes overlapping still-forming ones
        self.pats = [q for q in self.pats
                     if not (q.status == ST_FORMING and q.xs[-1] >= xs[0])]
        self.pats.append(p)
        while len(self.pats) > self.max_pat:
            self.pats.pop(0)
        return True

    # ---------------------------------------------------------------- detect
    def _use(self, key: str) -> bool:
        return self.use.get(key, True)

    def _detect(self) -> bool:
        tol = self.tol
        n = len(self.zzP)
        if n < 4:
            return False
        zp, zb = self._zp, self._zb
        p0, p1, p2, p3 = zp(0), zp(1), zp(2), zp(3)
        b0, b1, b2, b3 = zb(0), zb(1), zb(2), zb(3)
        d0 = self.zzD[-1]
        p4 = zp(4) if n >= 5 else NAN
        b4 = zb(4) if n >= 5 else -1
        p5 = zp(5) if n >= 6 else NAN

        # -- Triple Top
        if self._use("TT") and d0 == 1 and n >= 6:
            neck = min(p3, p1)
            hi = max(p0, p2, p4)
            lo = min(p0, p2, p4)
            hgt = hi - neck
            if hgt > 0 and hi - lo <= tol * hgt and lo > neck and p5 < neck:
                self._new_pat("Triple Top", -1, False, self._mk_xs(5), self._mk_ys(5),
                                 b4, neck, b0, neck, -1, NAN, -1, NAN, hgt, hi)
                return True

        # -- Triple Bottom
        if self._use("TB") and d0 == -1 and n >= 6:
            neck = max(p3, p1)
            lo = min(p0, p2, p4)
            hi = max(p0, p2, p4)
            hgt = neck - lo
            if hgt > 0 and hi - lo <= tol * hgt and hi < neck and p5 > neck:
                self._new_pat("Triple Bottom", 1, False, self._mk_xs(5), self._mk_ys(5),
                                 b4, neck, b0, neck, -1, NAN, -1, NAN, hgt, lo)
                return True

        # -- Head and Shoulders
        if self._use("HS") and d0 == 1 and n >= 6:
            head, ls, rs = p2, p4, p0
            neck_min = min(p3, p1)
            hgt = head - neck_min
            if (hgt > 0 and head - max(ls, rs) > tol * hgt and abs(ls - rs) <= 1.5 * tol * hgt
                    and min(ls, rs) > max(p3, p1) and abs(p3 - p1) <= 0.5 * hgt and p5 < neck_min):
                tgt = head - val_at(b3, p3, b1, p1, b2)
                self._new_pat("Head and Shoulders", -1, False, self._mk_xs(5), self._mk_ys(5),
                                 b3, p3, b1, p1, -1, NAN, -1, NAN, tgt, head)
                return True

        # -- Inverted Head and Shoulders
        if self._use("IHS") and d0 == -1 and n >= 6:
            head, ls, rs = p2, p4, p0
            neck_max = max(p3, p1)
            hgt = neck_max - head
            if (hgt > 0 and min(ls, rs) - head > tol * hgt and abs(ls - rs) <= 1.5 * tol * hgt
                    and max(ls, rs) < min(p3, p1) and abs(p3 - p1) <= 0.5 * hgt and p5 > neck_max):
                tgt = val_at(b3, p3, b1, p1, b2) - head
                self._new_pat("Inv. Head and Shoulders", 1, False, self._mk_xs(5), self._mk_ys(5),
                                 b3, p3, b1, p1, -1, NAN, -1, NAN, tgt, head)
                return True

        # -- Rectangle
        if self._use("REC") and n >= 5:
            if d0 == 1:
                hi_max, hi_min = max(p4, p2, p0), min(p4, p2, p0)
                lo_max, lo_min = max(p3, p1), min(p3, p1)
            else:
                hi_max, hi_min = max(p3, p1), min(p3, p1)
                lo_max, lo_min = max(p4, p2, p0), min(p4, p2, p0)
            top = (hi_max + hi_min) / 2
            bot = (lo_max + lo_min) / 2
            hgt = top - bot
            if hgt > 0 and hi_min > lo_max and hi_max - hi_min <= tol * hgt and lo_max - lo_min <= tol * hgt:
                self._new_pat("Rectangle", 0, True, self._mk_xs(5), self._mk_ys(5),
                                 b4, bot, b0, bot, b4, top, b0, top, hgt, NAN)
                return True

        # -- Double Top
        if self._use("DT") and d0 == 1:
            hgt = max(p2, p0) - p1
            if hgt > 0 and abs(p0 - p2) <= tol * hgt and p3 < p1:
                self._new_pat("Double Top", -1, False, self._mk_xs(3), self._mk_ys(3),
                                 b2, p1, b0, p1, -1, NAN, -1, NAN, hgt, max(p2, p0))
                return True

        # -- Double Bottom
        if self._use("DB") and d0 == -1:
            hgt = p1 - min(p2, p0)
            if hgt > 0 and abs(p0 - p2) <= tol * hgt and p3 > p1:
                self._new_pat("Double Bottom", 1, False, self._mk_xs(3), self._mk_ys(3),
                                 b2, p1, b0, p1, -1, NAN, -1, NAN, hgt, min(p2, p0))
                return True

        # -- Bullish Flag
        if self._use("BF") and d0 == -1 and n >= 5:
            pole = p3 - p4
            h_c = ((p3 - p2) + (p1 - p0)) / 2
            w_c = b0 - b3
            s_h = (p1 - p3) / max(1, b1 - b3)
            s_l = (p0 - p2) / max(1, b0 - b2)
            if (pole > 0 and h_c > 0 and pole >= 2.0 * h_c and p1 < p3 and p0 < p2
                    and p3 - min(p0, p2) <= 0.6 * pole
                    and abs(s_h - s_l) * w_c <= 2 * tol * h_c and b3 - b4 <= 2 * w_c):
                self._new_pat("Bullish Flag", 1, True, self._mk_xs(5), self._mk_ys(5),
                                 b2, p2, b0, p0, b3, p3, b1, p1, pole, p4 + 0.3 * pole)
                return True

        # -- Bearish Flag
        if self._use("SF") and d0 == 1 and n >= 5:
            pole = p4 - p3
            h_c = ((p2 - p3) + (p0 - p1)) / 2
            w_c = b0 - b3
            s_h = (p0 - p2) / max(1, b0 - b2)
            s_l = (p1 - p3) / max(1, b1 - b3)
            if (pole > 0 and h_c > 0 and pole >= 2.0 * h_c and p1 > p3 and p0 > p2
                    and max(p0, p2) - p3 <= 0.6 * pole
                    and abs(s_h - s_l) * w_c <= 2 * tol * h_c and b3 - b4 <= 2 * w_c):
                self._new_pat("Bearish Flag", -1, True, self._mk_xs(5), self._mk_ys(5),
                                 b3, p3, b1, p1, b2, p2, b0, p0, pole, p4 - 0.3 * pole)
                return True

        # -- Bullish Pennant
        if self._use("BP") and d0 == -1 and n >= 5:
            pole = p3 - p4
            h_c = p3 - p2
            if pole > 0 and h_c > 0 and pole >= 2.0 * h_c and p1 < p3 and p0 > p2 \
                    and b3 - b4 <= 2 * (b0 - b3):
                self._new_pat("Bullish Pennant", 1, True, self._mk_xs(5), self._mk_ys(5),
                                 b2, p2, b0, p0, b3, p3, b1, p1, pole, p4 + 0.3 * pole)
                return True

        # -- Bearish Pennant
        if self._use("SP") and d0 == 1 and n >= 5:
            pole = p4 - p3
            h_c = p2 - p3
            if pole > 0 and h_c > 0 and pole >= 2.0 * h_c and p1 > p3 and p0 < p2 \
                    and b3 - b4 <= 2 * (b0 - b3):
                self._new_pat("Bearish Pennant", -1, True, self._mk_xs(5), self._mk_ys(5),
                                 b3, p3, b1, p1, b2, p2, b0, p0, pole, p4 - 0.3 * pole)
                return True

        # -- Wedges & Triangle
        if n >= 5 and (self._use("RW") or self._use("FW") or self._use("TRI")):
            if d0 == 1:
                ux1, uy1, ux2, uy2 = b4, p4, b0, p0
                lx1, ly1, lx2, ly2 = b3, p3, b1, p1
            else:
                ux1, uy1, ux2, uy2 = b3, p3, b1, p1
                lx1, ly1, lx2, ly2 = b4, p4, b0, p0
            fit_err = abs(p2 - val_at(b4, p4, b0, p0, b2))
            h_s = val_at(ux1, uy1, ux2, uy2, b4) - val_at(lx1, ly1, lx2, ly2, b4)
            h_e = val_at(ux1, uy1, ux2, uy2, b0) - val_at(lx1, ly1, lx2, ly2, b0)
            w = b0 - b4
            if h_s > 0 and h_e > 0 and h_e <= 0.85 * h_s and fit_err <= tol * h_s:
                s_un = (uy2 - uy1) / max(1, ux2 - ux1) * w / h_s
                s_ln = (ly2 - ly1) / max(1, lx2 - lx1) * w / h_s
                xs_w = self._mk_xs(5)
                ys_w = self._mk_ys(5)
                if self._use("RW") and s_un > 0.2 and s_ln > 0.2:
                    self._new_pat("Rising Wedge", -1, True, xs_w, ys_w,
                                     lx1, ly1, lx2, ly2, ux1, uy1, ux2, uy2, h_s, max(ys_w))
                    return True
                elif self._use("FW") and s_un < -0.2 and s_ln < -0.2:
                    self._new_pat("Falling Wedge", 1, True, xs_w, ys_w,
                                     lx1, ly1, lx2, ly2, ux1, uy1, ux2, uy2, h_s, min(ys_w))
                    return True
                elif self._use("TRI") and s_un <= 0.2 and s_ln >= -0.2:
                    self._new_pat("Triangle", 0, True, xs_w, ys_w,
                                     lx1, ly1, lx2, ly2, ux1, uy1, ux2, uy2, h_s, NAN)
                    return True
        return False

    # ---------------------------------------------------------------- status
    def _upd_status(self, t: int, hi: float, lo: float, cl: float, ev: BreakEvent) -> None:
        for p in self.pats:
            if p.status == ST_FORMING:
                up_l = val_at(p.mx1, p.my1, p.mx2, p.my2, t) if p.two \
                    else val_at(p.nx1, p.ny1, p.nx2, p.ny2, t)
                dn_l = val_at(p.nx1, p.ny1, p.nx2, p.ny2, t)
                fail_now = (not _isna(p.inv_level)) and p.dir != 0 and \
                    (cl > p.inv_level if p.dir < 0 else cl < p.inv_level)
                broke_up = (not fail_now) and p.dir >= 0 and cl > up_l
                broke_dn = (not fail_now) and p.dir <= 0 and cl < dn_l
                did = broke_up or broke_dn
                if did:
                    dir_bk = 1 if broke_up else -1
                    p.dir = dir_bk
                    p.status = ST_AWAIT
                    line_p = up_l if dir_bk > 0 else dn_l
                    p.target = line_p + (p.tgt_size if dir_bk > 0 else -p.tgt_size)
                    if _isna(p.inv_level):
                        p.inv_level = min(p.ys) if dir_bk > 0 else max(p.ys)
                    w = float(p.xs[-1] - p.xs[0])
                    if dir_bk > 0 and _isna(ev.up_px):
                        ev.up_px, ev.up_inv, ev.up_w, ev.up_name = line_p, p.inv_level, w, p.name
                    if dir_bk < 0 and _isna(ev.dn_px):
                        ev.dn_px, ev.dn_inv, ev.dn_w, ev.dn_name = line_p, p.inv_level, w, p.name
                if not did and (fail_now or t > p.expiry_bar):
                    p.status = ST_FAILED
            elif p.status == ST_AWAIT:
                hit = hi >= p.target if p.dir > 0 else lo <= p.target
                if hit:
                    p.status = ST_REACHED
                elif (not _isna(p.inv_level)) and (cl < p.inv_level if p.dir > 0 else cl > p.inv_level):
                    p.status = ST_FAILED

    # ---------------------------------------------------------------- step
    def step(self, t: int, hi: float, lo: float, cl: float,
             ph: float, pl: float, piv_len: int) -> BreakEvent:
        """Process one confirmed HTF bar. ph/pl: pivot values confirmed AT this bar
        (na if none); pivot bar index = t - piv_len."""
        ev = BreakEvent()
        zz_event = False
        has_ph = not _isna(ph)
        has_pl = not _isna(pl)
        if has_ph and has_pl:
            if self.zzD and self.zzD[-1] == 1:
                if self._add_pivot(-1, pl, t - piv_len):
                    zz_event = True
                if self._add_pivot(1, ph, t - piv_len):
                    zz_event = True
            else:
                if self._add_pivot(1, ph, t - piv_len):
                    zz_event = True
                if self._add_pivot(-1, pl, t - piv_len):
                    zz_event = True
        elif has_ph:
            if self._add_pivot(1, ph, t - piv_len):
                zz_event = True
        elif has_pl:
            if self._add_pivot(-1, pl, t - piv_len):
                zz_event = True
        if zz_event:
            self._detect()
        self._upd_status(t, hi, lo, cl, ev)
        return ev
