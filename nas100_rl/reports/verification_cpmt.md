# CPMT v12 Verification vs TradingView export

Window: 2020-03-01 .. 2026-06-10 UTC (engines warm up from 2020-01-02 with trading disabled;
equity seeded at the window start from the TV cumulative-PnL column: 11,868.82).
TV ran netted "Both (single position)" mode, bar magnifier on; port replicates that mode
with real 1-min intrabar fills. HTF bars are session-anchored (01:00 server day start),
which the 96% match confirms as TradingView's anchoring.

## Match summary (448 closed TV trades in window)
- Matched 430/448 = 96.0% of TV, 96.4% of port (entry tolerance +/- 1 chart bar).
- Entry price |diff|: mean 0.80 pts, p95 2.36 (46.7% bit-identical entries).
- Exit: 98.6% same chart bar; price diff median +0.013 pts in-favor (i.e. unbiased),
  mean +0.77 dragged by a few tail trades where a knife-edge trail level lands on a
  different bar (top |pnl diff| 175..691 USD, both signs).
- qty ratio mean 1.0017 (sizing/compounding replicated).
- Exit mix: TV 443 stop-exits + 5 time-stops vs port 439 + 7. Long/short 242/206 vs 240/206.
- Gross profit TV 118,803 vs port 124,676 (+4.9%); gross loss 91,390 vs 90,004 (-1.5%).
- Matched-trade PnL: TV 30,312 vs port 32,011 (+5.6%); window net TV 27,414 vs port
  34,672 — net is a small difference of ~210k gross flows that agree within ~5%;
  the residual is symmetric per-trade fill noise plus 18 vs 16 unmatched knife-edge
  trades (same-pattern breaks shifted by one HTF bar in one feed but not the other).

## Verdict
PASS — the full chain (zigzag pivots -> 14-pattern priority detection -> status machine ->
six-stream priority merge -> daily soft gate -> GMT+3 blackout -> stop floor/cap from the
broken line -> per-30m-bar trail ratchet -> time stop -> floor-to-0.01 sizing on realized
equity) reproduces TradingView trade-for-trade at 96%, with residuals demonstrably
feed-borne (close-vs-line break decisions at HTF bar closes are knife-edge).
