# DMI (PROJECT 1.8) Verification vs TradingView export

Window: 2020-10-06 14:00 UTC .. 2026-06-10 (TV 1-min history starts 2020-10-06; fixed-$ risk
sizing means no equity re-basing is needed).

## Match summary
- ±2 min entry tolerance: 1176/1392 matched (84.5%); ±15 min: 1268/1392 (91.1%).
- Of matched trades, 81% enter at the EXACT same minute; 90% within 1 minute;
  95.7% exit within 1 minute of the TV exit. Entry-time diff mean +0.0 min (no bias).
- Matched-trade PnL: TV 27,932 vs port 27,406 (-1.9%).
- Exit reasons: TV 1328 SL / 64 TP vs port 1333 SL / 62 TP. No reversals either side.
- Long/short: TV 677/715, port 676/719. Hold-time distribution identical (med 49m, p90 128m).
- Gross profit TV 121,371 vs port 124,758 (+2.8%); gross loss 98,022 vs 98,592 (+0.6%).
- Net PnL: TV 23,349 vs port 26,166 (+12% of net, but net is a small difference of ~120k
  gross flows that agree within ~3%).

## Why residual jitter exists (feed, not logic)
The strategy runs on 1-minute bars and uses tick VOLUME in the entry condition
(volume > SMA14). MT5 tick counts differ from TV tick counts, and the arming trigger
(high >= potential + 1.5*ATR(5)) is knife-edge per minute. A 1-minute shift in arming
changes that day's single trade slightly (one entry per NY day by the entryTaken jank).
Jitter is symmetric (mean 0) and exits re-converge (95.7% same exit minute).

## Verdict
PASS — logic-equivalent. All quirks replicated: one-arming-per-NY-day, shared
potentialEntryPrice between directions, else-if arming precedence, no exit order on the
entry bar, frozen overnight stops after the NY-midnight var reset, rejected same-direction
re-entries overwriting stop/TP state.
