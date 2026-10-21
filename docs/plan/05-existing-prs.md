# The five draft PRs: what to do with each

The PRs are a stack: #1 on `main`, #2 on #1, #3 on #2, #4 on #3, #5 on #4. Each diff is
incremental. The recommendation is to merge the first three in order and fold the last
two into modules. The plan only recommends; nothing here has been merged, commented on
or edited.

## PR 1: Shared market environment, Black-Scholes then Heston (2,731 additions)

**Merge after review, rebased on `main`.**

What it gives: `Market` ABC, `Paths`, `Environment`, `terminal_pnl`, `Heston` with COS
and quadrature pricing, `HestonPolicy`, `rollout_positions`, `train_book`,
`HestonSetup`, session 01 (the indifference-price experiment of Buehler et al. (2019) on
a Heston market; a reproduction attempt, see the M00 review below), 49 tests. The
session 00 regression table is reproduced to the printed digits.

Debts it carries, to be paid in M02 and M05 rather than blocking the merge:

- `payoff`, `price_along`, `benchmark_positions`, `premium` are all abstract on
  `Market`, so a pure simulator (rough Bergomi) cannot be a `Market`. M02 splits
  `Generator` from `Pricer` and adds the Monte Carlo fallback.
- Black-Scholes and Heston hardcode a call on instrument 0. M03 moves the payoff out of
  the generator.
- `Environment` has no torch method; training loops rebuild discounted prices by hand
  (`train_heston`, `train_cost_session`, `write_bs_example`). M02 adds `pnl_torch`.
- `rollout_positions` is Markov with only `prev` as state. M05 adds policy state.
- `Heston` is a plain class while `BlackScholes` is a frozen dataclass. M02 makes both
  frozen with `params()`.

Review focus for Opus: the Heston full-truncation scheme with Feller violated
(`2 kappa theta = 0.08 < xi^2 = 4`), the variance-swap mark using pricing `kappa, theta`
when `kappa_sim` differs, and the 0.05 gap to the paper's 1.94 indifference price (PR
body attributes it to exact CIR versus truncation on 30 dates; worth one check at 480
steps before merging).

M00 review (21 October 2026): the code is correct and the session numbers reproduce
(premium plus CVaR 1.992 for the model hedge, 1.998 for the network, 30 steps, 20,000
paths). The gap to 1.94 was checked rather than asserted: simulating on 480 or 1,920 steps
and hedging on the same 30 dates gives 1.934 to 1.936 for the model hedge, so the gap is a
coarse-step simulation effect. On the 30-step grid the variance swap is not a discrete
martingale (drift about 7 per cent of its mark, 6.5 SE), and the network does not beat
the model hedge (1.998 against 1.992; 1.971 against 1.934 on sub-stepped paths). The
session is therefore a partial reproduction, not a reproduction of the paper's figure.
The PR text was corrected to say so before merge. Follow-ups for M02: simulate on a fine
grid and sample the hedging dates; COS greeks at small `v tau` (raise the number of terms
or fall back to quadrature).

## PR 2: Pluggable transaction costs, export schema (2,194 additions)

**Rebase on merged #1 and merge.**

What it gives: `costs.py` with `ProportionalCost`, `FixedPlusProportional`,
`QuadraticCost`, per-instrument `unit`, tanh surrogate for the fixed cost; `export.py`
with `deephedge.hedge/v1`; session 02; 13 more tests.

Debts for M04: the fixed cash charge is not discounted (`r = 0` assumed); `amount_torch`
and `label` are not in the `CostModel` protocol and are found by `hasattr`; the `unit`
field for the variance swap is a notional convention that must be named as such in the
spec. The PR body's own note that a charge per unit of variance "should stay out of the
simulator" is right and M04 keeps it in the cost block. A numeric `unit` is also cash on
the day and undiscounted, like the fixed charge.

M00 review (21 October 2026): the fixed-cost P&L reproduces (mean -1.50 for the network,
-1.55 for delta, fixed charge 0.05), but the PR's trade-frequency figures did not: under
the exact indicator the network changes its position on 97 per cent of slots and delta on
99.9 per cent. The network holds approximately but never exactly, so it makes small
trades that the tanh surrogate treats as nearly free and the exact evaluation charges in
full. The loss against delta is a mismatch between training and evaluation, not evidence
that a fixed charge rewards no band. The PR text was corrected before merge; M05 adds a
policy with an explicit no-trade output and M04 a surrogate-width study.

The export schema is a reasonable per-run P&L artefact. Owner decision D3: it is not
frozen, and M09 may replace it with its own run-store layout.

## PR 3: Whalley-Wilmott band against the learned hedge (1,821 additions)

**Rebase on merged #2 and merge, after the M00 wording fixes.** It is a finished session
with a logged result. The band formula is correct and the headline numbers reproduce
(correlation 0.998 between the network's cloud and the analytic Whalley-Wilmott width
across spot bins at 30 steps and 50 bp; fitted cost exponents 0.293 at 120 steps and 0.063
at 30 steps). The M00 review (21 October 2026) measured the same quantities on the
Whalley-Wilmott hedger and on a constant-width band over the same paths, and the reading
changes:

- The correlation does not establish a gamma-shaped band: a band of constant width scores
  0.97 to 0.998 on the same measure, because any band hedger's cloud follows the movement
  of delta between dates.
- Compared like for like, the network's cloud is 0.50 to 0.64 of the Whalley-Wilmott
  hedger's own cloud. Against the analytic width the ratio runs from 0.37 to 0.63 across
  grids and costs (0.54 at 30 steps and 50 bp), and it mixes in the hedger's own gap to
  its analytic width (its cloud is 0.59 to 0.92 of it).
- The reference exponent on a discrete grid is the Whalley-Wilmott hedger's own cloud
  exponent, 0.227 at 120 steps and 0.041 at 30 steps, not the continuous-time 1/3. At 120
  steps the network (0.293) scales closer to a cube root than that hedger does.
- The risk comparison (the asymptotic band ahead by 0.01 to 0.02 at 5 to 25 bp, the
  network ahead at 200 bp) is near the standard error and was not re-run; at 30 steps and
  50 bp the tuned constant band and the network are both ahead of the asymptotic band.

M08 adds a discrete-grid reference band and M10 a paired bootstrap for these differences.

For M08: `band.py` is numpy, Black-Scholes only and not a `Policy`. M08 wraps it as
`WhalleyWilmott(Policy)` taking gamma from any pricer, and keeps the PR 3 functions as the
reference implementation in `tests/test_band.py`. The twelve `.pt` weight files are
regenerated by M17's script and the originals can be removed from git history later if
size becomes a problem; not a blocker.

`GammaPolicy` (MLP with `n(d1)` as a fourth input) is superseded by a `Greeks` feature
block in M06; keep it only as a test of that block.

## PR 4: Python playground, train in one market and test in another (2,910 additions)

**Do not merge as is; fold into M05, M14, M15.**

Reusable: `upload.py` (`load_uploaded_market`, `validate_market` with the 4 SE martingale
check), `bootstrap.py` (`BlockBootstrap`, `parse_closes`), `templates/merton_jump.py`
(the model template), `GRUPolicy` with `reset_state`, the train-slot / test-slot idea
and `spec_id` hashing, the loss-streaming `on_epoch` callback.

Not reusable: `PlaygroundSpec` (40 flat fields, replaced by the graph spec),
`run_experiment` and `fit_policy` (a second training loop; replaced by the engine),
`hedging_sigma` and the isinstance ladders, `RoughBergomiStub`, and the fact that every
classical hedger uses Black-Scholes delta at one "hedging sigma" even on Heston and
uploads (the PR body says so). The static HTML export is a good idea and M15 keeps it.

The PR's own argument against a JavaScript drag-and-drop board (the author should be able
to read every part) is the argument for option C in PLAN.md.

Decided handling (owner, D2): the PR stays open as a draft until M14 and M15 have
absorbed it; only then close it, with a LOG.md line crediting the upload contract and the
GRU policy.

## PR 5: Live hedging simulation and LEARN.md (1,656 additions, 11 PNGs and a GIF)

**Fold into M16; merge LEARN.md separately as docs.**

Reusable: `ledger` and `Book` (per-step cash, cost, gains, mark-to-model P&L; takes any
position vector), the plotly-in-cell fix for the phone, the "score 2,000 paths and mark
this path on the histogram" idea, the finding that `Heston(xi = 2)` pins the truncated
variance to zero on a daily grid on 44 per cent of dates (useful for M02's Heston tests).

Not reusable: `_greeks` raising `TypeError` for anything but Black-Scholes and Heston,
`clip_positions` (duplicates `band_positions`), the hardcoded two presets, the
band-and-delta-only `LivePath`. M16 rebuilds the view as "step one path with any hedger
from a spec", with `ledger` as the accounting and the engine's rollout producing
positions.

`LEARN.md` is a readable eight-lesson ladder that references `exercises/band_epoch.py`
and `tests/test_learn_band.py`, which are not in the diff. It can merge now as a
docs-only PR if those references are marked as planned; it does not depend on the engine.

Decided handling (owner, D2): #5 stays open as a draft until M16 has absorbed it; only
then close it, with the same LOG.md credit. A small docs PR for `LEARN.md` remains an
option but does not close #5. The PNGs should not land on `main`; M16
regenerates one screenshot by script.

## Merge sequence

1. Review #1 (Opus), rebase on `main`, merge.
2. Rebase #2 on `main`, run the suite, merge.
3. Rebase #3 on `main`, run the suite, merge.
4. Open M01. From here every change is a module PR against the plan.
5. Keep #4 and #5 open as drafts; close each with a LOG.md credit only once M14 to M16
   have absorbed it (D2).
