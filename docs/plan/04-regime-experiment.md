# The pre-registered regime-switching experiment on this engine

Source of truth: `experiments/regime/PREREG.md` in the volsurface repository (status
draft, pinned 15 October 2026, not frozen). It states that the hedger and the experiment
runner live in deep-hedging. This document maps each PREREG item to engine nodes; it does
not restate the PREREG and must not drift from it. Where the two disagree the PREREG
wins and this file is updated.

## 1. Graph nodes

Generator node `generator.regime_bergomi`, adapter over
`volsurface.markets.RegimeSwitchingBergomi`:

```
RegimeSpec(generator=[[-1.5, 1.5], [0.8, -0.8]], params=[{"nu": 1.5}, {"nu": 3.0}],
           p0=[0.7, 0.3], switch="nu")
RegimeSwitchingBergomi(spec, xi0=0.04, rho=-0.4, H=0.1, price_filter="rv",
                       n_sub=21, n_mc=1024)
simulate_market(n_paths, times, seed=run_seed(entropy, split), store_forward=True)
```

The adapter maps `MarketPaths` to the engine's `Paths`: `instruments[:, :, 0] = F0 * Z`,
`features` as the feature blocks request, `state = {"variance": v, "regime": regime,
"regime_prob": regime_prob (chosen filter), "regime_prob_spot", "regime_prob_rv",
"regime_prob_latent", "xi", "xi_tenor", "realised_variance"}`. Open question 4 in
PLAN.md asks whether to pass the `MarketPaths` object through instead; the adapter can do
either.

Day count `1/252`. Episode start `t0 = 0.5` (index 126); paths are drawn from time 0 so
the filter has burn-in. The engine's `Paths.time` is the full grid; the contract and the
evaluator receive an `episode` window `(start_index, end_index)`.

Seeds: `run_seed(entropy, split)` from `volsurface.markets`, split 0 train, 1 validation,
2 test. The engine's own seed fields are set to these values and the validator records
that the seed source is "prereg".

Contract nodes:

- `L1`: `contract.vanilla(kind="call", strike="atm_at_start")`, life 21 days from
  `t0`. Payoff `max(Z_T / Z_entry - 1, 0)` on a unit forward; the engine prices in
  spot units and divides by `Z_entry` in the evaluator.
- `L2`: `contract.forward_start_straddle(start=15, life=30)`.

Instrument-set nodes:

- `I1`: `["spot"]`.
- `I2`: `["spot", "variance_swap_21d"]`; the mark is `VarianceSwapPricer` (remaining
  integral, left Riemann on `xi_tenor`) plus the left-point realised leg minus strike
  times elapsed period, as the PREREG specifies. The adapter's martingale pre-gate test
  (`hybrid_tail_mark_bias` under 1 per cent of the quote, untrimmed Monte Carlo
  regression interval containing 0) is a test in M12.
- `I3`: `["spot", "atm_call_21d"]` from `SurrogatePricer(predict, validation_error_bp,
  bar_bp)`; the node is disabled unless `supports` returns true, and the PREREG says it
  is untested until the surrogate exists.

Cost node: `cost.proportional(rate=[1e-4, 0.0, 0.0])`, spot only, entry and unwind
charged; variance swap and calls at model mid.

Feature nodes, one per policy:

| Policy | Features |
|---|---|
| `pi_L`, `pi_H`, `pi_mix` | time, log moneyness, prev position |
| `pi_filt` | plus `regime_prob` from the windowed realised-variance filter (`"rv"`) |
| `pi_oracle` | plus one-hot of `state["regime"]` (a bound, not deployable) |

`pi_L` trains on a generator node with `nu = (1.5, 1.5)` (the low regime only), `pi_H`
on `nu = (3.0, 3.0)`, `pi_mix` on the switching generator; all are the same hedger node
with different training generators, which is exactly the cross-environment matrix of M10.

Hedger node: `hedger.feature_mlp` (width and depth fixed in the experiment spec, chosen on
validation in the pilot and then frozen); `hedger.gru` is an exploratory extra, not part
of the gate.

Benchmarks: `hedger.delta` with Black-Scholes delta at `sqrt(v_t)` (and at flat
`sqrt(xi0) = 0.2`); the rough Bergomi model delta via the pricer only where the pilot can
afford it.

Objective node: `objective.entropic(lam=0.1)` on `X / premium`, with the premium from
65,536 Monte Carlo paths at `run_seed(1, 0)`, held fixed and its standard error reported.
Tail-weight rule from the PREREG (top 1 per cent of weights under 10 per cent of the
total) is an evaluator check; if it fails, halve `lam` once and freeze.

Evaluator node: `evaluator.gap` computing `Gap(X -> Y) = rho(pi_X on Y) - rho(pi_Y on Y)`
with the hierarchical paired percentile bootstrap: draw network seeds, then paths within
seeds, same draw for every policy, `B = 10,000`; one-sided lower bound at level
`1 - 0.05/3` for the family {H1, H2, H3}; BCa as a sensitivity report; CVaR95, mean,
standard deviation, turnover and cost as reports.

## 2. The hypotheses as evaluator contrasts

| Hypothesis | Contrast | Bar |
|---|---|---|
| H1 transfer cost | `Gap(L -> H)` for L1 with I1 | family lower bound above 0 |
| H2 vol-of-vol content | `Gap(L2 -> H) / premium(L2) - Gap(L1 -> H) / premium(L1)` | family lower bound above 0 |
| H3 hedgeability | `Gap` with I1 minus `Gap` with I2, for L1, `price_filter = "rv"` | family lower bound above 0 |
| H4 filters | Brier skill, hit rate, calibration of the three Hamilton filters | exploratory, no bar |
| H5 DYSANOS link | the `scale` argument of `simulate_factors`, off by default | exploratory, no bar |

Controls, each its own spec file in `experiments/regime/specs/`:

- Negative: `nu_H = nu_L = 1.5`; every gap interval must cover 0.
- Positive same-regime: interval covers 0 and `|gap| <= 0.10` of that regime's risk.
- Positive wrong-correlation: `rho` sign flipped between train and test; lower bound of
  the gap above 0.10 of the high-nu risk.
- Sensitivities (not gates): chains `Q5 = 2.3 Q` and `Q0.5 = 23 Q`; ratio grid `nu_H in
  {1.5, 2.25, 3}`.

## 3. Sample sizes and compute

Train 100,000 paths, validation 20,000, test 20,000; 5 network seeds, moving to 10 if the
seed-level variance of mean contrasts dominates (rule in the PREREG). Path counts never
change after the pilot.

Each path has 126 burn-in plus 21 (L1) or 45 (L2) daily steps with 21 intraday
sub-returns for the realised-variance filter, at `H = 0.1` through roughvol's hybrid
scheme. On the modest server:

- Simulation of 140k paths is minutes and is done once per `(spec_id, split)`, cached
  under `runs/paths/` as `.npy` with the seed in the filename.
- Training five seeds of a small MLP on 100k paths for the chosen epochs is the long
  pole, hours in total; the runner is resumable per `(policy, seed)`.
- Nested Monte Carlo for I2 and I3 marks along 20k test paths is expensive; the adapter
  computes the variance-swap mark analytically from `xi` (no nested MC) and only the I3
  surrogate needs a model.
- Bootstrap with `B = 10,000` over 20k paths and 5 seeds is seconds with vectorised
  resampling.

CI runs the whole pipeline at 1k paths, 1 seed, `B = 200`, as a smoke test that must
finish in under three minutes; it reports but does not gate.

## 4. Outputs

`experiments/regime/out/<spec_id>/`: the resolved specs, the PREREG commit hash, premiums
with standard errors, per-policy per-seed P&L arrays, bootstrap replicates, the three
family contrasts with their lower bounds, the control checks, and `REPORT.md` written in
the PREREG's vocabulary (met, not met, control failed). The gate verdict is reported
against the bar exactly as written; the report does not add specifications after seeing
the data. Exploratory items (H4, H5, the ratio grid, GRU) go in a separate section with
effect sizes and the next specification.

## 5. What is missing today and which module supplies it

| Need | Module |
|---|---|
| `Paths` with named state, episode window | M02 |
| forward-start straddle contract | M03 |
| policy with regime-probability or one-hot features | M05, M06 |
| entropic on P&L per unit premium, tail-weight rule | M07 |
| regime generator adapter, variance-swap instrument, martingale pre-gate | M12 |
| cross-environment matrix, hierarchical bootstrap, gap contrasts | M10 |
| the runner, specs, CI smoke test, report | M18 |
