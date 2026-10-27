# Module cards

Each card is one PR. The specification for it is this card plus
`01-interfaces.md`, the conventions in `CLAUDE.md` and the relevant section of `03-maths.md`. Every PR:
`ruff check` clean, `pytest -m "not slow"` green, a LOG.md entry, no roadmap box ticked,
no change to files outside the card's "touches" list without a note in the PR body.

Sizing: "S" under 600 added lines, "M" 600 to 1,500, "L" 1,500 to 2,500. Nothing above L;
split if it grows.

Order, decided 26 October 2026: M01, then M02, then M11a (rough Bergomi only), ahead of
M03 to M08. SABR is M11b and keeps its place in wave 3. Rough Bergomi is the question the
lab exists to ask, so it comes as soon as there is a generator interface to put it
behind. M02 is built in two PRs: M02a (paths, protocols, generators, pricers) and M02b
(the environment's P&L, torch P&L and ledger as the only P&L functions).

---

## M00 Merge the stack (PRs 1, 2, 3)

- Size: review only.
- Inputs: PRs #1, #2, #3 as they stand.
- Outputs: `main` containing `env.py`, `heston.py`, `costs.py`, `export.py`, `band.py`,
  sessions 01 to 03.
- Tests: the existing suites (97 tests in total by PR 5's count).
- Acceptance: `pytest -m "not slow"` green on `main`; session 00 table unchanged at the
  printed digits; LOG.md has the three session entries.

## M01 Graph spec and registry

- Size: M.
- Touches: new `deephedge/spec.py`, `deephedge/registry.py`, `docs/schema/graph-v1.json`,
  `tests/test_spec.py`.
- Inputs: `01-interfaces.md` section "Graph spec".
- Outputs: `GraphSpec` (frozen dataclass, `from_json`, `to_json`, `spec_id`),
  `Registry` (`register(type_name, cls, param_schema)`, `build(node)`), `validate(spec)
  -> list[Problem]` (empty list means valid), a JSON Schema file for the spec.
- Interfaces: every block class exposes `params()` and a class-level `param_schema`;
  nodes are built by `Registry.build`. No block implementations in this PR; stubs for
  the type names only so the validator can be tested.
- Tests: round trip JSON; `spec_id` stable under key order; validator catches unknown
  type, bad parameter, dimension mismatch, missing seed, coincident seeds, unknown edge;
  a documented example spec (the Heston example in `01-interfaces.md`) validates.
- Acceptance: `validate` runs without importing torch or simulating anything; the
  example spec validates; a spec with `seeds.train == seeds.test` is rejected with a
  message naming both fields.

## M02 Generator and Pricer protocols

- Size: L.
- Touches: `deephedge/env.py` (split), new `deephedge/generators/{base,bs,heston}.py`,
  `deephedge/pricing.py`, `deephedge/paths.py`; `heston.py` moves; `tests/test_env.py`,
  `tests/test_heston.py` adapted; new `tests/test_pricing.py`.
- Inputs: PR 1's `Market`, `Paths`, `Environment`, `terminal_pnl`; `01-interfaces.md`.
- Outputs: `Paths` with `state` and names; `Generator` and `Pricer` protocols;
  `BlackScholes` and `Heston` as frozen generators with `pricer()`;
  `MonteCarloPricer` with nested simulation, bump greeks, caching and standard errors;
  `Environment.pnl` and `Environment.pnl_torch` as the only P&L functions;
  `Environment.ledger` returning the per-step book.
- Interfaces: as in `01-interfaces.md`. `Market` stays as a thin deprecated alias for one
  release so sessions 01 to 03 still import.
- Tests: all PR 1 tests still pass through the aliases; `pnl_torch == pnl` at 1e-10 on
  two instruments and `r = 0.03`; Monte Carlo pricer matches the Black-Scholes closed
  form at 4 SE on 20k inner paths; Heston COS price matches the Monte Carlo pricer at 4
  SE; `ledger` final value equals `pnl` per path; state names validated.
- Acceptance: session 00 table reproduced through `Environment.pnl`; no function in the
  package computes a P&L other than `pnl` and `pnl_torch`.

## M03 Contract blocks

- Size: M.
- Touches: new `deephedge/contracts.py`, `tests/test_contracts.py`.
- Inputs: M02 `Paths`; `03-maths.md` section 5.
- Outputs: `Vanilla`, `Digital`, `Barrier`, `Asian`, `Lookback`, `VarianceSwap`,
  `ForwardStart` (call, put, straddle), `Basket`, `UserPayoff` (restricted expression
  language compiled to numpy and torch), `Monitoring`.
- Interfaces: `payoff(paths)`, `payoff_torch(paths)`, `monitoring()`, `params()`.
- Tests: numpy and torch payoffs agree at 1e-10 on random paths; Black-Scholes closed
  forms for vanilla, digital, geometric Asian, continuous barrier (with the discrete
  correction documented) at 4 SE on 20k paths; variance swap payoff equals the sum of
  squared log returns; forward-start strike equals the spot at the start index;
  `UserPayoff("max(S_T - 100, 0)")` equals `Vanilla`; the expression language rejects
  names outside its whitelist.
- Acceptance: every contract validates against the grid through `Monitoring`; a
  barrier with monitoring dates off the grid fails `validate` with a clear message.

## M04 Cost blocks hardened

- Size: S.
- Touches: `deephedge/costs.py`, `tests/test_costs.py`.
- Inputs: PR 2's three cost classes.
- Outputs: `CostModel` protocol with `amount`, `amount_torch(exact)`, `label`,
  `params`; `discount` argument so fixed costs are discounted; surrogate `width` in
  `params`; per-instrument rates as tuples; `register` entries `cost.proportional`,
  `cost.fixed_plus_proportional`, `cost.quadratic`.
- Tests: PR 2 tests kept; fixed cost at `r = 0.05` equals `fixed * discount[k]` per
  trade; surrogate gradient non-zero at a daily delta move for the recorded width;
  `params()` round-trips through the registry.
- Acceptance: no `hasattr` dispatch left in `hedging.py` or `env.py`.

## M05 Policy protocol with state, networks, one rollout

- Size: M.
- Touches: `deephedge/policy.py` (rewrite), new `deephedge/networks.py`,
  `tests/test_policy.py`.
- Inputs: PR 1 `rollout_positions`, PR 4 `GRUPolicy`, `01-interfaces.md` "Policy".
- Outputs: `Policy` base with `initial_state`, `forward(t, x, prev, state)`; `rollout`
  (single); `FeatureMLP(n_in, n_out, width, depth, activation)`, `GRUPolicy`,
  `PerStepMLP`, `OutputClip`; `position_grid` and `held_positions` generalised.
- Tests: rollout shape `(m, n, d)`; stateful policy resets between rollouts (PR 4 test
  kept); `FeatureMLP` with the session 00 features reproduces the session 00 cached
  weights' positions when loaded (a conversion helper for old `.pt` files);
  `PerStepMLP` has `n_steps` parameter sets; `OutputClip` bounds.
- Acceptance: `MLPPolicy`, `HestonPolicy`, `GammaPolicy` removed or reduced to thin
  wrappers over `FeatureMLP` plus a feature block, with the old `.pt` files loadable.

## M06 Feature blocks

- Size: M.
- Touches: new `deephedge/features.py`, `tests/test_features.py`.
- Inputs: M02 `Paths.state`, M05 `Policy`.
- Outputs: `Time`, `Spot`, `LogMoneyness`, `Variance`, `Greeks(which)`,
  `RealisedVol(window)`, `RegimeProb`, `RegimeOneHot`, `PrevPosition`,
  `Signature(depth, window)` (truncated level 2 and level 3 implemented by hand; no
  `iisignature` or `signatory` dependency, D7),
  `Concat`.
- Tests: `out_dim` equals the actual output width; `requires` names checked by the
  validator; `LogMoneyness` with `sqrt_tau` scaling reproduces session 00's feature;
  signature of a straight line matches the closed form; `RealisedVol` on Black-Scholes
  paths has mean `sigma` at 4 SE.
- Acceptance: a `Concat` of blocks can be built from a spec node and fed to a
  `FeatureMLP` whose `n_in` the validator derives.

## M07 Objective blocks

- Size: S.
- Touches: `deephedge/risk.py` (extend), new `deephedge/objectives.py`,
  `tests/test_risk.py` (extend).
- Inputs: PR 1 and PR 3 risk functions; `03-maths.md` section 2.
- Outputs: `Entropic`, `CVaR` (trainable threshold in training, empirical quantile in
  evaluation), `MeanVariance`, `Indifference`; `standard_error` on each; the PREREG
  tail-weight check as `Entropic.tail_weight_share`.
- Tests: closed forms on normal samples at 4 SE (existing); `CVaR.torch` with the
  threshold optimised matches the `topk` estimator at 1e-6 on a fixed sample;
  `Indifference(Entropic)` of a constant P&L equals the constant; tail-weight share on
  a normal sample matches the analytic value.
- Acceptance: `train` accepts any `Objective`; session 00 retrained with `Entropic(1.0)`
  reproduces its objective history to 1e-6 on the same seed.

## M08 Classical hedgers as policies

- Size: M.
- Touches: `deephedge/band.py` (keep as reference), new `deephedge/classical.py`,
  `tests/test_classical.py`.
- Inputs: PR 1 `benchmark_positions`, PR 3 `band.py`, M02 pricers, M06 `Greeks`.
- Outputs: `DeltaHedge`, `DeltaGammaHedge`, `DeltaVegaHedge`, `WhalleyWilmott`,
  `ConstantBand`. `StaticReplication` is not in this module: it is a later-wave item
  (D6), added after the first release as its own small PR with a listed-call instrument
  set.
- Tests: `DeltaHedge` on Black-Scholes equals PR 1's `delta_positions`; on Heston
  equals PR 1's `benchmark_positions`; `WhalleyWilmott` on Black-Scholes equals PR 3's
  `band_positions` at 1e-12; `DeltaVegaHedge` on Heston equals the PR 1 benchmark;
  (later wave, D6: `StaticReplication` of a call by itself has zero P&L at zero cost);
  the PR 3 fitted
  exponent (0.293 at 120 steps, 5 to 25 bp) reproduced as a slow regression test.
- Acceptance: every classical hedger goes through `rollout`; `band.py` is imported only
  by tests.

## M09 Engine: compile, train, run store

- Size: L.
- Touches: new `deephedge/engine.py`, `deephedge/runs.py`, `deephedge/train.py`
  (rewrite around `Objective` and `Environment.pnl_torch`), `tests/test_engine.py`,
  `.gitignore` for `runs/`.
- Inputs: M01 to M08.
- Outputs: `compile(spec) -> Run`; `Run.simulate(split)` with on-disk path cache;
  `Run.train(hedger_name)` with validation-split early stopping, `on_epoch` callback,
  thread count and timing recorded; `Run.evaluate()` producing per-hedger P&L and
  positions on the test split; `RunStore` writing `runs/<spec_id>/` (spec, resolved
  config, package and dependency versions, seeds, weights, P&L arrays, `stats.json`,
  `REPORT.md`). The PR 2 `deephedge.hedge/v1` schema is not frozen (D3): M09 may keep
  it or replace it with its own documented run-store layout; a `python -m deephedge.run spec.json` entry point.
- Tests: compile the example spec; a 64-path, 4-step, 1-epoch run end to end; same spec
  and seeds twice gives identical `stats.json`; path cache hit on the second run; a
  spec with a Monte Carlo pricer on 10k paths emits the validator warning.
- Acceptance: session 00 as a spec file reproduces the session 00 table from the engine
  at the printed digits; `train.py` no longer holds `Setup`, `HestonSetup` or
  `COST_SESSION`.

## M10 Evaluator: comparison, bootstrap, cross-environment

- Size: M.
- Touches: new `deephedge/evaluate.py`, `deephedge/report.py`, `tests/test_evaluate.py`.
- Inputs: M09 run outputs.
- Outputs: `compare(pnls, positions, paths, objectives, bootstrap)` with paired
  percentile and hierarchical (seed then path) bootstrap, `gap(train_spec, test_spec)`,
  the cross-environment matrix, turnover and cost summaries, hedge-ratio clouds at a
  chosen date, and a markdown and plotly report.
- Tests: paired bootstrap interval of a known difference covers it at the nominal rate
  on simulated normals (slow, 200 replications); hierarchical bootstrap reduces to the
  paired one with a single seed; `gap` equals the risk difference computed by hand;
  report renders for one and for five hedgers.
- Acceptance: no number in `REPORT.md` without a standard error or interval; the
  cross-environment matrix for Black-Scholes and Heston with delta and a network matches
  PR 4's default picture in direction (network better in sample, worse on Heston).

## M11a Generator: rough Bergomi (pulled forward, straight after M02)

- Size: M.
- Touches: new `deephedge/generators/rbergomi.py`, `pyproject.toml` extra `[rough]`
  pinning `rough-volatility` to the tag `v0.1.0` (commit 65f3f99), tests.
- Inputs, outputs, tests and acceptance: the rough Bergomi half of the card below.
- Benchmark hedge: the Black-Scholes delta at the current volatility, `sqrt(v_t)` from
  the variance state, with the time left to maturity. It is cheap and has no Monte Carlo
  error; the conditional Monte Carlo delta from `price_with_error` is a later option, not
  the benchmark.
- Until M03 to M08 land, the contract is the M02 vanilla call and the hedgers are the
  ones PR 1 has; the session 04 spec waits for M09.

## M11b Generator: SABR

- Size: S.
- Touches: new `deephedge/generators/sabr.py`, tests.
- Inputs, outputs, tests and acceptance: the SABR half of the card below. Wave 3, as
  before.

## M11 Generators: rough Bergomi and SABR (the original card, now M11a and M11b)

- Size: M.
- Touches: new `deephedge/generators/rbergomi.py`, `generators/sabr.py`,
  `pyproject.toml` extra `[rough]` pinning `rough-volatility` (tag `v0.1.0`), tests.
- Inputs: `roughvol.rbergomi.RoughBergomi(H, nu, rho, xi0).simulate(m, times, seed)`
  returning `Z (m, n+1)` and `state["v"]`; optional `state["xi"]` with `xi_tenors`.
- Outputs: `RoughBergomi` generator (spot `forward * Z`, variance state, forward
  variance state when tenors are set, variance-swap derived instrument from
  `roughvol.rbergomi.variance_swap`); pricer via `price_with_error` (conditional Monte
  Carlo) wrapped as a `Pricer` with standard errors; `SABR` generator with Hagan pricer
  and a Monte Carlo check.
- Tests: shapes and seed determinism; spot martingale at 4 SE on 20k paths; `H = 0.5`
  with `nu` matched reduces to Heston-free Black-Scholes-like variance (constant `xi0`
  when `nu = 0`); conditional Monte Carlo price against roughvol's own test values;
  SABR Hagan versus Monte Carlo at 4 SE for a near-the-money strike; adapter test that
  fails loudly if roughvol's signature changes.
- Acceptance: roadmap session 04 can be written as a spec; the lab preset `H = 0.1, nu
  = 1.5, rho = -0.7, xi0 = 0.04` simulates 20k by 30 in under 30 seconds on four cores.
  `H` is an adjustable spec parameter with 0.1 as the default (D5, matching the PREREG).
  The empirical reference, from the rough-volatility repo's own estimates, is H about
  0.148 on the Oxford-Man rv5 realised-variance series (Monte Carlo SE about 0.013) and
  about 0.167 from the SPX implied-volatility skew (0.115 to 0.207 across expiry
  windows); the preset is rougher than both.

## M12 Generator: regime-switching Bergomi (volsurface)

- Size: L.
- Touches: new `deephedge/generators/regime.py`, `generators/volsurface_adapters.py`,
  `pyproject.toml` extra `[regime]` pinning `volsurface`, tests.
- Inputs: `volsurface.markets.RegimeSwitchingBergomi`, `RegimeSpec`, `run_seed`,
  `VarianceSwapPricer`, `ClosedFormPricer`, `SurrogatePricer`, `hybrid_tail_mark_bias`;
  `04-regime-experiment.md`.
- Outputs: `RegimeBergomi` generator exposing spot, variance, `regime`, `regime_prob`
  (all three filters), `xi`, `realised_variance`; episode window support; derived
  instruments `VarianceSwapMark(tenor)` and `SurrogateCall`; `PreregSeeds` helper that
  sets the spec's seeds from `run_seed`.
- Tests: shapes; seed determinism; regime chain stationary distribution at 4 SE;
  variance-swap martingale pre-gate (drift under 1 per cent of quote; regression
  interval covering 0) at the PREREG's seeds `run_seed(40 + r, 2)`; `RegimeProb`
  feature equals `MarketPaths.regime_prob`; `Paths.source` is the unchanged
  `MarketPaths` object with its surface batches (D4); adapter test for API drift.
- Acceptance: a spec with `generator.regime_bergomi`, `L1`, `I1`, `pi_mix` runs on 1k
  paths through the engine in under three minutes.

## M13 Generators: local vol and the SPX surface route

- Size: L.
- Touches: new `deephedge/generators/localvol.py`, `generators/spx.py`, tests.
- Inputs: `volsurface.chains.load_snapshot`, `core.market_data_from_snapshot`,
  `fit.sanos.SanosFitter`, `fit.svi.SVIFitter`, `arbitrage.check.check_surface`,
  `models.localvol.dupire`, `models.localvol.LocalVol`, `fit.rbergomi.calibrate`; the
  committed `SPX_2026-10-14.parquet`.
- Outputs: `LocalVol` generator from a `(t_grid, k_grid, variance)` triple; `SPXSurface`
  generator that fits a snapshot (SANOS by default), checks arbitrage, builds the Dupire
  grid and either a `LocalVol` generator or a calibrated `RoughBergomi` generator;
  pricer from the fitted surface at `t = 0` and Monte Carlo along paths; a
  `surface_report` with the arbitrage check and fit residuals saved in the run store.
- Tests: local vol with flat variance reproduces Black-Scholes prices at 4 SE; the
  snapshot fits with zero arbitrage violations at the default tolerance; the generator's
  vanilla prices at `t = 0` match the surface's `call(T, k)` within the Monte Carlo
  standard error; seed determinism.
- Acceptance: a spec with `generator.spx_surface` and `contract.vanilla` runs end to
  end from the committed snapshot in under five minutes. The committed
  `SPX_2026-10-14.parquet` in volsurface is the lab's default SPX snapshot (D8).

## M14 Generators: uploads (paths CSV, block bootstrap, user module)

- Size: M.
- Touches: new `deephedge/generators/upload.py` (from PR 4 `upload.py`,
  `bootstrap.py`), `lab/templates/merton_jump.py`, tests from PR 4.
- Inputs: PR 4's `load_uploaded_market`, `validate_market`, `BlockBootstrap`,
  `parse_closes`, the Merton template.
- Outputs: `UploadedPaths` (a CSV of paths used directly, split by row), `BlockBootstrap`
  generator, `UserModule` generator (local use only, flagged in the spec and refused by
  the validator when `spec.run.trusted` is false); the template rewritten against
  `Generator` and optional `Pricer`.
- Tests: PR 4's upload and bootstrap tests kept; `UploadedPaths` rejects wrong shapes
  and non-finite values; validator refuses `UserModule` unless trusted.
- Acceptance: the Merton template passes `validate_market` and runs in the engine with
  `DeltaHedge` from its own pricer (not a Black-Scholes carrier).

## M15 Marimo lab app

- Size: L.
- Touches: new `lab/app.py`, `lab/widgets.py` (spec from widgets and back),
  `docs/lab/index.html` (static export by script), README section.
- Inputs: M09, M10; PR 4 `playground/app.py` as the starting layout.
- Outputs: a marimo app with: a generator card, a contract card, an instrument-set card,
  a hedger list (add, remove, reorder; each with its feature blocks and parameters), an
  objective card, a cost card, a run card (paths, steps, epochs, seeds), a spec JSON
  panel (editable both ways), a run button streaming the loss, and the M10 report
  (histograms, risk table with intervals, hedge-ratio cloud, cross-environment matrix
  when two generators are chosen). File drops for CSV and `.py` go to M14.
- Tests: `tests/test_lab.py`: widgets to spec to widgets round trip; the default spec is
  the Heston example; `marimo export html` succeeds in CI.
- Acceptance: the default picture (train on Black-Scholes, test on Heston, delta versus
  network; 2,000 paths, 20 epochs) computes in under two minutes on four cores (D9);
  every number shown comes from
  the run store, not from app code.

## M16 Live one-path view

- Size: M.
- Touches: new `lab/live.py`, `deephedge/ledger.py` (from PR 5 `ledger`, generalised),
  one screenshot regenerated by script, tests.
- Inputs: PR 5 `ledger`, `Book`; M09 rollout; M02 `Environment.ledger`.
- Outputs: a marimo app that takes a spec, draws one path, and steps through dates
  showing each hedger's position, trade, cost, cash and mark-to-model P&L together,
  with "score m paths" marking this path on the histogram; works for any generator,
  contract, instrument set and cost model the engine supports.
- Tests: `ledger` final value equals `Environment.pnl` per path for a two-instrument
  Heston book with a fixed-plus-proportional cost; app smoke test.
- Acceptance: no `isinstance(market, ...)` dispatch; the Heston `xi = 2` floor
  behaviour from PR 5 is shown as a note, not special-cased.

## M17 Sessions on the engine, weights regenerated

- Size: M.
- Touches: `sessions/00` to `03` rewritten to build a spec and call the engine;
  `scripts/regenerate.py`; `data/weights/` pruned to what the static docs need;
  `exercises/` and the `--impl` switch in `tests/conftest.py` re-keyed to the new module
  names (D10); README and ROADMAP updated.
- Outputs: four sessions reproducing their logged tables through the engine; a
  regression test per session at the printed digits; one script that regenerates every
  committed weight and figure.
- Acceptance: `pytest --impl exercises` still runs, skipping functions not yet
  written, against the re-keyed names (D10); each committed session cache regenerates in
  under five minutes (D9); `python scripts/regenerate.py` runs in under 20 minutes on four cores and
  leaves `git status` clean apart from floating-point-identical files.

## M18 Pre-registered regime experiment runner

- Size: L.
- Touches: new `experiments/regime/{run.py,specs/*.json,README.md}`, CI job.
- Inputs: `04-regime-experiment.md`, the PREREG (pinned commit), M10, M12.
- Outputs: spec files for every policy, liability, instrument set and control; a
  resumable runner over `(policy, seed)`; the gap contrasts with the hierarchical
  bootstrap; `REPORT.md` in the PREREG's vocabulary; a CI smoke run at 1k paths.
- Tests: the runner on 1k paths, 1 seed, `B = 200` completes in under three minutes;
  the negative control's gap interval covers 0 at that size (reported, not asserted
  strictly); every contrast's lower bound is computed at level `1 - 0.05/3`; the
  premium's standard error is in the report.
- Acceptance: the full run (100k/20k/20k, 5 seeds) completes on the server and the
  report states each of H1 to H3 as met or not met against the bar as written, with
  H4 and H5 as exploratory sections.

## M19 Paper SPX hedger from daily snapshots (no orders, ever)

- Size: M.
- Touches: new `paper/` package: `snapshot_to_spec.py`, `step.py`, `book.parquet`,
  README.
- Inputs: a daily SPX option snapshot in the volsurface `chains` format (IBKR or CBOE
  delayed; SPX only, never SPY), produced by the existing daily IBKR snapshot job, which
  stays where it is, outside this repository (D8); this module only reads recorded
  snapshots; M13's `SPXSurface`; a trained hedger from the run
  store.
- Outputs: a daily step that fits the surface, builds the day's features, asks the
  hedger for a position, appends it to a book with the day's marks, and writes the
  running P&L and risk; a report comparing the hedger with delta on the same book.
- Tests: a two-day synthetic snapshot pair steps the book and the P&L equals the
  engine's `ledger` on the same prices; the module has no import of any broker order
  API and a test asserts that.
- Acceptance: one month of daily steps replays from recorded snapshots with one command
  and no manual action (scheduling, like the snapshot job, lives outside this repo), and the README says in its first line that nothing here places
  orders.

## M20 Web canvas (deferred)

- Status: deferred by decision D1. Not started before M10 lands; reconsidered
  then only if the marimo lab (M15) feels limiting.

- Size: L.
- Touches: new `web/` (React Flow, TypeScript) and `deephedge/api.py` (FastAPI:
  `POST /validate`, `POST /runs`, `GET /runs/{id}`, `GET /registry`).
- Inputs: `docs/schema/graph-v1.json`, M09 run store, M10 report JSON.
- Outputs: a node canvas that edits the same spec, submits runs to a local job queue,
  and renders the report; uploads of `.py` disabled.
- Tests: schema-driven form generation from `GET /registry`; a spec built on the canvas
  validates identically in Python; a run submitted from the canvas matches the marimo
  run's `stats.json`.
- Acceptance: the marimo app and the web app read and write the same spec files and
  the same run store; the engine has no import from `web/` or `api.py`.
