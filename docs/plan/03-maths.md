# Key maths and methods, with assumptions

Plan only. Each section names what the engine computes and what it assumes. Citations:
Buehler, Gonon, Teichmann and Wood (2019), *Deep hedging*, Quantitative Finance 19(8);
Horvath, Teichmann and Zuric (2021), *Deep hedging under rough volatility*, Risks 9(7);
Whalley and Wilmott (1997), *An asymptotic analysis of an optimal hedging model for option
pricing with transaction costs*, Mathematical Finance 7(3); Rockafellar and Uryasev
(2000), *Optimization of conditional value-at-risk*, Journal of Risk 2; Carr and Madan
(1998), *Towards a theory of volatility trading*; Hagan, Kumar, Lesniewski and Woodward
(2002), *Managing smile risk*; Bayer, Friz and Gatheral (2016), *Pricing under rough
volatility*; Bennedsen, Lunde and Pakkanen (2017), *Hybrid scheme for Brownian
semistationary processes*.

## 1. The hedging problem

Discrete dates `0 = t_0 < ... < t_n = T`. Tradable instruments with time-0-discounted
prices `S_k in R^d`. A liability `Z` (the contract's payoff, discounted to time 0). A
policy chooses `delta_k in R^d` at `t_k` from an information set `F_k` and holds it over
`[t_k, t_{k+1}]`. The hedged P&L for a premium `p` is

```
X(delta) = p - Z + sum_{k=0}^{n-1} delta_k . (S_{k+1} - S_k) - C(delta, S)
```

with `delta_{-1} = delta_n = 0` so the opening and liquidating trades are charged. The
deep hedging objective is

```
minimise  rho( X(delta) )  over delta_k = pi_theta(F_k)
```

for a convex risk measure `rho`. Assumptions: self-financing, no funding or borrowing
cost beyond the discount factor, no dividends unless the generator models them,
liquidation at `T` at the simulated price. These match PR 1 and session 00.

## 2. Convex risk measures as optimised certainty equivalents

For a loss function `l` and `rho(X) = inf_w { w + E[ l(-X - w) ] }` (OCE form):

- Entropic, `l(x) = (exp(lam x) - 1) / lam`: `rho(X) = (1/lam) log E[exp(-lam X)]`,
  computed with logsumexp. Time consistent; optimal hedge is Markov in `(t, S, delta)`
  for Markov generators, which is why it is the default and why the band of session 00 is
  a clean object. Standard error by the delta method on the log of a sample mean.
- CVaR at level `alpha`, `l(x) = max(x, 0) / (1 - alpha)`: `rho(X) = inf_w { w +
  E[max(-X - w, 0)] / (1 - alpha) }`. In training `w` is a trainable scalar (Rockafellar
  and Uryasev); in evaluation it is the empirical quantile. The PR 1 `topk` form is kept
  as the evaluation estimator. Standard error from the tail sample.
- Mean-variance, `rho(X) = -E[X] + (lam/2) Var[X]`: not monotone, reported with that
  caveat; it is in the lab because people ask for it.
- Utility indifference: for exponential utility the indifference price equals
  `p* = rho_ent(-Z + gains - costs)` at the optimum. The engine reports
  `p* = rho(X(delta*) - p) + p` for any `rho`, that is the cash that makes the hedged
  position acceptable, and labels it "indifference price" only for the entropic case.

Every risk number carries a standard error and, where the evaluator has bootstrap
replicates, a percentile interval. The evaluator never prints a risk without one.

## 3. Cost handling

`C(delta, S) = sum_k c_k(delta_k - delta_{k-1}, S_k)` with:

- Proportional: `c = rate_i |dq_i| basis_i`, basis the discounted price or a per-instrument
  cash notional (PR 2's `unit`). Exact in training.
- Fixed plus proportional: adds `fixed_i 1{dq_i != 0}` in discounted cash. Training uses
  `tanh(|dq| / width)` with `width` a spec field; evaluation is exact. The width must be
  recorded; PR 2 showed `1e-2` saturates on daily moves.
- Quadratic impact: `c = impact_i dq_i^2 basis_i`. Smooth.

Correction to PR 2: a fixed cash charge at `t_k` must be multiplied by `discount[k]`
when `r > 0`; the interface now passes `discount` to `amount`.

## 4. Pricing of hedging instruments per generator

The engine needs, per generator and contract, a premium `p` and along-path prices and
greeks for the classical hedgers. Learned hedgers need only the premium (or none, if the
indifference price is the output).

| Generator | Vanilla | Other European | Path dependent | Greeks |
|---|---|---|---|---|
| Black-Scholes | closed form | digital closed form; barrier closed form for continuous monitoring, grid correction (Broadie, Glasserman, Kou) for discrete | Asian geometric closed form; arithmetic, lookback by Monte Carlo | analytic |
| Heston | COS and quadrature (PR 1) | COS with the contract's cosine coefficients | Monte Carlo | COS delta, vega (PR 1) |
| SABR | Hagan implied vol into Black, checked by Monte Carlo | same | Monte Carlo | bump on Hagan |
| Local vol | Monte Carlo on the Dupire grid (volsurface `LocalVol.price`) | same | Monte Carlo | bump |
| Rough Bergomi | conditional Monte Carlo given the vol path (`roughvol.RoughBergomi.price_with_error`) | same | nested Monte Carlo | bump on the conditional price; no delta function exists in roughvol |
| Regime Bergomi | `volsurface.markets` pricers: `ClosedFormPricer`, `VarianceSwapPricer`, `SurrogatePricer`, `surface_at` | `surface_at` strike grid | nested Monte Carlo | from the surface batch |
| SPX surface | the fitted SANOS or SVI surface at `t = 0`; along paths via the local vol generator | same | Monte Carlo | from the surface |
| Uploads | Black-Scholes carrier at the sample's realised vol (PR 4), labelled as such | same | Monte Carlo | carrier |

Monte Carlo prices carry a standard error; the evaluator propagates it into the premium
and reports it beside the risk. Nested Monte Carlo along 20k paths is the one thing that
will not run in minutes; the engine caches `(spec_id, split)` results on disk and the
validator warns.

## 5. How exotic payoffs plug in

A contract is a functional of `Paths`, not of the terminal spot. The torch and numpy
payoffs must agree to 1e-10. Discrete monitoring on the hedging grid is the default and
the contract declares `Monitoring(dates="grid" | explicit indices)`; the validator refuses
a barrier observed on dates the grid does not contain. Continuous-monitoring prices are
provided only as reference numbers with the discrete correction noted.

Variance swaps as contracts read `state["realised_variance"]` if the generator provides
it (regime Bergomi does, with intraday sub-steps) and otherwise sum squared log returns
on the grid. Forward-start contracts (the regime experiment's L2) read the spot at the
start index to fix the strike. Baskets need a generator with `d > 1` spot-like
instruments and are out of scope for every current generator except a correlated
Black-Scholes added for the purpose.

Payoff gradients: `payoff_torch` must be differentiable almost everywhere in the
positions' inputs only through the P&L gains term; the payoff itself does not depend on
the policy, so non-smooth payoffs (digital, barrier) do not hurt training. They hurt
hedging, which is the point of putting them in the lab.

## 6. Classical hedgers per generator

- Delta: `delta_k = dP/dS` from the pricer's greeks at `(t_k, state_k)`. Where only a
  Black-Scholes carrier exists (uploads) or the pricer is Monte Carlo, the lab offers
  "delta at `sqrt(v_t)`" (Black-Scholes delta at the current instantaneous vol) as the
  cheap benchmark, which is what the regime PREREG uses.
- Delta-gamma and delta-vega: with instruments `(S, V_1, ...)` solve the small linear
  system `J delta = g` where `J` holds the instruments' greeks and `g` the contract's,
  per path and date. Needs a pricer that returns greeks for the derived instruments.
  Heston's variance swap gives delta-vega as in PR 1 (`vega / (dP/dv)`).
- Whalley-Wilmott: half-width `(3/2 * rate * S * e^{-r tau} * Gamma^2 / lam)^{1/3}` around
  delta, trading to the nearer edge. Assumes small proportional cost, exponential utility
  with absolute risk aversion `lam`, and Black-Scholes dynamics with `mu = r`. On other
  generators the lab computes it from that generator's gamma and labels it "WW formula
  with model gamma". PR 3's fitted exponent (0.293 at 120 steps, 0.063 at 30 steps) is
  the regression test.
- Constant band: `clip(prev, delta - w, delta + w)` with `w` chosen on the validation
  split only.
- Static replication: for a European payoff `g(S_T)` and a strike grid `K_j`, weights
  from the Carr-Madan decomposition `g(S) = g(F) + g'(F)(S - F) + sum g''(K) (K - S)^+ dK
  + sum g''(K) (S - K)^+ dK` discretised on the grid; positions are set at `t_0` and
  held. Needs a listed-call instrument set. Its P&L is premium minus payoff minus the
  replication's own cost, which exposes basis risk between the grid and the payoff.

## 7. Features and memory

Markov generators (Black-Scholes, Heston, SABR, local vol) need only `(t, S, state, prev)`.
Rough and regime generators have memory. The lab offers windowed realised volatility,
the Hamilton filter probability from volsurface, truncated signatures of
`(t, log S, log v)` over a window (level 2 and 3 by hand, or via a library if the owner
allows one), and recurrent policies (GRU from PR 4). The roadmap's session 03 question,
whether a feature-based network loses to a recurrent one as the Hurst exponent falls, is
a direct comparison of two hedger nodes on one generator node.

## 8. Honest evaluation

- Three seeds per run (train, validation, test), distinct, in the spec.
- Hyperparameters and band widths tuned on validation only.
- Paired comparisons: every hedger is scored on the same test paths; differences are
  reported with paired bootstrap intervals (B in the spec, default 2,000; the regime
  experiment uses 10,000 and a hierarchical scheme over network seeds).
- Cross-environment: a `(train generator, test generator)` matrix, with the premium
  charged being the training generator's premium unless the spec says otherwise, and both
  premiums printed.
- Reproducibility: `spec_id`, package versions, `roughvol` and `volsurface` commits, torch
  thread count (training on four threads is not bit-stable, PR 4) all go into the run
  record. The acceptance test is "same spec, same seeds, same printed digits".
