# Log

Dated entries, newest first. What was learned, what surprised, open questions.

**Current state.** Sessions 00, 01, 02 and 03 are built. Black-Scholes and Heston
sit on the environment, costs are pluggable, and the Whalley-Wilmott band is
compared with the learned hedge on Black-Scholes in `sessions/03_whalley_wilmott.py`.
That file is roadmap item 02. The box stays open until the session has been worked through.
File `sessions/02_cost_models.py` is the cost-model picture, not that question.
**Next:** the same four hedgers under Heston, then under rough Bergomi with `roughvol`.

## 2026-10-26 The Heston COS delta was wrong at zero variance; fixing it barely moves session 01

What was wrong. `cos_call_greeks` put a floor of 0.3 on the half-width of the
truncation interval in log price. Near expiry at zero variance the log price spreads by
4e-4 to 6e-3, so the whole distribution sat inside a few of the 128 cosines and the
delta came out wrong: one day out at spot 99.945 and zero variance it gave 0.38 where the
quadrature gives 0.012. On 1,000 states drawn from the session 01 paths, against a
reference COS (4,096 terms, 40 standard deviations, no floor, itself within 1e-5 of the
quadrature on every state where the quadrature converges), the old delta error had
mean 1.3e-2, 99th percentile 0.18 and maximum 0.35.

The fix. The floor is now 1e-6 and the defaults are 512 terms and 20 standard
deviations: delta error mean 5.8e-5, 99th percentile 5.9e-4, maximum 1.1e-3. Twelve
standard deviations was not enough once the floor went: the zero-variance distribution
has fat tails (vol of vol 2), and a deep in-the-money delta came out 0.998. The
Black-Scholes limit (the variance path with vol of vol set to 0) is not a defensible
substitute at zero variance: with vol of vol 2 the variance does not follow that path,
and the limit gives a delta of 0.078 at spot 99.945 one day out (quadrature 0.012) and
0.724 at 100.046 two days out (quadrature 0.945; the old COS gave 0.599, the new 0.945). Nor is the quadrature: it fails to converge
on about 2.4 per cent of these states (a price below intrinsic, a delta above 1).
States are now priced in chunks of 4,096, because 1,024 terms on 120,000 states at
once ran the box out of memory.

Session 01 rows of `docs/results.csv`, seed 12345, 20,000 paths, 30 steps, 0 bp:

| row | mean | std | CVaR 50% | premium + CVaR |
|---|---|---|---|---|
| model hedge, before | -0.032256 | 0.402246 | 0.300559 | 1.992393 |
| model hedge, after | -0.032902 | 0.402249 | 0.301270 | 1.993104 |
| network, before and after | -0.012194 | 0.440656 | 0.305708 | 1.997542 |
| delta in the spot only, before | -0.121249 | 1.559154 | 1.281836 | 2.973670 |
| delta in the spot only, after | -0.121293 | 1.558941 | 1.281640 | 2.973474 |
| premium, before and after | 1.691834 | | | |

The model hedge got worse by 0.0007, a fifth of the bootstrap standard error of its
CVaR (0.0035), so the network's shortfall against it narrows from 0.0051 to 0.0044. The
network does not use the COS greeks, so its row is unchanged to the digit. Every
two-decimal number in the README and in session 01 at zero cost is unchanged (1.99
against 2.00, standard deviations 0.40, 0.44 and 1.56, CVaR 0.30, 0.31 and 1.28). On the
sub-stepped check the model hedge scores 1.934251 on 480 steps (was 1.934003) and
1.935830 on 1920 steps (was 1.935819); the network is unchanged at 1.970626 and
1.975836. At 100 bp the model hedge moves from mean -2.79 and CVaR 3.50 to -2.80 and
3.52, and session 01 now says so. The fixed-charge float32 row of session 02 was not
touched.

Why so little moves. The delta was wrong where it does not matter. Under full
truncation Euler a state with zero variance has a spot step of exactly `mu dt`, which is
0 here, so the number of shares held over that step earns nothing. The error only
reaches the P&L through states with small positive variance and through the variance
swap leg, whose weight is vega over the swap's slope.

Cost. 512 terms instead of 128: the benchmark positions on the 20,000 session 01 paths
take 160 seconds on the box instead of 42.

## 2026-10-26 M02a: paths, generators, pricers; the order changes; the fixed-charge drift

**Order.** M01, then M02, then M11a (rough Bergomi only), ahead of M03 to M08. SABR
becomes M11b and stays in wave 3. The rough Bergomi benchmark hedge is the
Black-Scholes delta at the current volatility, `sqrt(v_t)`: cheap and free of Monte
Carlo error. `rough-volatility` is tagged `v0.1.0` at 65f3f99, and the `[rough]`
extra will pin that tag. Recorded in `docs/plan/02-modules.md` and `PLAN.md`.

**The fixed-charge drift, explained.** The session 02 fixed-charge network row moves
in the third decimal between machines (entropic 1.786564 here, 1.785535 committed)
with the same code, weights and package versions. It is not threads, not the ATen
CPU path and not oneDNN: it moves only with MKL's single-precision matrix kernels.
Forcing them on this machine gives 1.786564 (AVX-512, the default), 1.786555
(AVX2), 1.784779 (SSE4.2) and 1.785005 (`MKL_CBWR=COMPATIBLE`); the committed
value lies inside that spread. The mechanism is the exact trade indicator on a
float32 rollout. Of 620,000 position slots, 601,552 trade, and 243,841 of those
trades are smaller than 1e-6 in absolute size, dust that still pays the full 0.05.
Between the AVX-512 and the compatible kernels the positions differ by at most
2.1e-5, yet 22,350 indicators flip (484 fewer trades net), which is the 0.001.
`docs/results.csv` is not regenerated: the row is reproducible to two or three
decimals, and the README's 1.79 holds. The fix belongs with the cost and engine
work (M04, M17): score the rollout in float64 or with a documented trade
threshold, and record the machine and BLAS next to each row.

**M02a.** `deephedge/paths.py` holds `Paths`, now with named `state`,
`instrument_names`, `feature_names` and `source`, all validated, and a `to_torch`.
`deephedge/generators/` holds the `Generator` and `Pricer` protocols, the
deprecated `Market` base, `BlackScholes` and `Heston` (moved from `env.py` and
`heston.py`, which re-export them). Both are generators now: a `name`, state names,
`params()` without the strike (that belongs to the contract) which validate against
the M01 schemas, `simulate_from` for restarting under the pricing measure, and
`pricer()`. `deephedge/pricing.py` has the Black-Scholes formulas and
`MonteCarloPricer`: nested simulation from each path's state, central-difference
delta and gamma with common random numbers, a standard error on everything, and a
cache on an explicit key. `deephedge/contracts.py` has a vanilla call and put, the
minimum the pricers need until M03. Heston paths are bit for bit what they were.

What the checks said. On 20,000 inner paths the Monte Carlo premium sits within
1.2 SE of the closed form for Black-Scholes and Heston, calls and puts (seed 7). The
first Heston delta check failed at 4 SE with a 1% spot bump: on the 2019
parameters the price is curved enough that the central difference is biased by
about 4 SE, so the default bump is 0.2%. The second failure was more
interesting: at zero variance the COS delta is wrong. With 128 terms it is 0.966
where the quadrature and the Monte Carlo pricer both give 0.980, 15 days out;
one day out and just below the strike COS gives 0.38 against a quadrature 0.01.
On the 2019 parameters 47% of hedging states have the truncated variance at 0
(Feller is violated). On 1,000 random states from 1,000 paths by 30 dates the mean
absolute gap between the COS and the quadrature delta is 0.013 (0.025 at zero
variance, 0.002 elsewhere), 28% of states are more than 0.01 apart, and the 99th
percentile is 0.18. The quadrature itself warns of slow convergence at a few
states near expiry, so the size of the gap there is uncertain; where I checked
with the Monte Carlo pricer it sided with the quadrature. More COS terms (1,024)
help away from expiry but not near it. The Heston benchmark hedge in session 01
uses that delta, so its numbers carry the error. Fixed on main in the squash of
PR #16 (merge 80b9017) and brought into this branch by merging main: the floor is
1e-6, defaults are 512 terms and 20 standard deviations, and session 01's model
hedge premium plus CVaR moves from 1.992393 to 1.993104. See the entry above.

Left for M02b: `Environment.pnl_torch` and `Environment.ledger`, and moving every
P&L in `hedging`, `train`, `band` and `export` onto them, with session 00 reproduced
through `Environment.pnl`.

## 2026-10-26 The lab starts: M00 checked, M01 graph spec and registry

M00 is the merge of PRs 1 to 3, which landed on 11 October. Checked again on
`main` at 05ab894: 71 tests pass, `ruff check` is clean, and `python -m
deephedge.figures` reproduces every session 00 row of `docs/results.csv` to the
printed six decimals (premium 2.287151; 50 bp delta entropic 1.543334, network
1.051947). LOG.md has the entries for sessions 01, 02 and 03. One row moved on
this machine: the session 02 fixed-charge network, scored with the exact
indicator, gives mean -1.499432 and entropic 1.786564 against the committed
-1.498475 and 1.785535. The README's 1.79 is unchanged. I have not confirmed why; the likely cause is a
handful of trades near zero flipping in or out of the indicator under a
different CPU's float arithmetic; the weights are the committed ones. The card's
"97 tests" was PR 5's count, which includes the two open drafts; `main` has 71.

M01 is `deephedge/spec.py` and `deephedge/registry.py`. A `deephedge.graph/v1`
file parses into a frozen `GraphSpec` and writes back unchanged. Its `spec_id`
is the first 16 hex digits of the SHA-256 of the canonical JSON without the
`id`, with keys and edges sorted, so key order and edge order do not move it.
`validate(spec)` returns a list of problems, empty when valid. It checks node
types, parameter types and ranges, seeds (all four present, and train,
validation and test pairwise distinct), edges, one node per single-use family,
the instrument set against the generator, the state each feature reads, the
feature width against a network's `n_in`, the instrument count against each
hedger's `n_out` and each per-instrument cost, and a pricer for each classical
hedger unless `run.mc_fallback` is on. It imports neither torch nor scipy and
builds nothing; a subprocess test holds it to that.

The blocks are stubs: a type name, a parameter schema and a `describe` method
the validator reads. M02 to M08 and M11 replace them under the same names.
`generator.rough_bergomi` is registered now with `H = 0.1` as the default (D5),
so a rough Bergomi spec can be written and checked before the adapter exists.
It reports no closed-form pricer, so a delta hedger on it needs
`run.mc_fallback`. The Heston example in `docs/plan/01-interfaces.md` validates;
its `id` placeholder is now the real id, `a9a7694034a69b0d`. The contract
monitoring check is left to M03, because the only contract so far is a vanilla
read at maturity. `docs/schema/graph-v1.json` is generated from the registry by
`python -m deephedge.registry`, and a test fails if the two drift apart.

## 2026-10-22 Static figures, one results file, a lock file

`python -m deephedge.figures` now scores every hedger from the cached weights on
seed 12345 and writes four PNGs (`docs/00_no_trade_band.png`, `01_heston_pnl.png`,
`02_cost_models.png`, `03_risk_vs_cost.png`) and `docs/results.csv`. Rerun on
22 October 2026, every README number matched the sessions: 50 bp delta -1.37 and
entropic 1.54, network -0.83 and 1.05; Heston premium plus CVaR 1.99 (model hedge)
and 2.00 (network); fixed charge 1.61 (delta) and 1.79 (network); at 200 bp, 6.83,
4.37, 3.72 and 3.35. The session 00 premium is 2.2872, so it rounds to 2.29, not the
2.28 the README had. Dependencies are locked in `uv.lock` (torch from the CPU index
off macOS), and `make reproduce` runs the sync, the tests and the figures.

## 2026-10-11 The Whalley-Wilmott band is right where the cost is small

The picture is `sessions/03_whalley_wilmott.py`. On Black-Scholes, with a
proportional cost only, the learned cloud tracks the asymptotic width across
the spot (correlation 0.998 at 30 steps and 50 bp), but so does the cloud of a
constant-width band, so that is not evidence of a gamma-shaped band. Cloud to
cloud, the network's is 0.50 to 0.64 of the Whalley-Wilmott hedger's. At 120
steps and 5 to 25 bp its cost exponent is 0.293, above the 0.227 of the
Whalley-Wilmott hedger's own cloud on the same grid. At 30 steps and 25 to
200 bp both are close to flat (0.063 and 0.041). The decision rule stays soft.
A gamma feature does not sharpen it.

The half-width is Whalley and Wilmott (1997), equation (17) and section 4.
A trade of N shares loses epsilon N S, which is `ProportionalCost(epsilon)`.
Their utility is `1 - exp(-gamma x)`. Maximising it is the same problem as
minimising the entropic risk of terminal wealth at aversion gamma. Our
entropic risk is applied to the time-0 P&L. At r = 0 that P&L is the change
in wealth, so their gamma equals our lambda. At r > 0 the two match when
lambda equals gamma times exp(r T). This session has r = 0 and lambda = 1.
When mu = r the centre is the Black-Scholes delta and the half-width, in
shares, is

`(3/2 * epsilon * S * exp(-r (T - t)) * Gamma^2 / gamma) ** (1/3)`.

`epsilon * S * Gamma^2 / gamma` is dimensionless, and so is the cube root.
Zakamouline (2006) writes the same expression. At r = 0 it is the PFHedge
formula. A drift mu different from r would add a Merton term to the centre.
This market has mu = r = 0, so that term is zero and is not applied. Inside
the band the hedger does not trade. Outside it, the hedger trades to the
nearer edge. The engine still charges liquidation for every hedger, which
the continuous-time band was not built to do. The comparison across hedgers
is on that same charge.

Hypotheses, stated before the runs. The analytic width scales as the cube
root of the cost. The learned cloud at S = 100 and t = T/2 should scale close
to that at 120 steps and 5 to 25 bp, and closer to flat than to a cube root
at 30 steps and 25 to 200 bp. The learned centre should sit at or above
delta, with the offset growing in the cost, while the asymptotic centre is
delta. Every band should beat delta on entropic risk. At small cost the
asymptotic band should match or beat a constant width tuned on another seed,
and the network should be close. At large cost the network may win, because
a centre shift is outside the approximation. Mean P&L of a band should be
less negative than delta's. The previous position is already an input, and
time and log moneyness already determine gamma, so adding gamma as a fourth
input should matter only if the network had missed that coordinate.

The cloud is half the 10-to-90 range of holding minus delta, for spots in
[99, 101] at t = T/2. Paths are the session 00 batch: 20 000 draws, seed
12345. The 30-step networks are the session 00 weights. The 120-step networks
use the same budget (width 32, depth 2, 100 epochs, training seed 0). The
fixed half-width is chosen on seed 0, on a grid from 0.02 to 0.40, and then
scored on 12345. It has one tuned parameter. The asymptotic band has none.
The 50 bp row reproduces session 00: network entropic risk 1.052, delta 1.543.

The fitted cloud exponent at 120 steps is 0.293. A 400-draw bootstrap of the
paths gives standard error 0.007, so the gap from 1/3 is 5.4 sampling
standard errors. The ordinary regression standard error on the four designed
costs is 0.028, which is lack of fit rather than sampling error, and on that
scale the gap is 1.4 standard errors. The widths are 0.050, 0.065, 0.071 and
0.081, against a cube root anchored at the 5 bp point of 0.050, 0.063, 0.072
and 0.085. The analytic exponent on the same costs is 0.333 with residual
zero. At 30 steps the cloud exponent is 0.063, bootstrap standard error
0.006, lack-of-fit standard error 0.035. The widths are 0.069, 0.079, 0.080
and 0.079. That is no scaling. A no-trade half-width defined by an absolute
trade below 0.005 shares is a different object: it stays under 0.05 and its
log-log slope is steep, because a steeper soft rule remains inside that
tolerance farther from the centre. It is not the cube-root test.

The analytic 1/3 is the wrong yardstick for a cloud on a discrete grid. The
same cloud measured on the Whalley-Wilmott hedger, which trades to the edge of
the analytic band on the same paths, has exponent 0.227 at 120 steps
(lack-of-fit standard error 0.033) and 0.041 at 30 steps (0.007). So the
network's 120-step cloud scales more like a cube root than the asymptotic
rule's own cloud does, and the flat 30-step cloud is what the asymptotic rule
shows on that grid too. The gaps from 1/3 above measure the grid as well as
the network. These reference values come from a one-off check outside the
session.

Across spot, in bins from 85 to 115, the 30-step cloud at 50 bp correlates
0.998 with the asymptotic width, at 0.54 of its level (median ratio). That
correlation does not show a learned gamma shape: on the same paths the cloud
of the constant-width band, whose half-width does not depend on the spot,
correlates 0.974 to 0.998 with the asymptotic width across the eight rows,
because delta moves by about gamma times the spot move between dates. The
level is not a band's level either. The Whalley-Wilmott hedger's own cloud is
0.59 to 0.92 of its analytic half-width, and the network's cloud is 0.50 to
0.64 of the Whalley-Wilmott hedger's cloud. Against the analytic width the
network's median ratio is 0.37 to 0.63 across the eight rows, not a constant
half.

The slope of the chosen position with respect to the incoming position, at
the centre, runs from 0.60 to 0.85 on the 30-step grid and from 0.66 to 0.89
on the 120-step grid. A sharp band has slope 1 inside and 0 outside. The
centre offset at 5 bp is +0.002 shares. At 200 bp it is +0.105. The
asymptotic centre does not move.

Giving the network `n(d1)` as a fourth input, and retraining at the same
budget, does not change the answer. Entropic risks agree with the base
network to about one standard error on every row. Slopes stay in the same
range. The 120-step cloud exponent moves from 0.293 to 0.306 (bootstrap
standard error 0.008). The spot correlation was already above 0.94. The
previous position was already a feature, and gamma was already a function of
the other two inputs. The miss is the soft rule, not a missing coordinate.
Session 00 had already found that a wider net, more epochs, SiLU and CVaR
did not sharpen it.

On the common batch, entropic risk (lower is better) and mean P&L. Standard
errors are in parentheses. Gamma is the network with the extra input.

120 steps, 5 bp. Fixed half-width 0.04. Asymptotic half-width at the money
0.090. Cloud 0.050.

| hedger | mean P&L | entropic |
|---|---:|---:|
| delta | -0.224 (0.001) | 0.244 (0.002) |
| Whalley-Wilmott | -0.101 (0.003) | 0.166 (0.003) |
| fixed width 0.04 | -0.129 (0.002) | 0.165 (0.002) |
| network | -0.133 (0.002) | 0.175 (0.002) |
| network with gamma | -0.134 (0.002) | 0.176 (0.002) |

120 steps, 25 bp. Fixed half-width 0.10. Asymptotic half-width 0.154.
Cloud 0.081.

| hedger | mean P&L | entropic |
|---|---:|---:|
| delta | -1.120 (0.003) | 1.205 (0.003) |
| Whalley-Wilmott | -0.420 (0.004) | 0.582 (0.004) |
| fixed width 0.10 | -0.452 (0.004) | 0.589 (0.004) |
| network | -0.473 (0.004) | 0.597 (0.004) |
| network with gamma | -0.478 (0.004) | 0.601 (0.004) |

30 steps, 50 bp. Fixed half-width 0.10. Asymptotic half-width 0.194.
Cloud 0.079.

| hedger | mean P&L | entropic |
|---|---:|---:|
| delta | -1.366 (0.004) | 1.543 (0.006) |
| Whalley-Wilmott | -0.740 (0.006) | 1.141 (0.008) |
| fixed width 0.10 | -0.802 (0.005) | 1.055 (0.006) |
| network | -0.828 (0.005) | 1.052 (0.007) |
| network with gamma | -0.825 (0.005) | 1.054 (0.007) |

30 steps, 200 bp. Fixed half-width 0.16. Asymptotic half-width 0.307.
Cloud 0.079.

| hedger | mean P&L | entropic |
|---|---:|---:|
| delta | -5.472 (0.012) | 6.828 (0.032) |
| Whalley-Wilmott | -2.632 (0.015) | 4.365 (0.018) |
| fixed width 0.16 | -2.739 (0.010) | 3.721 (0.016) |
| network | -2.700 (0.007) | 3.350 (0.016) |
| network with gamma | -2.701 (0.008) | 3.364 (0.017) |

Every band beats delta, on entropic risk and on mean P&L. At 5 to 25 bp the
asymptotic band ties the tuned constant width and beats the network by about
0.01 to 0.02, a few standard errors. At 100 and 200 bp the network beats both
the constant band and the asymptotic band. The asymptotic band is too wide
there (0.31 against a tuned constant 0.16 at 200 bp) and it stays on delta
while the learned centre has moved up by 0.10.

The next measurement is the same four books under Heston, with a proportional
cost on the spot and a stated charge on the variance swap. After that, the
same comparison under rough Bergomi, using the `roughvol` simulator from the
rough-volatility repository.

## 2026-10-11 Transaction costs are a model, not a rate

The cost sits on the environment, in `deephedge/costs.py`. Three models:
`ProportionalCost`, `FixedPlusProportional`, and `QuadraticCost`. Numpy `amount`
and torch `amount_torch(..., exact=True)` apply each one the same way. A number
or a tuple is still a proportional rate, which is what sessions 00 and 01 pass.

The variance swap is not requoted. Its mark stays the pricing instrument, so the
gains and the hedge ratio `vega / (dP/dv)` are unchanged. A proportional rate
times that mark (about 0.003) does not charge a swap of a few hundred contracts.
The cost model takes a per-instrument `unit` instead. `None` means the discounted
price. `unit=(None, 1.0)` keeps the spot on its price and charges one currency
unit per variance swap traded. `ProportionalCost(rate=(0.005, 0.002), unit=(None, 1.0))`
is 50 bp on the spot and 0.002 cash per swap.

Liquidation stays charged. The book is flat before the first date and flat at
maturity, and both trades are in `amount`. A constant holding pays a fixed cost
twice and nothing in between. The paper's numerical examples zero the maturity
cost. This engine does not, which is what session 00 already did.

The fixed cost is an indicator, so a training step has no gradient from it.
`amount_torch(..., exact=False)` uses `tanh(|change| / width)`. Evaluation uses
the indicator. The width does not enter the exact charge. The default `1e-2`
is already saturated on a daily delta move of a few hundredths, so a network
that is already trading gets no signal from the fixed cost. Session 02 trains
the fixed cost at width `0.05`. Proportional and quadratic costs have no
surrogate. `amount` does not see the discount curve, so a fixed charge is
time-0 cash. Sessions use `r = 0`, where that is the cash paid on the day.

`sessions/02_cost_models.py` is the picture: market and cost model as controls,
a charge slider, cached networks. It is not roadmap item 02. That item is the
Whalley-Wilmott band, and it should be the next research session.

On 20 000 Black-Scholes paths, seed 12345, premium 2.29, entropic risk at
lambda 1. Standard errors of the means are under 0.01.

| cost | network mean | network entropic | delta mean | delta entropic |
|---|---|---|---|---|
| none | +0.00 | 0.06 | +0.00 | 0.06 |
| proportional 50 bp | -0.83 | 1.05 | -1.37 | 1.54 |
| fixed 0.05 per trade | -1.50 | 1.79 | -1.55 | 1.61 |
| quadratic impact 0.02 | -0.54 | 0.94 | -1.88 | 2.46 |

At S = 100 and t = T/2, moving the incoming position from 0.2 to 0.8 moves the
chosen position by 0.01 of that gap with no cost, by 0.66 at 50 bp, and by 0.57
under the quadratic cost. Under the fixed cost the three incoming positions all
map to about 0.55, and the network's entropic risk is worse than delta's. Under
the exact indicator the network's position changes on 97 per cent of trade slots
and delta's on 99.9 per cent. 15 per cent of the network's changes between open
and liquidation exceed 0.01 shares, against 80 per cent of delta's. The network
never holds exactly still, and the indicator charges the full 0.05 on small
changes that the tanh surrogate charged almost nothing. So this run does not show
that a fixed charge rewards no band; it shows the surrogate and the policy class
are the wrong pair for it. An earlier version of this entry said the network
trades on 28 per cent of dates and delta on 93; those figures were not computed
by the session and do not reproduce.

On 4 000 Heston paths, same seed, premium 1.692, 50 per cent CVaR. The networks
are small (width 32, depth 2, 8 000 training paths, 50 epochs). Charged 50 bp on
each mark, the network's mean absolute swap holding is 232 against 499 for the
model hedge. Mean P&L -0.52 against -1.41, CVaR 1.30 against 1.84. Standard
error of the network mean is 0.017. Charged 0.002 per swap, with 50 bp still on
the spot, the network's mean absolute swap holding is 0.005. Mean P&L -0.31,
CVaR 1.43. The model hedge, which keeps the swap, has mean -7.34 and CVaR 10.57.

The first research session should be the Whalley-Wilmott band on Black-Scholes
with `ProportionalCost` only. The asymptotic half-width is
`(3/2 * c * S * exp(-r(T-t)) * Gamma^2 / gamma)^(1/3)`, and it belongs on the
same axes as the learned band in session 00, across spot, time and the
proportional cost. The fixed charge and the Heston swap are the follow-up: a
fixed cost does not scale with that cube root. The fixed-charge run above shows
no band, but it is confounded by the training surrogate.

`deephedge/export.py` writes schema `deephedge.hedge/v1` for a future widget.
The schema is provisional, not frozen: a later run-store layout may replace it.
The example is `docs/examples/bs_delta_50bp.json`.

## 2026-10-11 Heston on the same interface, session 01

Heston is `deephedge.heston.Heston`. Two instruments, the spot and a strike-0
variance swap. Features are the spot and the variance, and the variance column is
`max(v, 0)`. Feature column 0 is still the spot.

The scheme is full truncation Euler (Lord, Koekkoek and Van Dijk 2010), the same
truncation the sibling `roughvol` package uses. The 2019 parameters violate Feller
(`2 kappa theta = 0.08 < xi^2 = 4`), so the variance hits zero and the latent value
is allowed to go negative while the drift and the diffusion see only the positive
part. One seeded draw, `standard_normal((m, n, 2))`. Column 0 is the variance shock.
Column 1 is orthogonal. The spot shock is `rho` times column 0 plus `sqrt(1 - rho^2)`
times column 1.

Pricing drifts are `r` on the spot and `(kappa, theta)` in the variance. Simulation
drifts are `mu`, `kappa_sim` and `theta_sim`. `None` means the pricing parameter.
The premium does not read the simulation drifts. Defaults are section 5.2 of
Buehler, Gonon, Teichmann and Wood (2019), drawn under the pricing measure.

The swap's time-t price is the discount from `T` of the realised integral plus the
remaining leg `L(t, v^+)`. With `v0 = theta` and `r = 0` that is `theta * T` at
inception, 0.003288. The call premium is the little-trap quadrature of the
characteristic function: 1.6918339240921085, which is the paper's printed 1.69.
Along a path, price, delta and vega use a cosine expansion (Fang and Oosterlee
2008), 128 terms. At inception that price differs from the quadrature by about
5e-5. The benchmark holds delta shares and `vega / (dP/dv)` variance swaps.
`xi = 0` is not taken as a limit of the transform. It is the Black-Scholes call
with total variance `L`.

The torch P&L now sums several instruments. The one-instrument branch is the
Black-Scholes training loss, unchanged. On two instruments, two rates, float64,
it matches `terminal_pnl` to 1e-10, and the payoff is charged once.

Session 01 redraws their Figure 2, qualitatively: the histogram of hedging P&L, model hedge
against the network, both sold at the model price, zero cost. Their figure uses
an exact CIR draw, a simplified Broadie-Kaya spot, a network with separate weights
on each date, and one million paths. This one uses full truncation, one shared
network, and 20 000 paths (seed 12345). The network minimises 50% CVaR. On that
sample:

| book | mean | std | CVaR at 0.5 |
|---|---|---|---|
| model hedge | -0.03 | 0.40 | 0.30 |
| network, trained at 0 bp | -0.01 | 0.44 | 0.31 |
| delta, no variance swap | -0.12 | 1.56 | 1.28 |
| no hedge | -0.11 | 2.27 | 1.70 |

Premium plus CVaR is an indifference price of 1.99 for the model hedge and 2.00
for the network. Their Figure 2 overlaps in the same way, and their indifference
price at alpha 0.5 is 1.94, sold at 1.69. The overlap is what this sample shows.
The network does not beat the model hedge (2.00 against 1.99). The 0.05 gap on
the indifference price is not noise: a bootstrap standard error of each CVaR is
about 0.004.

The gap is the coarse variance step, checked once outside the session. Same seed,
variance simulated on 480 steps, hedge on every 16th date (the same 30 dates): the
model hedge's premium plus CVaR is 1.934, and 1.936 on 1920 steps. The network
trained on 30-step paths scores 1.971 there, about 0.04 behind the model hedge. On
the 30-step grid the variance sits at zero on 45 per cent of hedging dates, and
the variance swap's mark is not a discrete martingale: its mean at maturity exceeds
its inception price by 2.4e-4, about 7 per cent of the mark and 6.5 standard
errors. On 480 steps the drift is within noise; the martingale test runs there.

At 100 bp the model hedge, which keeps the zero-cost weights, has mean -2.79 and
CVaR 3.50. The network trained at 100 bp has mean -0.42, standard deviation 1.77
and CVaR 1.54. It holds less of both instruments.

What this leaves open for the cost model. A proportional rate times the swap's
mark is almost inert: the mark is about 0.003, so a few hundred swaps at 100 bp
cost a few hundredths, while half a share of the spot costs about 0.5 to open.
`CostModel.amount` already receives discounted prices and positions, so a fixed
cost plus a proportional cost, in cash, fits the signature. The paper defines
both, and its numerical examples set the maturity cost to zero. This engine
charges liquidation, which is what session 00 does, and that stays. A charge per
unit of variance, or per unit of the swap position, is not a function of the mark
alone. That needs a notional the current signature does not see, unless the swap
is requoted. It should not be folded into the Heston simulator.

## 2026-10-11 Market environment, Black-Scholes first

The market now sits behind one interface in `deephedge/env.py`. A `Market` simulates
paths of the instruments you can trade and of the features a hedger may observe, and it
exposes the payoff, the price along a path and a benchmark hedge. An `Environment` pairs
that market with a cost model. `terminal_pnl` turns positions in the instruments into a
P&L in time-0 money. Risk measures take that P&L and do not see the market.

Black-Scholes is the first implementation. One instrument, the spot. The feature path is
the spot. The benchmark hedge is the analytic delta and the premium is the analytic call
price. The log-price draw is the existing `bs_paths`, same seeds, same one Gaussian
array of shape `(m, n)`. `ProportionalCost` is the only cost model, the proportional
cost from session 00, and it lives on the environment rather than inside the simulator.

Left open, and not implemented. Heston will use features `(spot, variance)` and may add
a variance swap as a second instrument, with that swap's price in `instruments` and the
variance factor only in `features`. `rollout_positions` already calls a policy with a
feature row of width `f` and expects a position of width `d`, and `d` need not equal
`f`. Rough Bergomi will put volatility memory into the feature path; the simulator
stays the sibling package `roughvol`. Further cost models implement `CostModel.amount`.

Session 00 numbers did not move at the precision of the 2026-10-08 table. On 20 000
paths with seed 12345 the entropic risk at 50 bp is still 1.05 for the network and
1.54 for delta (means -0.83 and -1.37). The whole table rounds to the same printed
digits. The network P&L matches the previous `pnl_from_positions` pathwise. The delta
P&L is lower by 2.8e-14 on every path. The environment sells the call at the analytic
price of S0 = 100. `delta_hedge_pnl` sold at the price of the simulated S_0, and
exp(log 100) is one unit in the last place above 100. That gap is below the reported
digits. The network was already sold at the price of S0 = 100.

## 2026-10-08 Session 00: delta hedging loses money when trading costs money

Built `sessions/00_cost_of_hedging.py`: a short 30-day ATM call (S0 = K = 100,
sigma = 0.2, r = mu = 0), hedged by Black-Scholes delta, by a feature-based network
(inputs tau / T, log(S / K) / sqrt(tau), previous position; two hidden layers of 32,
ReLU) trained to minimise entropic risk at lambda = 1, and not at all. Cached weights for
costs 0, 25, 50, 100 and 200 bp at 30 steps, each trained on 20 000 paths for 100
epochs in about 30 seconds on four cores.

Results on 20 000 fresh paths (seed 12345), 30 steps:

| cost | delta mean | network mean | delta entropic | network entropic |
|---|---|---|---|---|
| 0 bp | +0.002 | +0.003 | 0.063 | 0.062 |
| 25 bp | -0.68 | -0.47 | 0.78 | 0.61 |
| 50 bp | -1.37 | -0.83 | 1.54 | 1.05 |
| 100 bp | -2.74 | -1.47 | 3.18 | 1.86 |
| 200 bp | -5.47 | -2.70 | 6.83 | 3.35 |

No hedge: mean -0.03, entropic 13.3.

What was learned.

- Delta hedging's rebalancing cost grows like sqrt(n): at 100 bp its mean P&L goes
  -1.96, -2.74, -3.47, -4.48 for n = 10, 30, 60, 120. The network trained at n = 30 and
  evaluated at the other n goes -1.26, -1.47, -1.64, -1.89, and its standard deviation
  falls from 1.15 to 0.63.
- Without costs the network reproduces delta: its positions at T/2 lie within 0.01 of
  delta near the money.

What surprised. Leads for session 02, in EXPLORE mode.

- The learned band is soft. Inside it the network still trades part of the way towards
  the middle: at S = 100, T/2, cost 50 bp, moving the incoming position from 0.3 to 0.7
  moves the chosen position by 0.69 of that, not 1. Longer training (250 epochs), ReLU
  against SiLU and a wider net (64 by 3) did not sharpen it, and training with CVaR at
  alpha = 0.9 gave a similar soft band. A sharp constant-width band around delta,
  clip(prev, delta - w, delta + w), ties with the network: best entropic risk 1.055 at
  w = 0.10 against 1.052. Under CVaR_0.9 the CVaR-trained network beat every constant
  band (1.84 against 2.01 at w = 0.08). So a soft band costs little here, and the
  objective is flat across quite different policies. Next specification: the
  Whalley-Wilmott band with its spot- and time-dependent width, same paths.
- The band barely widens with cost. Half-range of positions minus delta for spots in
  [99, 101] at T/2: 0.18, 0.20, 0.21, 0.20 at 25, 50, 100, 200 bp. The Whalley-Wilmott
  half-width with gamma = lambda = 1 is 0.15, 0.19, 0.24, 0.31. Effect: the learned
  width is flat in cost where the theory says cube root. Candidate reasons: the
  asymptotics are for small costs, rehedging is discrete (30 dates), the network's band
  is soft. Next specification: compare at n = 120 and at 5 to 25 bp, where the
  asymptotics should hold best.
- The band sits above delta and moves up with cost: mean of position minus delta near
  the money is +0.02, +0.04, +0.07, +0.09 at 25 to 200 bp. The positions as a function
  of spot are flatter than delta (over-hedged out of the money, under-hedged in it), as
  if the network lags delta. Open question: is the asymmetry a higher-order cost effect,
  the entropic tail aversion of a short call, or the flat start and liquidation at
  maturity?

Reframes logged. Panel (b) was first drawn as a shaded region where |trade| < 0.005 on a
grid of (spot, previous position). With a soft band that region is nearly empty, so the
panel now shows the cloud of positions the network actually holds at T/2. The training
objective moved from CVaR to entropic risk: entropic is time consistent, so the optimal
hedge is Markov in (t, S, position), which is what the network sees, and it is the
exponential-utility criterion of Whalley and Wilmott.
