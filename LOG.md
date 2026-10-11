# Log

Dated entries, newest first. What was learned, what surprised, open questions.

**Current state.** Sessions 00, 01 and 02 are built. Black-Scholes and Heston sit on
the environment, and the cost model is pluggable. **Next:** the Whalley-Wilmott band
(roadmap item 02). Work through 00 and 01 first. File `sessions/02_cost_models.py` is
the cost-model picture, not that research question.

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
map to about 0.55. The network trades on 28 per cent of dates and delta on 93
per cent, and the network's entropic risk is worse than delta's. A fixed charge
does not reward a soft band.

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
fixed cost does not scale with that cube root, and this sample already shows
no band under a fixed charge of 0.05.

`deephedge/export.py` writes schema `deephedge.hedge/v1` for a future widget.
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

Session 01 reproduces their Figure 2: the histogram of hedging P&L, model hedge
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
The 0.05 gap on the indifference price is not the standard error of the mean,
which is about 0.003. Their variance is exact. This one is truncated.

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
