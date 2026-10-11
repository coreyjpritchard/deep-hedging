# Log

Dated entries, newest first. What was learned, what surprised, open questions.

**Current state.** Session 00 is built and ready to play. The market sits behind the
environment interface in `deephedge/env.py` (Black-Scholes only). **Next:** work through
00, then start 01 (reproduce a Buehler et al. 2019 figure in a Heston market).

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

Session 00 numbers did not move. On 20 000 paths with seed 12345 the entropic risk at
50 bp is still 1.05 for the network and 1.54 for delta (means -0.83 and -1.37). The
evaluation goes through `Environment.pnl` and `benchmark_positions`, and it matches the
previous `delta_hedge_pnl` pathwise.

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
