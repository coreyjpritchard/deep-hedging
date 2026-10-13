# Deep hedging from scratch

Eight lessons, picture first. Each one adds a single idea. None of them is
built yet. The last one draws a learned hedge on the live view in
`playground/live.py`, next to delta and the fixed band that view already
steps through. Run that view with `marimo run playground/live.py`.

The apps will import `deephedge`. The exercises are Corey's own functions
in `exercises/`, checked with `pytest --impl=exercises`. A test whose
function is missing is skipped, which is the rule already in
`exercises/README.md`. When a lesson is built, its module name goes into
`MODULES` in `tests/conftest.py`.

## 1. A hedge is gains minus costs

The picture is one Black-Scholes path. Four numbers sit under it: the
premium, the sum of share gains, the sum of costs, and the call payoff.
Their combination is the terminal P/L, and a slider on the proportional
cost changes only the cost number.

The idea is the account the live view already walks. Cash starts as the
premium. Each trade pays the discounted spot times the change in shares,
plus a proportional cost, including the opening trade and the sale at
maturity. Gains are the shares held over each step times the change in the
discounted spot. At maturity the discounted payoff is paid. The result is
`premium + gains - costs - discounted payoff`, which is `terminal_pnl`.

Exercise. `exercises/pnl.py`, function `step_pnl(spot, time, position, payoff, premium, rate, r)`.
`position` has shape `(n,)` and is held over each interval. The function
returns gains, total cost, and terminal P/L, in time-0 money.

Test. `tests/test_learn_pnl.py`. On one Black-Scholes path with a positive
rate and a positive interest rate, the three numbers match `terminal_pnl`
and `ProportionalCost` to an absolute tolerance of `1e-10`. The opening
trade and the liquidation are both charged. This is an identity, so the
tolerance is not a standard error.

## 2. Risk measures, and why the choice matters

The picture is one sample of terminal P/L drawn as a histogram, with three
numbers beside it: variance, CVaR, and entropic risk. Sliders set the CVaR
level and the entropic lambda. A second sample, with the same variance and
a heavier left tail, swaps the ranking when the measure changes.

The idea is that a hedge is chosen by minimising a number attached to the
whole sample, and different numbers care about different parts of it.
Entropic risk and CVaR are monetary: adding a sure cash amount `c` lowers
the risk by `c`. Variance does not move when you add cash, so it cannot
price the premium.

Exercise. `exercises/risk.py`, function `variance_risk(pnl)`, the sample
variance with divisor `n - 1`. `cvar` and `entropic` are the names already
in `deephedge/risk.py`. Implementing those two is the existing risk
exercise. `variance_risk` is the new one.

Test. `tests/test_learn_risk.py`. On a sample of 500 draws, seed 0,
`variance_risk(x + 2)` equals `variance_risk(x)` and `entropic(x + 2, 1)`
equals `entropic(x, 1) - 2`, to `1e-10`. A two-point sample is given in
the test, one hedge with a thin left tail and one with a lump there, same
variance. Entropic risk at lambda 1 ranks them one way. The test states
both values and which hedge each measure prefers.

## 3. One step, solved by hand

The picture is a one-period spot, two outcomes, equal probability. A
slider sets the share holding. The curve is the entropic risk of the two
possible P/Ls, and a mark sits at the minimum. At zero cost the mark is
the replicating hedge. With a proportional cost it moves toward zero.

The idea is that hedging is a small optimisation problem. In one step the
decision is a single number, and the objective can be written on a page
and minimised on a grid before any network exists.

Exercise. `exercises/one_step.py`, function
`best_hedge(s0, s_up, s_down, payoff_up, payoff_down, premium, rate, lam)`.
It minimises entropic risk over a grid of holdings. The grid and the
objective are visible in the function.

Test. `tests/test_learn_one_step.py`. With `rate = 0` and payoffs that a
holding `q*` replicates, the returned holding matches `q*` to `1e-3`,
which is the grid spacing the test fixes. With a positive rate the
absolute holding is smaller than `q*`. Both checks are on numbers written
in the test, not on a Monte Carlo sample.

## 4. A linear policy, in numpy

The picture is the loss at each epoch, and beside it the three weights.
The first epoch prints the gradient, one entry per weight, so the update
is a subtraction you can check by hand.

The idea is a policy `q = w · features` trained by gradient descent with
the gradient written out. Features at a date are `(1, spot / s0 - 1, time / T)`.
There is no torch in this lesson. The loss is entropic risk of `terminal_pnl`
on a batch.

Exercise. `exercises/linear_policy.py`, functions `features`, `policy`,
`loss_and_grad`, and `train_linear`. `loss_and_grad` returns the loss and
the gradient of that loss with respect to the weights. The derivative of
the entropic risk and the derivative through the gains are written as
numpy, not as a finite difference.

Test. `tests/test_learn_linear.py`. On 32 Black-Scholes paths, 4 steps,
seed 0, proportional cost 50 bp, lambda 1, the analytic gradient matches
a central difference of step `1e-6` to an absolute tolerance of `1e-4`.
`train_linear` from zero weights, step size `0.05`, 30 steps, finishes
with a training loss at least 10 percent below the loss at step 0 on that
same batch. The batch size is 32.

## 5. A tiny network, still in numpy

The picture is the share holding as a function of spot, at one date,
before training and after. Delta is on the same axes. With zero cost the
curve moves toward delta. The network is one hidden layer, two units,
`tanh`.

The idea is a forward pass and a backward pass written as matrix
multiplies. The same loss as lesson 4 flows back to every weight. Two
units are enough to see a bend that a linear policy cannot make.

Exercise. `exercises/mlp.py`, functions `forward`, `backward`, and
`train_mlp`. `backward` returns the gradient of the loss with respect to
every weight and bias. Initial weights are an argument, so a test can
fix them.

Test. `tests/test_learn_mlp.py`. On 32 paths, 4 steps, seed 1, the
backward gradient matches a central difference of step `1e-6` to an
absolute tolerance of `1e-4`, one coordinate at a time on a subset of
five weights. After training at zero cost, the mean squared gap between
the network's holding and delta, on a spot grid of 21 points at the
middle date, is smaller than the gap of the linear policy from lesson 4
trained on the same paths. The test prints both gaps.

## 6. Torch, checked against the numpy network

The picture is two loss curves on one set of axes, numpy and torch, from
the same initial weights. They lie on top of each other.

The idea is that autograd is a check on the handwritten backward pass,
not a new model. The torch network has the same two hidden units and the
same `tanh`. Training uses float64 so the comparison is not a float32
rounding story.

Exercise. `exercises/mlp_torch.py`, function `train_mlp_torch`, same
arguments as `train_mlp`. One step of gradient descent uses the same step
size.

Test. `tests/test_learn_torch.py`. From a fixed initial weight vector,
one step on 32 paths, seed 1, the torch weights match the numpy weights
to an absolute tolerance of `1e-8`, and the two losses match to `1e-8`.

## 7. The 2019 setup

The picture is the histogram of terminal P/L for a network trained to
minimise entropic risk, with delta hedging on the same axes. First
Black-Scholes, then Heston, spot only. This is the experiment in Buehler,
Gonon, Teichmann and Wood (2019), reduced to the smallest network that
still has a hidden layer.

The idea is to put lessons 1, 2 and 5 into one loop. Paths come from the
market. Positions come from the network. P/L comes from `terminal_pnl`.
The objective is entropic risk. Costs are proportional. Under the pricing
measure the discounted spot is a martingale, so any predictable hedge has
expected gains equal to zero and the expected P/L equals minus the
expected costs. A mean near zero at zero cost checks the account. It does
not, by itself, mean the hedge has learned delta.

Exercise. `exercises/buehler.py`, function `train_entropic(market, m, n, seed, cost, lam, steps)`.
It returns positions of shape `(m, n)` on a fresh batch drawn with a
different seed, plus the training loss at each step.

Test. `tests/test_learn_buehler.py`. The returned positions have shape
`(m, n)`. `terminal_pnl` on those positions matches a second call that
uses the same positions, to `1e-8`. The training loss at the last step is
below the loss at step 0. At zero cost, `mu = r`, 2000 fresh paths, the
mean P/L is within 3 standard errors of 0. The test states the mean, the
standard error and the path count.

## 8. The learned hedge on the live view

The picture is `playground/live.py` with a third holding on the same
path: the network. A slider moves through saved epochs. At zero cost the
holding settles onto delta. At a positive cost a band of dates appears
where the holding does not change, and the dots that mark rebalances get
rarer as the epochs pass. Delta and the fixed-width band stay on the
axes, so the learned rule can be compared with both.

The idea is that the no-trade band is not an input. It is what the
optimiser does when a trade has a cost and the objective cares about the
tail. `ledger` in `deephedge/live.py` already prices any spot holding, so
this lesson passes the network's positions through that function and does
not write a second account.

Exercise. `exercises/band_epoch.py`, function `epoch_grid(market, cost, lam, epochs, seed)`.
It returns, at each epoch, the holding on a grid of spots at the middle
date, for two previous holdings (flat, and half a share). It also returns
the fraction of dates on which a fixed batch of paths trades.

Test. `tests/test_learn_band.py`. At zero cost the mean squared distance
from the grid to delta is smaller at the last epoch than at epoch 0. At
50 bp the fraction of dates with a trade, on 500 paths and 30 steps, seed
3, is below delta's fraction on those paths. The test reports both
fractions and the two distances. It does not compare the width with
Whalley and Wilmott. That comparison is roadmap item 02, and it stays
open.
