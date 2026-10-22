# deep-hedging

Learn a hedging policy with a neural network when trading costs money, and compare it
with Black-Scholes delta hedging. The capstone is deep hedging under rough volatility.

![Positions held halfway to expiry: on delta without costs, a band around it with costs](docs/00_no_trade_band.png)

A short 30-day at-the-money call, hedged on 30 dates. Each red dot is one simulated path:
the spot halfway to expiry and the position the network holds there. Without costs the
network learns delta (left). With a 50 bp proportional cost it holds a band around delta
and trades only when its position drifts too far (right). The black curves are its
decision rule for three starting positions.

## What is here so far

| Session | Question | Open it |
|---|---|---|
| 00 | Why does delta hedging lose money when trading costs money, and what does a network do instead? | `sessions/00_cost_of_hedging.py`, static copy in [`docs/00_cost_of_hedging.html`](docs/00_cost_of_hedging.html) |
| 01 | In the 2019 Heston market, does a network match the model hedge once a variance swap is traded? | `sessions/01_heston_deep_hedge.py`, static copy in [`docs/01_heston_deep_hedge.html`](docs/01_heston_deep_hedge.html) |
| 02 | How does the learned hedge change when the cost model changes? | `sessions/02_cost_models.py`, static copy in [`docs/02_cost_models.html`](docs/02_cost_models.html) |

Findings from session 00, on 20 000 fresh paths with a 50 bp cost:

- Delta hedging has mean P&L -1.37 against a premium of 2.28; the network, -0.83. The
  network's entropic risk at lambda = 1 is 1.05 against 1.54 for delta.
- With cost 100 bp, going from 10 to 120 rehedges takes delta hedging's mean P&L from
  -1.96 to -4.48, close to the square-root-of-n law. The network loses far less.
- The learned band is softer than the Whalley-Wilmott band, barely widens between 50
  and 200 bp, and sits above delta. That comparison is the next research session
  on the roadmap. File `sessions/02_cost_models.py` is the cost-model picture, not
  that question.

Findings from session 01, on 20 000 fresh Heston paths at zero cost (seed 12345). The
call is sold at the quadrature price 1.692. The paper prints 1.69 and, for the same
figure, an indifference price of 1.94 at 50% CVaR.

- The model hedge (delta, plus a variance swap hedging vega) has mean P&L -0.03,
  standard deviation 0.40, CVaR 0.30. The network has -0.01, 0.44, 0.31. The two
  histograms overlap. Premium plus CVaR is 1.99 and 2.00, against the paper's 1.94.
  The network does not beat the model hedge. The 0.05 gap is the coarse variance
  step: simulated on 480 steps and hedged on the same 30 dates, the model hedge gives
  1.934.
- Delta with no variance swap has standard deviation 1.56 and CVaR 1.28. The tight
  hedge is the swap.
- At 100 bp the model hedge's mean falls to -2.79. A network trained at that cost
  has mean -0.42 and CVaR 1.54.

Findings from session 02, seed 12345. Black-Scholes uses 20 000 paths and entropic
risk at lambda 1. Heston uses 4 000 paths, 50 per cent CVaR, and a smaller network
than session 01.

- At 50 bp the network's entropic risk is 1.05 against 1.54 for delta. Under a
  quadratic impact of 0.02 it is 0.94 against 2.46. Both show a no-trade band.
- Under a fixed charge of 0.05 per trade this network does not beat delta: entropic
  risk 1.79 against 1.61. It never holds exactly still, so the exact charge falls
  on 97 per cent of its trade slots, while the tanh surrogate it was trained on
  charged those small changes almost nothing. That mismatch, not the fixed cost
  itself, is what the run shows.
- On Heston, 50 bp on the swap's mark leaves the network holding mean absolute
  swap weight 232. Charging 0.002 per contract, it holds 0.005. The model hedge,
  which keeps the swap, has mean P&L -7.34 under that charge.

The plan is in [ROADMAP.md](ROADMAP.md) and dated notes are in [LOG.md](LOG.md).

## Market environments

A hedger does not import a volatility model. It is trained against an environment: a market plus a cost model. The market simulates the instruments you can trade and the features you may observe, and it prices the liability you are hedging. The P&L engine turns a position in those instruments into a terminal P&L. Risk measures see only that P&L.

Black-Scholes is the first market. There is one instrument, the spot, and the feature is the spot. The premium is the analytic call price and the benchmark hedge is the analytic delta.

Heston is the second, in `deephedge.heston`. Features are the spot and the variance. Instruments are the spot and a variance swap, and the swap's price is an instrument while the variance factor stays a feature. Paths are a full truncation Euler scheme. The premium is a quadrature of the characteristic function, and the benchmark hedge is delta in the spot plus a variance-swap hedge of vega. Rough Bergomi is next: its features have to carry the volatility's memory, and the simulator comes from the sibling repository [rough-volatility](https://github.com/coreyjpritchard/rough-volatility).

Costs are not inside the simulator. `ProportionalCost`, `FixedPlusProportional` and `QuadraticCost` live in `deephedge/costs.py` and replace each other on the environment. One batch of paths can be scored at several costs. Every trade is charged, including the open and the liquidation at maturity. A proportional rate times the variance swap's mark barely charges that instrument, because the mark is an integrated variance of order 10^-3. Pass `unit=(None, 1.0)` to keep the spot on its price and charge one currency unit per variance swap. The swap's price is not requoted.

```python
from deephedge.env import BlackScholes, Environment, ProportionalCost
from deephedge.heston import Heston

market = BlackScholes(s0=100, strike=100, maturity=30 / 365, sigma=0.2)
env = Environment(market, ProportionalCost(0.005))
paths = market.simulate(20_000, 30, seed=1)
pnl = env.pnl(paths, market.benchmark_positions(paths))

heston = Heston()  # Buehler et al. (2019), section 5.2, under the pricing measure
h_paths = heston.simulate(20_000, 30, seed=1)
swap_cost = ProportionalCost(rate=(0.005, 0.002), unit=(None, 1.0))
h_pnl = Environment(heston, swap_cost).pnl(h_paths, heston.benchmark_positions(h_paths))
```

## Why

Buehler, Gonon, Teichmann and Wood (2019, *Deep hedging*, Quantitative Finance 19(8))
replace the hedging formula by a network trained end to end on simulated paths to
minimise a convex risk measure of the hedged P&L. Costs, constraints and incomplete
markets then cost nothing extra to model. Horvath, Teichmann and Zuric (2021, *Deep
hedging under rough volatility*, Risks 9(7)) take this to rough Bergomi, where the
volatility has memory and the hedge depends on the path, not only on today's spot. This
repository builds up to that setting one session at a time.

## How to run

Python 3.11 or later; torch runs on CPU.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                       # once
marimo run sessions/00_cost_of_hedging.py     # session 00; marimo edit to see the code
marimo run sessions/01_heston_deep_hedge.py   # session 01, the 2019 Heston figure
marimo run sessions/02_cost_models.py         # session 02, cost models and the swap unit
pytest -m "not slow"                          # tests, a few seconds
python -m deephedge.train                     # retrain the cached weights, about 3 minutes
python -m deephedge.figures                   # redraw the README figure
```

Each app opens with cached weights in `data/weights/`. Retraining takes about 30 seconds
on four cores.

## Layout

- `deephedge/env.py` the market environment: instruments, features, payoff, price,
  benchmark hedge, and a pluggable cost. Black-Scholes is the first market.
- `deephedge/costs.py` proportional, fixed-plus-proportional and quadratic costs.
  A `unit` is the notional the charge multiplies when the mark is the wrong one.
- `deephedge/export.py` schema `deephedge.hedge/v1`, a compact JSON of paths,
  positions and P&L. The example is `docs/examples/bs_delta_50bp.json`. The schema
  is provisional, not frozen.
- `deephedge/heston.py` the Heston market: full truncation Euler, a variance swap,
  and the characteristic-function price and hedge.
- `deephedge/markets.py` the Black-Scholes path simulator the environment draws from.
- `deephedge/hedging.py` P&L of a hedged short option, one instrument or several.
  The Black-Scholes price and delta are re-exported from `deephedge.env`.
- `deephedge/risk.py` entropic risk and CVaR, numpy and torch. They take a P&L sample
  and nothing about the market.
- `deephedge/policy.py` the feature-based network, the Heston network, a delta policy,
  and rollouts.
- `deephedge/train.py` end-to-end training and the cached settings of sessions 00, 01 and 02.
- `tests/` checks against closed forms and known rates; `pytest --impl=exercises` runs
  them against Corey's own implementations in `exercises/`.

MIT licence.
