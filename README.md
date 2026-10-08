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

Findings from session 00, on 20 000 fresh paths with a 50 bp cost:

- Delta hedging has mean P&L -1.37 against a premium of 2.28; the network, -0.83. The
  network's entropic risk at lambda = 1 is 1.05 against 1.54 for delta.
- With cost 100 bp, going from 10 to 120 rehedges takes delta hedging's mean P&L from
  -1.96 to -4.48, close to the square-root-of-n law. The network loses far less.
- The learned band is softer than the Whalley-Wilmott band, barely widens between 50
  and 200 bp, and sits above delta. These are the questions of session 02.

The plan is in [ROADMAP.md](ROADMAP.md) and dated notes are in [LOG.md](LOG.md).

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
marimo run sessions/00_cost_of_hedging.py     # the app; marimo edit to see the code
python sessions/00_cost_of_hedging.py         # runs every cell as a script
pytest -m "not slow"                          # tests, a few seconds
python -m deephedge.train                     # retrain the cached weights, about 3 minutes
python -m deephedge.figures                   # redraw the README figure
```

The app opens with cached weights in `data/weights/`. The *Retrain with these settings*
button trains a new network in about 30 seconds at 30 steps on four cores.

## Layout

- `deephedge/markets.py` Black-Scholes path simulator, numpy and torch.
- `deephedge/hedging.py` Black-Scholes price and delta, and one P&L function that every
  policy goes through.
- `deephedge/risk.py` entropic risk and CVaR, numpy and torch.
- `deephedge/policy.py` the feature-based network, a delta policy with the same call
  signature, and rollouts.
- `deephedge/train.py` end-to-end training and the cached settings of session 00.
- `tests/` checks against closed forms and known rates; `pytest --impl exercises` runs
  them against Corey's own implementations in `exercises/`.

MIT licence.
