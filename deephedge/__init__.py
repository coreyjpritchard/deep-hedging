"""Deep hedging under transaction costs, compared with delta hedging.

The market API is `deephedge.env`: `BlackScholes`, `Environment`,
`ProportionalCost`, `FixedPlusProportional`, `QuadraticCost`, `Paths`.
Heston, with a variance swap, is `deephedge.heston.Heston`. A hedger
sees features from a `Paths` batch. The P&L and the risk measures do
not import a volatility model. Costs are `deephedge.costs`.

The playground in `playground/app.py` is a marimo app on top of this
package. Training, costs, P&L and networks are the functions above.
There is no second implementation in another language.

The live view in `playground/live.py` steps one path. The account it
walks is `deephedge.live`, and the terminal P/L is `terminal_pnl`.
"""
