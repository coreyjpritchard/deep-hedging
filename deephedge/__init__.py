"""Deep hedging under transaction costs, compared with delta hedging.

The market API is `deephedge.env`: `BlackScholes`, `Environment`,
`ProportionalCost`, `Paths`. Heston, with a variance swap, is
`deephedge.heston.Heston`. A hedger sees features from a `Paths`
batch. The P&L and the risk measures do not import a volatility model.
"""
