"""Market environments: a volatility model plus the liability being hedged.

A hedger, the P&L engine and the risk measures talk only to this interface.
The market simulates paths and prices the claim. A cost model prices trades.
Swapping the market, or the cost model, does not change what a hedger sees:
the time, the feature row at that date, and the position it already holds.

`Paths.instruments` has shape `(m, n + 1, d)`, the prices of the `d`
tradable instruments, including `t = 0`. These prices enter the P&L.
`Paths.features` has shape `(m, n + 1, f)`, what a hedger may observe.
Features need not be prices. A Markov policy reads one date. A recurrent
policy may read the path up to that date. The hedger's own position is not
a feature; the rollout passes it in separately. `Paths.discount` holds
`e^{-rt}` to time 0, shape `(n + 1,)` while rates are deterministic.

Black-Scholes has `d = 1`, the spot, and `f = 1`, the same spot. The
benchmark hedge is the analytic call delta. The premium is the analytic
call price.

Heston is `deephedge.generators.heston.Heston`. Features are `(spot, variance)`,
so `f = 2`. Instruments are the spot and a variance swap, so `d = 2`.
The swap's price belongs in `instruments`; the variance factor stays in
`features`, because a feature is not something the P&L can trade. The
benchmark hedge has one weight per instrument.

Costs sit on the environment, not inside the simulator, so one batch of
paths can be scored under several charges. `ProportionalCost` multiplies
the discounted price, or a cash `unit` when the mark is the wrong notional
(the variance swap). `FixedPlusProportional` adds a cash charge each time
an instrument trades, including liquidation at maturity.
`QuadraticCost` is a smooth market-impact charge. See `deephedge.costs`.

Since M02 the pieces live in their own modules and are re-exported here, so
PR 1 imports keep working: `Paths` in `deephedge.paths`, the `Market` base in
`deephedge.generators.base` (deprecated, one release), `BlackScholes` in
`deephedge.generators.bs`, and the Black-Scholes formulas in
`deephedge.pricing`. New code uses a `Generator`, a contract and a `Pricer`.

Not implemented yet, and where it will plug in:

* Rough Bergomi: the volatility has memory, so the feature path has to carry
  it. The simulator will come from the sibling package `roughvol` (M11a).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from deephedge.costs import (
    FixedPlusProportional,
    ProportionalCost,
    QuadraticCost,
    as_cost,
)
from deephedge.generators.base import Market, call_on_spot
from deephedge.generators.base import check_maturity as _check_maturity  # noqa: F401
from deephedge.generators.bs import BlackScholes
from deephedge.paths import Paths
from deephedge.pricing import bs_call_delta, bs_call_price

__all__ = [
    "BlackScholes",
    "CostModel",
    "Environment",
    "FixedPlusProportional",
    "Market",
    "Paths",
    "ProportionalCost",
    "QuadraticCost",
    "bs_call_delta",
    "bs_call_price",
    "call_on_spot",
    "terminal_pnl",
]


class CostModel(Protocol):
    """Cost of a book of trades. Pluggable, and independent of the market.

    `prices` has shape `(m, n + 1, d)` and is what the charge is levied
    on. The P&L engine passes discounted prices, so a cash cost paid at `t`
    is already in time-0 money: the proportional cost `c |dq| S_t` becomes
    `c |dq| e^{-rt} S_t`. `positions` has shape `(m, n, d)`, the
    holding on each interval. The book is flat before `t_0` and flat again
    at maturity. Liquidation is charged: the closing trade is part of
    `amount`, not an optional extra.

    The concrete models are `ProportionalCost`, `FixedPlusProportional`
    and `QuadraticCost` in `deephedge.costs`. A model may charge a cash
    notional per instrument (`unit`) instead of the price. The variance
    swap uses that, because its mark is not a meaningful notional. The
    swap's price, the gains and the hedge ratio stay as the market
    quoted them. A torch training step calls `amount_torch` on those
    models. Evaluation calls `amount`, which is exact.

    Returns the total cost per path, shape `(m,)`.
    """

    def amount(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Total cost per path."""
        ...


def _as_positions(positions: np.ndarray, n_instruments: int) -> np.ndarray:
    """Positions as an array of shape (m, n, d). A (m, n) array means d = 1."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 2:
        if n_instruments != 1:
            raise ValueError(f"positions have shape {pos.shape}; expected (m, n, {n_instruments})")
        return pos[:, :, None]
    if pos.ndim != 3 or pos.shape[-1] != n_instruments:
        raise ValueError(f"positions have shape {pos.shape}; expected (m, n, {n_instruments})")
    return pos


def terminal_pnl(
    paths: Paths,
    positions: np.ndarray,
    payoff: np.ndarray,
    cost: CostModel | float | tuple[float, ...] = 0.0,
    premium: float = 0.0,
) -> np.ndarray:
    """Terminal P&L in time-0 money of a short liability, shape (m,).

    `P&L = premium - discount_T * payoff + gains - costs`.

    `positions` has shape `(m, n)` for one instrument or `(m, n, d)`
    in general. Position k is held over `[t_k, t_{k+1}]`. Gains sum, over
    instruments and dates, the position times the change in the discounted
    instrument price. Costs are whatever `cost` charges on those discounted
    prices; a number or a tuple is a proportional rate.
    """
    prices = paths.discounted_instruments()
    pos = _as_positions(positions, prices.shape[-1])
    if pos.shape[0] != prices.shape[0] or pos.shape[1] != prices.shape[1] - 1:
        raise ValueError(f"positions shape {pos.shape} does not match prices shape {prices.shape}")
    gains = (pos * np.diff(prices, axis=1)).sum(axis=(1, 2))
    pay = np.asarray(payoff, dtype=float) * paths.discount_terminal()
    costs = as_cost(cost).amount(prices, pos)
    return np.asarray(premium, dtype=float) - pay + gains - costs


@dataclass(frozen=True)
class Environment:
    """What a hedger is trained against: a market and a cost model.

    Simulation uses only the market, so the same paths can be scored under
    another cost without drawing again. `pnl` is the terminal P&L in
    time-0 money. Risk measures take that array and nothing else.
    """

    market: Market
    cost: CostModel | float | tuple[float, ...] = 0.0

    def simulate(self, m: int, n: int, seed: int) -> Paths:
        return self.market.simulate(m, n, seed)

    def pnl(
        self,
        paths: Paths,
        positions: np.ndarray,
        premium: float | None = None,
    ) -> np.ndarray:
        """P&L of `positions` in this environment. Shape (m,).

        The premium defaults to the market's time-0 price. Pass another
        number to sell the claim at a different price.
        """
        if premium is None:
            premium = self.market.premium()
        return terminal_pnl(paths, positions, self.market.payoff(paths), self.cost, premium)
