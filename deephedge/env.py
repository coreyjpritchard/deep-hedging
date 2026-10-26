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
    "Ledger",
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

    Deprecated alias, kept for PR 1 imports: `Environment(None, cost).pnl(paths,
    positions, premium, payoff=payoff)`. `payoff` is undiscounted.
    """
    return Environment(None, cost).pnl(paths, positions, premium, payoff=payoff)


def _torch_prices_and_positions(prices, positions):
    """Prices as `(m, n + 1, d)` and positions as `(m, n, d)`, checked."""
    if prices.ndim == 2:
        prices = prices.unsqueeze(-1)
        positions = positions.unsqueeze(-1)
    elif prices.ndim != 3:
        raise ValueError(f"prices must have shape (m, n + 1) or (m, n + 1, d), got {prices.shape}")
    if positions.shape != (prices.shape[0], prices.shape[1] - 1, prices.shape[2]):
        raise ValueError(f"positions shape {tuple(positions.shape)} does not match {prices.shape}")
    return prices, positions


@dataclass(frozen=True)
class Ledger:
    """The per-date book of a hedge, in time-0 money. Built by `Environment.ledger`.

    Dates are `t_0, ..., t_n`; `d` instruments. At date k the hedger trades
    from the previous position to `position[:, k]` (zero before `t_0` and
    after `t_n`, where the book is liquidated) and pays `cost[:, k]`.

    * `price` `(m, n + 1, d)`: discounted instrument prices.
    * `position` `(m, n + 1, d)`: held after trading at `t_k`; the last is 0.
    * `trade` `(m, n + 1, d)`: signed change in position at `t_k`.
    * `cost` `(m, n + 1)`: cost paid at `t_k`.
    * `gain` `(m, n + 1)`: gain of the previous position over `[t_{k-1}, t_k]`; 0 at `t_0`.
    * `cash` `(m, n + 1)`: premium, minus what trades and costs paid, up to `t_k`.
    * `holdings` `(m, n + 1)`: `position . price` after trading at `t_k`.
    * `payoff` `(m,)`: discounted liability paid at `t_n`.
    * `value` `(m, n + 1)`: premium plus cumulative gains minus cumulative
      costs, less the payoff at `t_n`. It equals `cash + holdings` before
      maturity, and its last column is the terminal P&L.
    """

    time: np.ndarray
    price: np.ndarray
    position: np.ndarray
    trade: np.ndarray
    cost: np.ndarray
    gain: np.ndarray
    cash: np.ndarray
    holdings: np.ndarray
    payoff: np.ndarray
    value: np.ndarray

    @property
    def final_value(self) -> np.ndarray:
        """Terminal P&L per path, shape (m,). Equals `Environment.pnl`."""
        return self.value[:, -1]


@dataclass(frozen=True)
class Environment:
    """What a hedger is trained against: a market and a cost model.

    Simulation uses only the market, so the same paths can be scored under
    another cost without drawing again. `pnl` (numpy) and `pnl_torch`
    (differentiable) are the only two terminal P&L functions in the package;
    `ledger` is the same arithmetic date by date. Risk measures take the
    P&L array and nothing else.

        P&L = premium - discount_T * payoff + gains - costs

    Gains sum, over instruments and dates, the position held over
    `[t_k, t_{k+1}]` times the change in the discounted instrument price.
    Costs are whatever `cost` charges on those discounted prices, opening and
    liquidation included; a number or a tuple is a proportional rate.

    `market` may be `None` for array-level scoring (the training loops and
    the PR 1 aliases): then pass `premium` and `payoff` explicitly.
    """

    market: Market | None
    cost: CostModel | float | tuple[float, ...] = 0.0

    def simulate(self, m: int, n: int, seed: int) -> Paths:
        return self._market().simulate(m, n, seed)

    def _market(self) -> Market:
        if self.market is None:
            raise ValueError("this Environment has no market; pass premium and payoff")
        return self.market

    def _premium(self, premium):
        return self._market().premium() if premium is None else premium

    def _payoff(self, paths: Paths, payoff) -> np.ndarray:
        """Discounted liability at maturity, shape (m,)."""
        raw = self._market().payoff(paths) if payoff is None else payoff
        return np.asarray(raw, dtype=float) * paths.discount_terminal()

    def pnl(
        self,
        paths: Paths,
        positions: np.ndarray,
        premium: float | None = None,
        *,
        payoff: np.ndarray | None = None,
    ) -> np.ndarray:
        """Terminal P&L of `positions` in time-0 money. Shape (m,).

        `positions` has shape `(m, n)` for one instrument or `(m, n, d)`.
        The premium defaults to the market's time-0 price and `payoff` to the
        market's undiscounted liability; pass either to override.
        """
        prices = paths.discounted_instruments()
        pos = _as_positions(positions, prices.shape[-1])
        if pos.shape[0] != prices.shape[0] or pos.shape[1] != prices.shape[1] - 1:
            raise ValueError(
                f"positions shape {pos.shape} does not match prices shape {prices.shape}"
            )
        gains = (pos * np.diff(prices, axis=1)).sum(axis=(1, 2))
        pay = self._payoff(paths, payoff)
        costs = as_cost(self.cost).amount(prices, pos)
        return np.asarray(self._premium(premium), dtype=float) - pay + gains - costs

    def pnl_torch(
        self,
        prices,
        positions,
        payoff=None,
        premium: float | None = None,
        *,
        exact: bool = True,
    ):
        """Terminal P&L in torch, differentiable in `positions`. Shape (m,).

        `prices` is a `Paths` or a tensor of discounted instrument prices,
        shape `(m, n + 1, d)` or `(m, n + 1)` for one instrument. From a
        `Paths` the prices, the discounted payoff and the premium come from
        the market in float64 unless given. With a tensor, `payoff` is the
        discounted liability, a tensor of shape `(m,)` or a callable of the
        terminal spot (column 0), and `premium` is required unless the
        environment has a market.

        `exact=True` matches `pnl`, including the fixed-cost indicator.
        `exact=False` asks a cost model for its smooth training surrogate.
        A rate, or a tuple of rates, charges `rate * |change| * price` and
        has no surrogate. Gains and costs sum over instruments; the payoff is
        charged once.
        """
        import torch

        if isinstance(prices, Paths):
            paths = prices
            disc = torch.as_tensor(paths.discounted_instruments(), dtype=torch.float64)
            if payoff is None:
                payoff = torch.as_tensor(self._payoff(paths, None), dtype=torch.float64)
            prices = disc
        if payoff is None:
            raise ValueError("pass payoff with a tensor of prices")
        premium = self._premium(premium)
        model = self.cost if hasattr(self.cost, "amount_torch") else None
        if prices.ndim == 2 and model is None:
            # One instrument and a rate: the Black-Scholes training loss, kept in this
            # order so a float32 training run sums exactly as it did in PR 2.
            pay = payoff(prices[:, -1]) if callable(payoff) else payoff
            gains = (positions * (prices[:, 1:] - prices[:, :-1])).sum(dim=1)
            zero = torch.zeros_like(prices[:, :1])
            full = torch.cat([zero, positions, zero], dim=1)
            costs = self.cost * ((full[:, 1:] - full[:, :-1]).abs() * prices).sum(dim=1)
            return premium - pay + gains - costs
        prices, positions = _torch_prices_and_positions(prices, positions)
        pay = payoff(prices[:, -1, 0]) if callable(payoff) else payoff
        gains = (positions * (prices[:, 1:] - prices[:, :-1])).sum(dim=(1, 2))
        if model is not None:
            costs = model.amount_torch(prices, positions, exact=exact)
        else:
            m, _, d = prices.shape
            zero = torch.zeros(m, 1, d, dtype=prices.dtype, device=prices.device)
            full = torch.cat([zero, positions, zero], dim=1)
            traded = (full[:, 1:] - full[:, :-1]).abs()
            rate = torch.as_tensor(self.cost, dtype=prices.dtype, device=prices.device)
            costs = (rate * traded * prices).sum(dim=(1, 2))
        return premium - pay + gains - costs

    def ledger(
        self,
        paths: Paths,
        positions: np.ndarray,
        premium: float | None = None,
        *,
        payoff: np.ndarray | None = None,
    ) -> Ledger:
        """The book of `positions` date by date. `final_value` equals `pnl`.

        Same inputs as `pnl`. The cost model must split its charge by date
        (`per_date`); the three in `deephedge.costs` do.
        """
        model = as_cost(self.cost)
        if not hasattr(model, "per_date"):
            raise TypeError(f"{type(model).__name__} has no per_date, so no ledger")
        prices = paths.discounted_instruments()
        pos = _as_positions(positions, prices.shape[-1])
        if pos.shape[0] != prices.shape[0] or pos.shape[1] != prices.shape[1] - 1:
            raise ValueError(
                f"positions shape {pos.shape} does not match prices shape {prices.shape}"
            )
        m, _, d = pos.shape
        flat = np.zeros((m, 1, d))
        held = np.concatenate([pos, flat], axis=1)
        trade = np.diff(np.concatenate([flat, pos, flat], axis=1), axis=1)
        cost = model.per_date(prices, pos)
        gain = np.zeros(cost.shape)
        gain[:, 1:] = (pos * np.diff(prices, axis=1)).sum(axis=2)
        premium = float(self._premium(premium))
        cash = premium - np.cumsum((trade * prices).sum(axis=2) + cost, axis=1)
        holdings = (held * prices).sum(axis=2)
        pay = self._payoff(paths, payoff)
        value = premium + np.cumsum(gain - cost, axis=1)
        value[:, -1] -= pay
        return Ledger(
            time=np.asarray(paths.time, dtype=float),
            price=prices,
            position=held,
            trade=trade,
            cost=cost,
            gain=gain,
            cash=cash,
            holdings=holdings,
            payoff=pay,
            value=value,
        )
