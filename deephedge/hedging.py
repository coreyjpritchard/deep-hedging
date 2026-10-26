"""Hedging P&L of a short European option.

The P&L engine scores a position path. It does not know which volatility
model produced the prices. Build an `Environment` from `deephedge.env`
and call `env.pnl`, or call the functions here.

Convention. The hedger sells the option for `premium`, trades the
instruments, and pays costs. Positions are chosen at `t_0, ..., t_{n-1}`;
position k is held over `[t_k, t_{k+1}]`. The hedger starts flat and
liquidates at maturity.

    P&L = premium - discount_T * payoff + gains - costs

Gains are computed on discounted instrument prices, which folds in interest
at a constant rate. `pnl_from_positions` is the one-instrument form: pass
prices that are already discounted, and a payoff that is already multiplied
by `discount_T`. `delta_hedge_pnl` builds a Black-Scholes environment
and does that discounting for you.

With one instrument and a proportional cost `c`, every trade costs
`c * |change in position| * S`, including the opening and closing trades
(Buehler, Gonon, Teichmann, Wood 2019, section 3). Several instruments are
summed the same way. Every P&L here goes through `Environment.pnl` or
`Environment.pnl_torch`; the functions below are the PR 2 entry points.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from deephedge.env import (
    BlackScholes,
    Environment,
    Paths,
    ProportionalCost,
    bs_call_delta,
    bs_call_price,
)

__all__ = [
    "bs_call_delta",
    "bs_call_price",
    "call_payoff",
    "delta_hedge_pnl",
    "delta_positions",
    "pnl_from_positions",
    "pnl_from_positions_torch",
    "trading_costs",
]


def call_payoff(K: float) -> Callable:
    """Call payoff max(S - K, 0). Works on numpy arrays and torch tensors alike."""

    def payoff(s_T):
        return (s_T - K).clip(0.0)

    return payoff


def trading_costs(paths: np.ndarray, positions: np.ndarray, cost: float) -> np.ndarray:
    """Total proportional cost per path, including the opening and liquidating trades."""
    paths = np.asarray(paths, dtype=float)
    return ProportionalCost(float(cost)).amount(paths[:, :, None], positions)


def pnl_from_positions(
    paths: np.ndarray,
    positions: np.ndarray,
    payoff,
    cost: float,
    premium: float = 0.0,
) -> np.ndarray:
    """Terminal P&L of a short option hedged with `positions`, shape (m, n).

    `paths` are prices of one instrument, shape (m, n + 1), already
    discounted if interest is nonzero. `payoff` is a function of `S_T`
    or an array of terminal payoffs, shape (m,), already in the same units
    as `paths`. Every policy (delta, network, no hedge) can go through
    this function. It is `Environment.pnl` on one undiscounted instrument.
    """
    paths = np.asarray(paths, dtype=float)
    positions = np.asarray(positions, dtype=float)
    pay = payoff(paths[:, -1]) if callable(payoff) else np.asarray(payoff, dtype=float)
    n = paths.shape[1] - 1
    wrapped = Paths(
        time=np.linspace(0.0, 1.0, n + 1),
        instruments=paths[:, :, None],
        features=paths[:, :, None],
        discount=np.ones(n + 1),
    )
    return Environment(None, cost).pnl(wrapped, positions, premium, payoff=pay)


def pnl_from_positions_torch(
    paths: torch.Tensor,
    positions: torch.Tensor,
    payoff,
    cost: float | tuple[float, ...] = 0.0,
    premium: float = 0.0,
    *,
    exact: bool = True,
) -> torch.Tensor:
    """Torch terminal P&L. Alias of `Environment(None, cost).pnl_torch(...)`.

    `paths` are discounted prices, shape `(m, n + 1)` or `(m, n + 1, d)`;
    `payoff` is discounted, a tensor or a callable of the terminal spot.
    """
    return Environment(None, cost).pnl_torch(paths, positions, payoff, premium, exact=exact)


def delta_positions(
    paths: np.ndarray, K: float, T: float, sigma: float, r: float = 0.0
) -> np.ndarray:
    """Black-Scholes delta at t_0, ..., t_{n-1}, shape (m, n).

    This is the spot column of `BlackScholes.benchmark_positions`.
    """
    paths = np.asarray(paths, dtype=float)
    market = BlackScholes(s0=float(paths[0, 0]), strike=K, maturity=T, sigma=sigma, r=r)
    return market.benchmark_positions(market.paths_from_spot(paths))[:, :, 0]


def delta_hedge_pnl(
    paths: np.ndarray, K: float, T: float, sigma: float, r: float, cost: float
) -> np.ndarray:
    """P&L of a short call sold at its Black-Scholes price and delta hedged.

    Amounts are discounted to t = 0. Returns shape (m,). The hedge is the
    market's benchmark position, and the P&L is the environment's.
    """
    paths = np.asarray(paths, dtype=float)
    market = BlackScholes(s0=float(paths[0, 0]), strike=K, maturity=T, sigma=sigma, r=r)
    sim = market.paths_from_spot(paths)
    return Environment(market, cost).pnl(sim, market.benchmark_positions(sim))
