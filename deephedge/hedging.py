"""Hedging P&L of a short European option.

Convention. A hedger sells the option for `premium`, trades the underlying, and pays
proportional costs. Positions are chosen at t_0, ..., t_{n-1}; position k is held over
[t_k, t_{k+1}]. The hedger starts flat and liquidates at maturity, and every trade costs
cost * |change in position| * S (Buehler, Gonon, Teichmann, Wood 2019, section 3).

    P&L = premium - payoff(S_T) + sum_k delta_k (S_{k+1} - S_k) - costs

With a nonzero rate r, pass discounted prices e^{-r t} S and a discounted payoff;
`delta_hedge_pnl` does this for you.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import norm

from deephedge.markets import time_grid


def bs_call_price(S, K: float, tau, sigma: float, r: float = 0.0):
    """Black-Scholes call price with time to maturity tau. Vectorised in S and tau."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    intrinsic = np.maximum(S - K, 0.0)
    safe_tau = np.where(tau > 0, tau, 1.0)
    sd = sigma * np.sqrt(safe_tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * safe_tau) / sd
    d2 = d1 - sd
    price = S * norm.cdf(d1) - K * np.exp(-r * safe_tau) * norm.cdf(d2)
    return np.where(tau > 0, price, intrinsic)


def bs_call_delta(S, K: float, tau, sigma: float, r: float = 0.0):
    """Black-Scholes call delta N(d1). At tau = 0 returns the indicator S > K."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.where(tau > 0, tau, 1.0)
    sd = sigma * np.sqrt(safe_tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * safe_tau) / sd
    return np.where(tau > 0, norm.cdf(d1), (S > K).astype(float))


def call_payoff(K: float) -> Callable:
    """Call payoff max(S - K, 0). Works on numpy arrays and torch tensors alike."""

    def payoff(s_T):
        return (s_T - K).clip(0.0)

    return payoff


def trading_costs(paths: np.ndarray, positions: np.ndarray, cost: float) -> np.ndarray:
    """Total proportional cost per path, including the opening and liquidating trades."""
    m = paths.shape[0]
    full = np.concatenate([np.zeros((m, 1)), positions, np.zeros((m, 1))], axis=1)
    return cost * (np.abs(np.diff(full, axis=1)) * paths).sum(axis=1)


def pnl_from_positions(
    paths: np.ndarray,
    positions: np.ndarray,
    payoff,
    cost: float,
    premium: float = 0.0,
) -> np.ndarray:
    """Terminal P&L of a short option hedged with `positions`, shape (m, n).

    `paths` are prices, shape (m, n + 1). `payoff` is a function of S_T or an array of
    terminal payoffs, shape (m,). Every policy (delta, network, no hedge) goes through
    this one function.
    """
    paths = np.asarray(paths, dtype=float)
    positions = np.asarray(positions, dtype=float)
    pay = payoff(paths[:, -1]) if callable(payoff) else np.asarray(payoff, dtype=float)
    gains = (positions * np.diff(paths, axis=1)).sum(axis=1)
    return premium - pay + gains - trading_costs(paths, positions, cost)


def pnl_from_positions_torch(
    paths: torch.Tensor,
    positions: torch.Tensor,
    payoff,
    cost: float,
    premium: float = 0.0,
) -> torch.Tensor:
    """Torch version of `pnl_from_positions`, differentiable in positions."""
    pay = payoff(paths[:, -1]) if callable(payoff) else payoff
    gains = (positions * (paths[:, 1:] - paths[:, :-1])).sum(dim=1)
    zero = torch.zeros_like(paths[:, :1])
    full = torch.cat([zero, positions, zero], dim=1)
    costs = cost * ((full[:, 1:] - full[:, :-1]).abs() * paths).sum(dim=1)
    return premium - pay + gains - costs


def delta_positions(
    paths: np.ndarray, K: float, T: float, sigma: float, r: float = 0.0
) -> np.ndarray:
    """Black-Scholes delta at t_0, ..., t_{n-1}, shape (m, n)."""
    n = paths.shape[1] - 1
    tau = T - time_grid(n, T)[:-1]
    return bs_call_delta(paths[:, :-1], K, tau, sigma, r)


def delta_hedge_pnl(
    paths: np.ndarray, K: float, T: float, sigma: float, r: float, cost: float
) -> np.ndarray:
    """P&L of a short call sold at its Black-Scholes price and delta hedged at each step.

    Amounts are discounted to t = 0. Returns shape (m,).
    """
    paths = np.asarray(paths, dtype=float)
    n = paths.shape[1] - 1
    discount = np.exp(-r * time_grid(n, T))
    premium = float(bs_call_price(paths[0, 0], K, T, sigma, r))
    positions = delta_positions(paths, K, T, sigma, r)
    payoff = discount[-1] * np.maximum(paths[:, -1] - K, 0.0)
    return pnl_from_positions(paths * discount, positions, payoff, cost, premium)
