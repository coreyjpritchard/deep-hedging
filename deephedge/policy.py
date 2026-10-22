"""Hedging policies.

A policy maps the information at one date to a position in the instruments:
the time, the market's feature row, and the position held coming into that
date. It does not see the market object. The benchmark hedge is the
exception: it is `Market.benchmark_positions`, computed by the market.
`DeltaPolicy` is that hedge for Black-Scholes, written in torch so it can
share a rollout with the network. On a Heston market the benchmark is
`Heston.benchmark_positions`, and `HestonPolicy` is the network that
reads spot and variance and returns a spot weight and a variance-swap
weight. `DeltaPolicy` stays the Black-Scholes one.

Session 00's network reads three scalars built from the spot (feature column
0): time to maturity over T, log moneyness over sqrt(tau), and the previous
position. Those weights are cached. `rollout` is that spot calling
convention, previous position shape (m,) and a returned position shape (m,).
`rollout_positions` is the general one: features `(m, f)` in, position
`(m, d)` out, with `d` equal to the number of instruments and not
necessarily to `f`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class MLPPolicy(nn.Module):
    """Feedforward network on (time to maturity, log moneyness, previous position).

    Features are scaled to order one: tau / T and log(S / K) / sqrt(tau). Scaling the log
    moneyness by sqrt(tau) is the one piece of prior knowledge given to the network: the
    Black-Scholes delta N(d1) is then a gentle function of the inputs even near maturity.
    The previous position enters because with costs the best trade depends on where you
    start.
    """

    def __init__(self, strike: float, maturity: float, width: int = 32, depth: int = 2):
        super().__init__()
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.width = width
        self.depth = depth
        layers: list[nn.Module] = []
        d_in = 3
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.ReLU()]
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def features(self, t: float, S: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        tau = self.maturity - t
        moneyness = torch.log(S / self.strike) / math.sqrt(tau)
        return torch.stack([torch.full_like(S, tau / self.maturity), moneyness, prev], dim=-1)

    def forward(self, t: float, S: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(t, S, prev)).squeeze(-1)

    def config(self) -> dict:
        return {
            "strike": self.strike,
            "maturity": self.maturity,
            "width": self.width,
            "depth": self.depth,
        }


class DeltaPolicy(nn.Module):
    """Black-Scholes delta N(d1). Ignores the previous position."""

    def __init__(self, strike: float, maturity: float, sigma: float, r: float = 0.0):
        super().__init__()
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.sigma = float(sigma)
        self.r = float(r)

    def forward(self, t: float, S: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        tau = self.maturity - t
        if tau <= 0:
            return (S > self.strike).to(S.dtype)
        sd = self.sigma * math.sqrt(tau)
        d1 = (torch.log(S / self.strike) + (self.r + 0.5 * self.sigma**2) * tau) / sd
        return torch.special.ndtr(d1)


class HestonPolicy(nn.Module):
    """Shared-weight network for the spot and a variance swap.

    Five inputs, each order one: time to maturity over T, log moneyness
    over sqrt(tau), variance over theta, the incoming spot position, and
    the incoming variance-swap position divided by `varswap_scale`.

    The model hedge's variance-swap weight is a few hundred, because the
    swap is an integrated variance of order 10^-3 while vega is order 10.
    The last layer outputs that weight in units of `varswap_scale`
    (default 100) and the forward pass multiplies it back, so the network
    is asked for a number of order one. It starts at half a share of the
    spot and a flat variance swap, with small last-layer weights.
    """

    def __init__(
        self,
        strike: float,
        maturity: float,
        theta: float,
        width: int = 64,
        depth: int = 3,
        varswap_scale: float = 100.0,
    ):
        super().__init__()
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.theta = float(theta)
        self.width = int(width)
        self.depth = int(depth)
        self.varswap_scale = float(varswap_scale)
        layers: list[nn.Module] = []
        d_in = 5
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.ReLU()]
            d_in = width
        layers.append(nn.Linear(d_in, 2))
        self.net = nn.Sequential(*layers)
        last = self.net[-1]
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[0] = 0.5

    def forward(self, t: float, features: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(features, dtype=torch.float32)
        prev = torch.as_tensor(prev, dtype=torch.float32)
        tau = max(self.maturity - float(t), 1e-8)
        spot = features[:, 0]
        variance = features[:, 1]
        scaled = torch.stack(
            [
                torch.full_like(spot, tau / self.maturity),
                torch.log(spot / self.strike) / math.sqrt(tau),
                variance / self.theta,
                prev[:, 0],
                prev[:, 1] / self.varswap_scale,
            ],
            dim=-1,
        )
        raw = self.net(scaled)
        return torch.stack([raw[:, 0], raw[:, 1] * self.varswap_scale], dim=-1)

    def config(self) -> dict:
        return {
            "strike": self.strike,
            "maturity": self.maturity,
            "theta": self.theta,
            "width": self.width,
            "depth": self.depth,
            "varswap_scale": self.varswap_scale,
        }


class NoHedgePolicy(nn.Module):
    """Hold nothing. The P&L is premium minus payoff."""

    def forward(self, t: float, S: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(S)


def rollout(policy: nn.Module, paths: torch.Tensor, T: float) -> torch.Tensor:
    """Positions chosen by `policy` along price paths (m, n + 1). Returns (m, n).

    The hedger starts flat; each step feeds the previous position back in.
    """
    m, n_plus_1 = paths.shape
    n = n_plus_1 - 1
    dt = T / n
    prev = torch.zeros(m, dtype=paths.dtype)
    positions = []
    for k in range(n):
        prev = policy(k * dt, paths[:, k], prev)
        positions.append(prev)
    return torch.stack(positions, dim=1)


def rollout_numpy(policy: nn.Module, paths: np.ndarray, T: float) -> np.ndarray:
    """`rollout` without gradients, numpy in and out, for evaluation."""
    with torch.no_grad():
        return rollout(policy, torch.as_tensor(paths, dtype=torch.float32), T).double().numpy()


def rollout_positions(
    policy: nn.Module,
    features: torch.Tensor | np.ndarray,
    time: torch.Tensor | np.ndarray,
    n_instruments: int,
) -> torch.Tensor:
    """Positions along a feature path. Shape (m, n, n_instruments).

    `features` has shape `(m, n + 1, f)` and `time` has shape
    `(n + 1,)`. At date k the policy is called as
    `policy(t_k, features[:, k, :], prev)` and must return a tensor of
    shape `(m, n_instruments)`. `prev` starts at zero. `f` and
    `n_instruments` are independent: a Heston policy can read
    `(spot, variance)` and still return a single weight in the spot, or
    a weight in the spot and one in a variance swap.

    Session 00's policies do not use this signature. They take the spot as
    a vector of shape `(m,)`. `rollout` and `rollout_paths` are that
    case.
    """
    feat = torch.as_tensor(features, dtype=torch.float32)
    times = np.asarray(time, dtype=float)
    m, n_plus, _f = feat.shape
    n = n_plus - 1
    prev = torch.zeros(m, n_instruments, dtype=feat.dtype)
    chosen = []
    for k in range(n):
        prev = policy(float(times[k]), feat[:, k, :], prev)
        if tuple(prev.shape) != (m, n_instruments):
            raise ValueError(
                f"policy returned shape {tuple(prev.shape)}, expected {(m, n_instruments)}"
            )
        chosen.append(prev)
    return torch.stack(chosen, dim=1)


def rollout_paths(policy: nn.Module, paths) -> np.ndarray:
    """Spot-policy positions along a market path, shape (m, n).

    Reads feature column 0 as the spot and calls `rollout_numpy`. That is
    the Black-Scholes observation. A path with more than one feature should
    go through `rollout_positions` and a policy that reads the whole row.
    """
    features = np.asarray(paths.features)
    if features.ndim != 3 or features.shape[-1] != 1:
        raise ValueError(
            "rollout_paths reads a single feature, the spot. "
            f"Got features of shape {features.shape}. Use rollout_positions."
        )
    maturity = float(np.asarray(paths.time)[-1])
    return rollout_numpy(policy, features[:, :, 0], maturity)


def position_grid(
    policy: nn.Module, t: float, spots: np.ndarray, prevs: np.ndarray
) -> np.ndarray:
    """Position chosen at time t for every (previous position, spot) pair.

    Returns shape (len(prevs), len(spots)).
    """
    s, p = np.meshgrid(spots, prevs)
    with torch.no_grad():
        out = policy(
            t,
            torch.as_tensor(s.ravel(), dtype=torch.float32),
            torch.as_tensor(p.ravel(), dtype=torch.float32),
        )
    return out.double().numpy().reshape(s.shape)


def held_positions(
    policy: nn.Module, paths: np.ndarray, T: float, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Spot at step k and the position the policy holds after trading at step k, per path.

    The cloud of these pairs is the empirical no-trade band: where the policy's positions
    actually sit relative to delta. Requires 0 <= k < n.
    """
    positions = rollout_numpy(policy, paths, T)
    return paths[:, k], positions[:, k]


def held_instrument(
    policy: nn.Module,
    features: np.ndarray,
    time: np.ndarray,
    n_instruments: int,
    k: int,
    instrument: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature column 0 at date k, and the position in one instrument after trading there.

    `features` has shape `(m, n + 1, f)`. The returned position is column
    `instrument` of `rollout_positions`. Feature column 0 is the spot.
    """
    with torch.no_grad():
        positions = rollout_positions(policy, features, time, n_instruments).double().numpy()
    return np.asarray(features)[:, k, 0], positions[:, k, instrument]


def heston_spot_response(
    policy: nn.Module,
    t: float,
    spots: np.ndarray,
    variance: float,
    prev_spot: float,
    prev_swap: float,
) -> np.ndarray:
    """Spot weight and variance-swap weight along a grid of spots.

    Variance and both incoming positions are held fixed, which is the
    Heston analogue of `position_grid`. Returns shape `(len(spots), 2)`.
    """
    spots = np.asarray(spots, dtype=np.float32).reshape(-1)
    features = torch.zeros(spots.size, 2)
    features[:, 0] = torch.as_tensor(spots)
    features[:, 1] = float(variance)
    prev = torch.zeros(spots.size, 2)
    prev[:, 0] = float(prev_spot)
    prev[:, 1] = float(prev_swap)
    with torch.no_grad():
        out = policy(float(t), features, prev)
    return out.double().numpy()
