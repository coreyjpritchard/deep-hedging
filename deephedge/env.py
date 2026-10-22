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

Heston is `deephedge.heston.Heston`. Features are `(spot, variance)`,
so `f = 2`. Instruments are the spot and a variance swap, so `d = 2`.
The swap's price belongs in `instruments`; the variance factor stays in
`features`, because a feature is not something the P&L can trade. The
benchmark hedge has one weight per instrument.

Not implemented yet, and where they will plug in:

* Rough Bergomi: the volatility has memory, so the feature path has to carry
  it. The simulator will come from the sibling package `roughvol`.
* Further cost models: implement `CostModel.amount`. `ProportionalCost`
  is the only one so far, the cost in Buehler, Gonon, Teichmann and Wood
  (2019), section 3. Costs sit on the environment, not inside the simulator,
  so one batch of paths can be scored at several costs. A proportional
  rate times the variance swap's mark is a poor model of the cost of
  trading that swap, because the mark is an integrated variance of order
  10^-3. That is left for a later cost model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.stats import norm

from deephedge.markets import bs_paths, time_grid

__all__ = [
    "BlackScholes",
    "CostModel",
    "Environment",
    "Market",
    "Paths",
    "ProportionalCost",
    "bs_call_delta",
    "bs_call_price",
    "call_on_spot",
    "terminal_pnl",
]


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


def call_on_spot(instruments: np.ndarray, strike: float) -> np.ndarray:
    """Undiscounted call payoff on the first instrument, shape (m,)."""
    return np.maximum(np.asarray(instruments, dtype=float)[:, -1, 0] - strike, 0.0)


@dataclass(frozen=True)
class Paths:
    """One Monte Carlo batch drawn by a market.

    `time` has shape `(n + 1,)` and runs from 0 to maturity.
    `instruments` has shape `(m, n + 1, d)`.
    `features` has shape `(m, n + 1, f)`.
    `discount` has shape `(n + 1,)`, or `(m, n + 1)` if rates are ever
    stochastic. Entries are factors to time 0.
    """

    time: np.ndarray
    instruments: np.ndarray
    features: np.ndarray
    discount: np.ndarray

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        instruments = np.asarray(self.instruments, dtype=float)
        features = np.asarray(self.features, dtype=float)
        discount = np.asarray(self.discount, dtype=float)
        if instruments.ndim != 3:
            raise ValueError(
                f"instruments must have shape (m, n + 1, d), got {instruments.shape}"
            )
        m, n_plus, _d = instruments.shape
        if features.shape[:2] != (m, n_plus):
            raise ValueError(
                "features must share the path and time axes of instruments, "
                f"got {features.shape} against {instruments.shape}"
            )
        if time.shape != (n_plus,):
            raise ValueError(f"time must have shape ({n_plus},), got {time.shape}")
        if discount.shape != (n_plus,) and discount.shape != (m, n_plus):
            raise ValueError(
                f"discount must have shape ({n_plus},) or ({m}, {n_plus}), "
                f"got {discount.shape}"
            )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "instruments", instruments)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "discount", discount)

    @property
    def spot(self) -> np.ndarray:
        """First instrument, shape (m, n + 1). The underlying of a call."""
        return self.instruments[:, :, 0]

    @property
    def n_steps(self) -> int:
        return int(self.time.size - 1)

    @property
    def n_instruments(self) -> int:
        return int(self.instruments.shape[-1])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[-1])

    def discounted_instruments(self) -> np.ndarray:
        """Instrument prices in time-0 money, shape (m, n + 1, d)."""
        if self.discount.ndim == 1:
            return self.instruments * self.discount[None, :, None]
        return self.instruments * self.discount[:, :, None]

    def discount_terminal(self) -> np.ndarray | float:
        """Discount factor at maturity. A scalar, or shape (m,) if rates vary."""
        if self.discount.ndim == 1:
            return self.discount[-1]
        return self.discount[:, -1]


class Market(ABC):
    """A volatility model and the liability being hedged.

    `simulate` draws instruments and features. `payoff` is the liability
    at maturity, undiscounted. `price_along` is the model price of that
    liability at each date, in time-t money. `benchmark_positions` is the
    hedge in each instrument at `t_0, ..., t_{n-1}`, shape `(m, n, d)`.
    `premium` is the time-0 price in time-0 money.

    Subclasses decide what is tradable and what is merely observed. The P&L
    engine only trades `instruments`.
    """

    maturity: float

    @property
    @abstractmethod
    def n_instruments(self) -> int:
        """How many instruments `simulate` puts on the path."""

    @property
    @abstractmethod
    def n_features(self) -> int:
        """How many features `simulate` puts on the path."""

    @property
    @abstractmethod
    def instrument_names(self) -> tuple[str, ...]:
        """Names of the instrument axis, length `n_instruments`."""

    @property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        """Names of the feature axis, length `n_features`."""

    @abstractmethod
    def simulate(self, m: int, n: int, seed: int) -> Paths:
        """Draw `m` paths on `n` steps over `[0, maturity]`.

        `seed` is passed to `np.random.default_rng`. The same seed and
        the same market give the same paths.
        """

    @abstractmethod
    def payoff(self, paths: Paths) -> np.ndarray:
        """Undiscounted liability at maturity, shape (m,)."""

    @abstractmethod
    def price_along(self, paths: Paths) -> np.ndarray:
        """Model price of the liability at each date, shape (m, n + 1).

        Prices are in time-t money. Multiply by `paths.discount` to get
        the time-0 value, which is a martingale under the pricing measure.
        """

    @abstractmethod
    def benchmark_positions(self, paths: Paths) -> np.ndarray:
        """Benchmark hedge at each trading date, shape (m, n, n_instruments).

        Position k is held over `[t_k, t_{k+1}]`. There is no position at
        maturity: the book is liquidated and the payoff is paid.
        """

    @abstractmethod
    def premium(self) -> float:
        """Price of the liability at t = 0, in time-0 money."""


class CostModel(Protocol):
    """Cost of a book of trades. Pluggable, and independent of the market.

    `prices` has shape `(m, n + 1, d)` and is what the charge is levied
    on. The P&L engine passes discounted prices, so a cash cost paid at `t`
    is already in time-0 money: the proportional cost `c |dq| S_t` becomes
    `c |dq| e^{-rt} S_t`. `positions` has shape `(m, n, d)`, the
    holding on each interval. The book is flat before `t_0` and flat again
    at maturity, and both of those trades are charged.

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
            raise ValueError(
                f"positions have shape {pos.shape}; expected (m, n, {n_instruments})"
            )
        return pos[:, :, None]
    if pos.ndim != 3 or pos.shape[-1] != n_instruments:
        raise ValueError(
            f"positions have shape {pos.shape}; expected (m, n, {n_instruments})"
        )
    return pos


@dataclass(frozen=True)
class ProportionalCost:
    """Proportional cost `rate * sum |change in position| * price`.

    `rate` is a scalar, or one rate per instrument as a tuple of length
    `d`. The opening trade, every rebalance, and the liquidating trade are
    included. This is the cost in Buehler et al. (2019), section 3.

    A later model (a fixed cost, a spread, a cost that depends on the
    sign of the trade) implements `amount` and replaces this object on
    the environment. The market stays put.
    """

    rate: float | tuple[float, ...] = 0.0

    def amount(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        pos = _as_positions(positions, prices.shape[-1])
        if pos.shape[0] != prices.shape[0] or pos.shape[1] != prices.shape[1] - 1:
            raise ValueError(
                f"positions shape {pos.shape} does not match prices shape {prices.shape}"
            )
        rate = np.asarray(self.rate, dtype=float)
        m, _n_plus, d = prices.shape
        flat = np.zeros((m, 1, d))
        full = np.concatenate([flat, pos, flat], axis=1)
        traded = np.abs(np.diff(full, axis=1))
        return (rate * traded * prices).sum(axis=(1, 2))


def as_cost(cost: CostModel | float | tuple[float, ...]) -> CostModel:
    """A `CostModel`, or a proportional rate standing in for one."""
    if isinstance(cost, ProportionalCost):
        return cost
    if isinstance(cost, tuple):
        return ProportionalCost(cost)
    if isinstance(cost, (int, float, np.floating)):
        return ProportionalCost(float(cost))
    if hasattr(cost, "amount"):
        return cost
    raise TypeError(f"cost must be a rate or a CostModel, got {type(cost).__name__}")


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
        raise ValueError(
            f"positions shape {pos.shape} does not match prices shape {prices.shape}"
        )
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


def _check_maturity(paths: Paths, maturity: float) -> None:
    if not np.isclose(paths.time[-1], maturity, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"path ends at {paths.time[-1]}, market maturity is {maturity}"
        )


@dataclass(frozen=True)
class BlackScholes(Market):
    """Black-Scholes market with one instrument, the spot.

    `dS = mu S dt + sigma S dW`, simulated exactly on a uniform grid, so
    the log price has no time-discretisation error. The liability is a
    European call with this strike and maturity. The benchmark hedge is the
    analytic delta in the spot.

    `mu` is the drift of the simulation measure. The price and the delta
    use `r`, the risk-free rate. Set `mu = r` to draw under the pricing
    measure: that is the measure in which the discounted price is a
    martingale. Session 00 uses `mu = r = 0`.

    Features and the instrument are both the spot. A Heston market keeps
    this call on the first instrument and adds the variance as a second
    feature; it does not change the meaning of feature column 0.
    """

    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 30 / 365
    sigma: float = 0.2
    r: float = 0.0
    mu: float = 0.0

    @property
    def n_instruments(self) -> int:
        return 1

    @property
    def n_features(self) -> int:
        return 1

    @property
    def instrument_names(self) -> tuple[str, ...]:
        return ("spot",)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("spot",)

    def simulate(self, m: int, n: int, seed: int) -> Paths:
        """Draw `m` spot paths on `n` steps.

        The Gaussian draw is `bs_paths`, which uses
        `np.random.default_rng(seed)` and consumes one standard normal
        array of shape `(m, n)`. A later market that needs several
        Brownian motions should document the order it draws them from that
        same generator.
        """
        spot = bs_paths(
            m, n, self.maturity, self.sigma, s0=self.s0, mu=self.mu, seed=seed
        )
        return self.paths_from_spot(spot)

    def paths_from_spot(self, spot: np.ndarray) -> Paths:
        """Wrap a spot array of shape `(m, n + 1)` as a path of this market.

        Does not draw. The time grid runs to `maturity`. Discount factors
        use `r`. The feature column is the spot.
        """
        spot = np.asarray(spot, dtype=float)
        if spot.ndim != 2:
            raise ValueError(f"spot must have shape (m, n + 1), got {spot.shape}")
        n = spot.shape[1] - 1
        time = time_grid(n, self.maturity)
        instruments = spot[:, :, None]
        return Paths(
            time=time,
            instruments=instruments,
            features=instruments,
            discount=np.exp(-self.r * time),
        )

    def payoff(self, paths: Paths) -> np.ndarray:
        return call_on_spot(paths.instruments, self.strike)

    def price(self, spot, tau):
        """Black-Scholes call price. Vectorised in spot and in time to maturity."""
        return bs_call_price(spot, self.strike, tau, self.sigma, self.r)

    def delta(self, spot, tau):
        """Black-Scholes call delta. Vectorised in spot and in time to maturity."""
        return bs_call_delta(spot, self.strike, tau, self.sigma, self.r)

    def price_along(self, paths: Paths) -> np.ndarray:
        _check_maturity(paths, self.maturity)
        tau = self.maturity - paths.time
        return self.price(paths.instruments[:, :, 0], tau)

    def benchmark_positions(self, paths: Paths) -> np.ndarray:
        """Analytic delta at `t_0, ..., t_{n-1}`, shape `(m, n, 1)`."""
        _check_maturity(paths, self.maturity)
        tau = self.maturity - paths.time[:-1]
        delta = self.delta(paths.instruments[:, :-1, 0], tau)
        return np.asarray(delta, dtype=float)[..., None]

    def premium(self) -> float:
        return float(np.asarray(self.price(self.s0, self.maturity)))
