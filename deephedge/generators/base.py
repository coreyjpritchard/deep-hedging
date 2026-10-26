"""The generator and pricer protocols, and the PR 1 `Market` kept as a deprecated base.

A `Generator` is the simulator half of a market: it draws `Paths` and says what
it puts on them. A `Pricer` values a contract on those paths and gives its
greeks. A generator may offer a closed-form pricer through `pricer()`; when it
returns `None` the engine falls back to `deephedge.pricing.MonteCarloPricer`.

`Market` is PR 1's interface, where the market also carried the liability
(`payoff`, `price_along`, `benchmark_positions`, `premium` for one call). It
stays for one release so sessions 00 to 03 import unchanged. `BlackScholes`
and `Heston` implement both: they are `Market` subclasses and conform to
`Generator`. New generators implement `Generator` only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from deephedge.paths import Paths

if TYPE_CHECKING:
    from deephedge.pricing import PriceResult

__all__ = ["Generator", "Market", "Pricer", "call_on_spot", "check_maturity"]


@runtime_checkable
class Generator(Protocol):
    """Draws paths. `params()` is what the graph spec stores for this node.

    `instrument_names`, `feature_names` and `state_names` are what `simulate`
    puts on the returned `Paths`, in order. `simulate(m, n, seed)` uses
    `np.random.default_rng(seed)` and documents its draw order.

    `simulate_from(start, tau, n, n_inner, seed)` restarts the dynamics from
    a state, for nested Monte Carlo pricing: `start` maps names (at least
    `"spot"`, plus the state the dynamics need) to arrays of shape `(p,)`,
    and the result holds `p * n_inner` paths over `[0, tau]`, outer path
    major (rows `i * n_inner` to `(i + 1) * n_inner - 1` start from `i`).
    """

    name: str
    maturity: float

    @property
    def instrument_names(self) -> tuple[str, ...]: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def state_names(self) -> tuple[str, ...]: ...

    def simulate(self, m: int, n: int, seed: int) -> Paths: ...

    def simulate_from(
        self, start: Mapping[str, np.ndarray], tau: float, n: int, n_inner: int, seed: int
    ) -> Paths: ...

    def params(self) -> dict[str, Any]: ...

    def pricer(self) -> Pricer | None: ...


@runtime_checkable
class Pricer(Protocol):
    """Values a contract. Greeks are per instrument: arrays of shape `(m, n + 1, d)`.

    `price_along` is in time-t money, shape `(m, n + 1)`. `premium` is the
    time-0 price with its standard error (0 for a closed form).
    """

    def supports(self, contract: Any) -> bool: ...

    def premium(self, contract: Any) -> PriceResult: ...

    def price_along(self, paths: Paths, contract: Any) -> np.ndarray: ...

    def greeks(
        self, paths: Paths, contract: Any, which: tuple[str, ...] = ("delta",)
    ) -> dict[str, np.ndarray]: ...


def call_on_spot(instruments: np.ndarray, strike: float) -> np.ndarray:
    """Undiscounted call payoff on the first instrument, shape (m,)."""
    return np.maximum(np.asarray(instruments, dtype=float)[:, -1, 0] - strike, 0.0)


def check_maturity(paths: Paths, maturity: float) -> None:
    if not np.isclose(paths.time[-1], maturity, rtol=0.0, atol=1e-12):
        raise ValueError(f"path ends at {paths.time[-1]}, market maturity is {maturity}")


class Market(ABC):
    """Deprecated: a volatility model and the liability being hedged (PR 1).

    `simulate` draws instruments and features. `payoff` is the liability
    at maturity, undiscounted. `price_along` is the model price of that
    liability at each date, in time-t money. `benchmark_positions` is the
    hedge in each instrument at `t_0, ..., t_{n-1}`, shape `(m, n, d)`.
    `premium` is the time-0 price in time-0 money.

    Kept for one release so the sessions import unchanged. New code uses a
    `Generator`, a contract and the generator's `Pricer`.
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
