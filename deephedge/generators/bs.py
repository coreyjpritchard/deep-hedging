"""Black-Scholes generator, and its closed-form pricer for vanilla calls and puts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from deephedge.generators.base import Market, call_on_spot
from deephedge.generators.base import check_maturity as _check_maturity
from deephedge.markets import bs_paths, time_grid
from deephedge.paths import Paths
from deephedge.pricing import (
    PriceResult,
    bs_call_delta,
    bs_call_gamma,
    bs_call_price,
    bs_call_vega,
)

__all__ = ["BlackScholes", "BlackScholesPricer"]


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

    As a generator (M02) it is named `"black_scholes"`, has no state, and
    `params()` leaves out the strike, which belongs to the contract. The
    strike stays here for the deprecated `Market` interface.
    """

    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 30 / 365
    sigma: float = 0.2
    r: float = 0.0
    mu: float = 0.0

    name: ClassVar[str] = "black_scholes"
    state_names: ClassVar[tuple[str, ...]] = ()
    exact_simulation: ClassVar[bool] = True

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

    def params(self) -> dict:
        """Generator parameters for the graph spec. The strike is the contract's."""
        return {
            "s0": self.s0,
            "maturity": self.maturity,
            "sigma": self.sigma,
            "r": self.r,
            "mu": self.mu,
        }

    def pricer(self) -> BlackScholesPricer:
        return BlackScholesPricer(self)

    def simulate(self, m: int, n: int, seed: int) -> Paths:
        """Draw `m` spot paths on `n` steps.

        The Gaussian draw is `bs_paths`, which uses
        `np.random.default_rng(seed)` and consumes one standard normal
        array of shape `(m, n)`. A later market that needs several
        Brownian motions should document the order it draws them from that
        same generator.
        """
        spot = bs_paths(m, n, self.maturity, self.sigma, s0=self.s0, mu=self.mu, seed=seed)
        return self.paths_from_spot(spot)

    def simulate_from(
        self, start: Mapping[str, np.ndarray], tau: float, n: int, n_inner: int, seed: int
    ) -> Paths:
        """`n_inner` exact paths per start spot over `[0, tau]`, under the pricing measure.

        One draw `default_rng(seed).standard_normal((p * n_inner, n))`, outer
        path major. The drift is `r`, not `mu`.
        """
        spot0 = np.asarray(start["spot"], dtype=float).reshape(-1)
        if tau <= 0 or n < 1 or n_inner < 1:
            raise ValueError("tau must be positive and n, n_inner at least 1")
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((spot0.size * n_inner, n))
        dt = tau / n
        steps = (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
        log_s = np.empty((z.shape[0], n + 1))
        log_s[:, 0] = np.log(np.repeat(spot0, n_inner))
        np.cumsum(steps, axis=1, out=log_s[:, 1:])
        log_s[:, 1:] += log_s[:, :1]
        spot = np.exp(log_s)
        spot[:, 0] = np.repeat(spot0, n_inner)
        time = time_grid(n, tau)
        instruments = spot[:, :, None]
        return Paths(
            time=time,
            instruments=instruments,
            features=instruments,
            discount=np.exp(-self.r * time),
            instrument_names=self.instrument_names,
            feature_names=self.feature_names,
        )

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
            instrument_names=self.instrument_names,
            feature_names=self.feature_names,
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


class BlackScholesPricer:
    """Closed-form prices and greeks of a vanilla call or put on a `BlackScholes` generator.

    Puts come from put-call parity. Greeks: `"delta"` and `"gamma"` have
    shape `(m, n + 1, 1)`, one column per instrument; `"vega"` is the
    derivative in sigma, shape `(m, n + 1)`. At maturity the delta is the
    payoff's indicator and gamma and vega are 0.
    """

    def __init__(self, generator: BlackScholes):
        self.generator = generator

    def supports(self, contract) -> bool:
        return getattr(contract, "kind", None) in ("call", "put") and hasattr(contract, "strike")

    def _check(self, contract) -> None:
        if not self.supports(contract):
            raise ValueError(f"BlackScholesPricer prices vanilla calls and puts, got {contract!r}")

    def _price(self, spot, tau, contract):
        g = self.generator
        call = bs_call_price(spot, contract.strike, tau, g.sigma, g.r)
        if contract.kind == "call":
            return call
        return call - spot + contract.strike * np.exp(-g.r * np.asarray(tau, dtype=float))

    def premium(self, contract) -> PriceResult:
        self._check(contract)
        g = self.generator
        return PriceResult(value=float(self._price(g.s0, g.maturity, contract)), stderr=0.0)

    def price_along(self, paths: Paths, contract) -> np.ndarray:
        self._check(contract)
        _check_maturity(paths, self.generator.maturity)
        tau = (self.generator.maturity - paths.time)[None, :]
        return self._price(paths.spot, tau, contract)

    def greeks(self, paths: Paths, contract, which=("delta",)) -> dict[str, np.ndarray]:
        self._check(contract)
        unknown = set(which) - {"delta", "gamma", "vega"}
        if unknown:
            raise ValueError(f"BlackScholesPricer has no greeks {sorted(unknown)}")
        g = self.generator
        _check_maturity(paths, g.maturity)
        tau = (g.maturity - paths.time)[None, :]
        spot = paths.spot
        out: dict[str, np.ndarray] = {}
        if "delta" in which:
            delta = bs_call_delta(spot, contract.strike, tau, g.sigma, g.r)
            if contract.kind == "put":
                at_expiry = np.broadcast_to(tau <= 0, spot.shape)
                delta = np.where(at_expiry, -(spot < contract.strike).astype(float), delta - 1.0)
            out["delta"] = np.asarray(delta, dtype=float)[..., None]
        if "gamma" in which:
            out["gamma"] = bs_call_gamma(spot, contract.strike, tau, g.sigma, g.r)[..., None]
        if "vega" in which:
            out["vega"] = bs_call_vega(spot, contract.strike, tau, g.sigma, g.r)
        return out
