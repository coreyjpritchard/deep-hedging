"""Pricing: Black-Scholes formulas, `PriceResult` and the nested Monte Carlo pricer.

`MonteCarloPricer` is the fallback for a generator without a closed form, and
the check on the ones that have one. It prices a contract on the spot by
restarting the generator from each path's state at each date
(`Generator.simulate_from`), under the pricing measure, and averaging the
discounted payoff over `n_inner` inner paths. Every value comes with its
standard error. Delta and gamma are central differences in the spot with
common random numbers: the bumped and unbumped inner paths share one draw.

Cost. One `price_along` on `m` outer paths and `n` dates simulates
`m * n * n_inner` inner paths. It is meant for checks and for small
evaluation sets, not for training. Results are cached on an explicit
`cache_key` (the engine passes `(spec_id, split)`), never on object identity.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from deephedge.paths import Paths

__all__ = [
    "MonteCarloPricer",
    "PriceResult",
    "bs_call_delta",
    "bs_call_gamma",
    "bs_call_price",
    "bs_call_vega",
]


@dataclass(frozen=True)
class PriceResult:
    """A price and its standard error. Arrays or floats; 0 for a closed form."""

    value: Any
    stderr: Any


def _d1(S, K, tau, sigma, r):
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.where(tau > 0, tau, 1.0)
    sd = sigma * np.sqrt(safe_tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * safe_tau) / sd
    return S, tau, safe_tau, sd, d1


def bs_call_price(S, K: float, tau, sigma: float, r: float = 0.0):
    """Black-Scholes call price with time to maturity tau. Vectorised in S and tau."""
    S, tau, safe_tau, sd, d1 = _d1(S, K, tau, sigma, r)
    intrinsic = np.maximum(S - K, 0.0)
    d2 = d1 - sd
    price = S * norm.cdf(d1) - K * np.exp(-r * safe_tau) * norm.cdf(d2)
    return np.where(tau > 0, price, intrinsic)


def bs_call_delta(S, K: float, tau, sigma: float, r: float = 0.0):
    """Black-Scholes call delta N(d1). At tau = 0 returns the indicator S > K."""
    S, tau, _, _, d1 = _d1(S, K, tau, sigma, r)
    return np.where(tau > 0, norm.cdf(d1), (S > K).astype(float))


def bs_call_gamma(S, K: float, tau, sigma: float, r: float = 0.0):
    """Black-Scholes gamma `n(d1) / (S sigma sqrt(tau))`. 0 at tau = 0."""
    S, tau, _, sd, d1 = _d1(S, K, tau, sigma, r)
    return np.where(tau > 0, norm.pdf(d1) / (S * sd), 0.0)


def bs_call_vega(S, K: float, tau, sigma: float, r: float = 0.0):
    """Black-Scholes vega `S n(d1) sqrt(tau)`, per unit of sigma. 0 at tau = 0."""
    S, tau, safe_tau, _, d1 = _d1(S, K, tau, sigma, r)
    return np.where(tau > 0, S * norm.pdf(d1) * np.sqrt(safe_tau), 0.0)


def _seed(seed: int, *salt: int) -> int:
    """A child seed for one date or one bump, independent of the others."""
    return int(np.random.SeedSequence((seed, *salt)).generate_state(1)[0])


class MonteCarloPricer:
    """Nested Monte Carlo prices and greeks for a terminal contract on the spot.

    `generator` is any `Generator`. `n_inner` inner paths per outer state.
    `dt_max` caps the inner step, so an Euler generator is priced on a fine
    grid (the default, 1/5840 of a year, is 16 steps a day, 480 steps on 30
    days). A generator with `exact_simulation = True` (Black-Scholes) is
    stepped once to the horizon instead, since a terminal payoff needs only
    the terminal spot. `bump` is the relative spot bump for delta and gamma.
    The default, 0.2%, keeps the central-difference bias below the Monte
    Carlo error on the 2019 Heston parameters, where a 1% bump biases the
    delta by about 4 standard errors at 20 000 inner paths; gamma's error
    grows as `1 / bump`. `seed`
    fixes every inner draw: date `k` uses a child seed of `(seed, k)`, and
    the bumped valuations at that date reuse it. `max_floats` bounds the
    inner array, by splitting the outer paths into chunks.

    Contracts need `terminal_payoff(spot_T) -> (m,)`, the undiscounted payoff
    from the terminal spot. Path-dependent contracts need the history and
    are not supported here.
    """

    def __init__(
        self,
        generator,
        n_inner: int = 10_000,
        seed: int = 0,
        dt_max: float = 1.0 / 5840.0,
        bump: float = 0.002,
        max_floats: int = 20_000_000,
    ):
        if n_inner < 2:
            raise ValueError("n_inner must be at least 2 for a standard error")
        if dt_max <= 0 or bump <= 0:
            raise ValueError("dt_max and bump must be positive")
        self.generator = generator
        self.n_inner = int(n_inner)
        self.seed = int(seed)
        self.dt_max = float(dt_max)
        self.bump = float(bump)
        self.max_floats = int(max_floats)
        self._cache: dict[Hashable, Any] = {}
        self.n_inner_simulations = 0

    def supports(self, contract) -> bool:
        return callable(getattr(contract, "terminal_payoff", None))

    def _check(self, contract) -> None:
        if not self.supports(contract):
            raise ValueError(
                f"MonteCarloPricer needs a contract with terminal_payoff(spot_T), got {contract!r}"
            )

    def _steps(self, tau: float) -> int:
        if getattr(self.generator, "exact_simulation", False):
            return 1
        return max(1, math.ceil(tau / self.dt_max - 1e-9))

    def _value(self, start: dict[str, np.ndarray], tau: float, contract, seed: int):
        """Price in time-t money and its standard error, each `(p,)`."""
        spot = start["spot"]
        if tau <= 1e-14:
            return contract.terminal_payoff(spot), np.zeros(spot.shape)
        n = self._steps(tau)
        p = spot.size
        per_chunk = max(1, self.max_floats // (self.n_inner * (n + 1) * 4))
        value = np.empty(p)
        stderr = np.empty(p)
        for lo in range(0, p, per_chunk):
            hi = min(p, lo + per_chunk)
            part = {k: v[lo:hi] for k, v in start.items()}
            inner = self.generator.simulate_from(part, tau, n, self.n_inner, _seed(seed, lo))
            self.n_inner_simulations += inner.n_paths
            disc = inner.discount_terminal()
            pay = (disc * contract.terminal_payoff(inner.spot[:, -1])).reshape(hi - lo, -1)
            value[lo:hi] = pay.mean(axis=1)
            stderr[lo:hi] = pay.std(axis=1, ddof=1) / math.sqrt(self.n_inner)
        return value, stderr

    def _start(self, paths: Paths, k: int, spot_scale: float = 1.0) -> dict[str, np.ndarray]:
        start = {"spot": paths.spot[:, k] * spot_scale}
        for name in self.generator.state_names:
            if name not in paths.state:
                raise ValueError(f"paths have no state {name!r}, the generator needs it")
            column = paths.state[name]
            start[name] = column[:, k]
        return start

    def _initial(self) -> Paths:
        return self.generator.simulate(1, 1, seed=self.seed)

    def premium(self, contract) -> PriceResult:
        """Time-0 price in time-0 money, with its standard error."""
        self._check(contract)
        first = self._initial()
        value, stderr = self._value(
            self._start(first, 0), self.generator.maturity, contract, _seed(self.seed, 0)
        )
        return PriceResult(value=float(value[0]), stderr=float(stderr[0]))

    def price_along_with_error(
        self, paths: Paths, contract, cache_key: Hashable | None = None
    ) -> PriceResult:
        """Prices `(m, n + 1)` in time-t money and their standard errors."""
        self._check(contract)
        key = None if cache_key is None else ("price", cache_key, repr(contract))
        if key is not None and key in self._cache:
            return self._cache[key]
        value = np.empty(paths.spot.shape)
        stderr = np.empty(paths.spot.shape)
        for k, t in enumerate(paths.time):
            tau = self.generator.maturity - float(t)
            value[:, k], stderr[:, k] = self._value(
                self._start(paths, k), tau, contract, _seed(self.seed, k)
            )
        result = PriceResult(value=value, stderr=stderr)
        if key is not None:
            self._cache[key] = result
        return result

    def price_along(self, paths: Paths, contract, cache_key: Hashable | None = None) -> np.ndarray:
        return self.price_along_with_error(paths, contract, cache_key).value

    def greeks(
        self,
        paths: Paths,
        contract,
        which: tuple[str, ...] = ("delta",),
        cache_key: Hashable | None = None,
    ) -> dict[str, np.ndarray]:
        """`"delta"` and `"gamma"` in the spot, each `(m, n + 1, d)`, with `"<name>_stderr"`.

        Only the spot column is filled; the other instruments get 0. A
        generator with a second tradable (a variance swap) needs its own
        pricer for the hedge in it.
        """
        self._check(contract)
        unknown = set(which) - {"delta", "gamma"}
        if unknown:
            raise ValueError(f"MonteCarloPricer has no greeks {sorted(unknown)}")
        key = None if cache_key is None else ("greeks", cache_key, repr(contract), tuple(which))
        if key is not None and key in self._cache:
            return self._cache[key]
        m, n_plus, d = paths.instruments.shape
        delta = np.zeros((m, n_plus, d))
        gamma = np.zeros((m, n_plus, d))
        delta_se = np.zeros((m, n_plus, d))
        gamma_se = np.zeros((m, n_plus, d))
        h = self.bump
        for k, t in enumerate(paths.time):
            tau = self.generator.maturity - float(t)
            seed = _seed(self.seed, k)
            spot = paths.spot[:, k]
            if tau <= 1e-14:
                up = contract.terminal_payoff(spot * (1 + h))
                down = contract.terminal_payoff(spot * (1 - h))
                mid = contract.terminal_payoff(spot)
                delta[:, k, 0] = (up - down) / (2 * h * spot)
                gamma[:, k, 0] = (up - 2 * mid + down) / (h * spot) ** 2
                continue
            diff, diff_se, curv, curv_se = self._bumped(
                paths, k, tau, contract, seed, "gamma" in which
            )
            delta[:, k, 0] = diff / (2 * h * spot)
            delta_se[:, k, 0] = diff_se / (2 * h * spot)
            gamma[:, k, 0] = curv / (h * spot) ** 2
            gamma_se[:, k, 0] = curv_se / (h * spot) ** 2
        out: dict[str, np.ndarray] = {}
        if "delta" in which:
            out["delta"], out["delta_stderr"] = delta, delta_se
        if "gamma" in which:
            out["gamma"], out["gamma_stderr"] = gamma, gamma_se
        if key is not None:
            self._cache[key] = out
        return out

    def _bumped(self, paths: Paths, k: int, tau: float, contract, seed: int, want_gamma: bool):
        """Per-path differences `V(S+h) - V(S-h)` and `V(S+h) - 2V(S) + V(S-h)`.

        The three valuations share the inner draw. The standard errors are of
        the per-inner-path differences, so the common noise cancels in them.
        """
        h = self.bump
        n = self._steps(tau)
        p = paths.n_paths
        per_chunk = max(1, self.max_floats // (self.n_inner * (n + 1) * 4 * 3))
        diff = np.empty(p)
        diff_se = np.empty(p)
        curv = np.zeros(p)
        curv_se = np.zeros(p)
        for lo in range(0, p, per_chunk):
            hi = min(p, lo + per_chunk)
            rows = slice(lo, hi)
            pays = []
            scales = (1 + h, 1 - h, 1.0) if want_gamma else (1 + h, 1 - h)
            for scale in scales:
                start = {key: v[rows] for key, v in self._start(paths, k, scale).items()}
                inner = self.generator.simulate_from(start, tau, n, self.n_inner, _seed(seed, lo))
                self.n_inner_simulations += inner.n_paths
                disc = inner.discount_terminal()
                pay = disc * contract.terminal_payoff(inner.spot[:, -1])
                pays.append(pay.reshape(hi - lo, -1))
            first = pays[0] - pays[1]
            diff[rows] = first.mean(axis=1)
            diff_se[rows] = first.std(axis=1, ddof=1) / math.sqrt(self.n_inner)
            if want_gamma:
                second = pays[0] - 2 * pays[2] + pays[1]
                curv[rows] = second.mean(axis=1)
                curv_se[rows] = second.std(axis=1, ddof=1) / math.sqrt(self.n_inner)
        return diff, diff_se, curv, curv_se
