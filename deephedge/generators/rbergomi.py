"""Rough Bergomi generator, adapter over `roughvol.rbergomi.RoughBergomi` (M11a).

Needs the `[rough]` extra: `rough-volatility` pinned at tag `v0.1.0` (commit
65f3f99). Spot is `s0 * exp(r t) * Z`, variance sits in path state, and the
second instrument is the variance swap mark from `roughvol.rbergomi.variance_swap`.
The classical benchmark hedge is the Black-Scholes delta at the current
volatility `sqrt(v_t)`, with no Monte Carlo error. Option prices use
`price_with_error` (conditional Monte Carlo) and come with a standard error.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from deephedge.generators.base import Market, call_on_spot
from deephedge.generators.base import check_maturity as _check_maturity
from deephedge.markets import time_grid
from deephedge.paths import Paths
from deephedge.pricing import PriceResult, bs_call_delta

__all__ = ["RoughBergomi", "RoughBergomiPricer", "rbergomi_nu"]


def _roughvol():
    """Import `roughvol` lazily so the package loads without the `[rough]` extra."""
    try:
        import roughvol.estimate as estimate
        import roughvol.rbergomi as rbergomi
    except ImportError as exc:
        raise ImportError(
            "RoughBergomi needs the [rough] extra: "
            'pip install "deep-hedging[rough]" '
            "(pins rough-volatility at tag v0.1.0)"
        ) from exc
    return rbergomi, estimate


def rbergomi_nu(nu_per_day: float, H: float, trading_days: float = 252.0) -> float:
    """Annual rough Bergomi vol of vol from a per-trading-day estimate.

    `nu = nu_per_day * trading_days**H / sqrt(2 H c_H)`, from
    `roughvol.estimate.rbergomi_nu`.
    """
    _rbergomi, estimate = _roughvol()
    return float(estimate.rbergomi_nu(nu_per_day, H, trading_days=trading_days))


def _engine(H, nu, rho, xi0, xi_tenors, n_steps, seed):
    rbergomi, _ = _roughvol()
    return rbergomi.RoughBergomi(
        H=float(H),
        nu=float(nu),
        rho=float(rho),
        xi0=float(xi0) if not callable(xi0) else xi0,
        n_steps=int(n_steps),
        seed=int(seed),
        xi_tenors=None if xi_tenors is None else np.asarray(xi_tenors, dtype=float),
    )


class RoughBergomi(Market):
    """Rough Bergomi market: spot and a variance swap, variance in path state.

    Adapter over `roughvol.rbergomi.RoughBergomi` at tag `v0.1.0`. The
    lab preset is `H = 0.1`, `nu = 1.5`, `rho = -0.7`, `xi0 = 0.04` (D5).
    Spot is `s0 * exp(r t) * Z` with unit-mean `Z`. The variance swap column
    is the remaining integrated forward variance, discounted by `exp(-r tau)`.
    The classical benchmark is the Black-Scholes delta at `sqrt(v_t)`.
    """

    name: ClassVar[str] = "rough_bergomi"
    state_names: ClassVar[tuple[str, ...]] = ("variance", "xi")
    exact_simulation: ClassVar[bool] = False

    def __init__(
        self,
        s0: float = 100.0,
        maturity: float = 30 / 365,
        r: float = 0.0,
        mu: float = 0.0,
        H: float = 0.1,
        nu: float = 1.5,
        rho: float = -0.7,
        xi0: float = 0.04,
        xi_tenors: list[float] | tuple[float, ...] | None = None,
        n_price_paths: int = 4_000,
        price_seed: int = 0,
        strike: float = 100.0,
    ) -> None:
        if s0 <= 0 or maturity <= 0 or strike <= 0:
            raise ValueError("s0, maturity and strike must be positive")
        if not 0.0 < H < 1.0:
            raise ValueError("H must lie in (0, 1)")
        if nu < 0 or xi0 <= 0:
            raise ValueError("nu must be non-negative and xi0 positive")
        if not -1.0 <= rho <= 0.0:
            raise ValueError("rho must lie in [-1, 0]")
        if n_price_paths < 2:
            raise ValueError("n_price_paths must be at least 2")
        self.s0 = float(s0)
        self.maturity = float(maturity)
        self.r = float(r)
        self.mu = float(mu)
        self.H = float(H)
        self.nu = float(nu)
        self.rho = float(rho)
        self.xi0 = float(xi0)
        self.xi_tenors = None if xi_tenors is None else tuple(float(x) for x in xi_tenors)
        self.n_price_paths = int(n_price_paths)
        self.price_seed = int(price_seed)
        self.strike = float(strike)
        if abs(self.mu - self.r) > 1e-15:
            # roughvol draws Z under the pricing measure; a different simulation
            # drift is not wired through yet.
            raise ValueError("RoughBergomi currently requires mu = r (pricing measure)")

    @classmethod
    def describe(cls, params, **context):
        del context
        return {
            "instruments": ("spot", "variance_swap"),
            "state": ("variance", "xi"),
            "has_pricer": True,
        }

    @property
    def n_instruments(self) -> int:
        return 2

    @property
    def n_features(self) -> int:
        return 2

    @property
    def instrument_names(self) -> tuple[str, ...]:
        return ("spot", "variance_swap")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("spot", "variance")

    def params(self) -> dict[str, Any]:
        """Generator parameters for the graph spec. The strike is the contract's."""
        return {
            "s0": self.s0,
            "maturity": self.maturity,
            "r": self.r,
            "mu": self.mu,
            "H": self.H,
            "nu": self.nu,
            "rho": self.rho,
            "xi0": self.xi0,
            "xi_tenors": None if self.xi_tenors is None else list(self.xi_tenors),
            "n_price_paths": self.n_price_paths,
            "price_seed": self.price_seed,
        }

    def pricer(self) -> RoughBergomiPricer:
        return RoughBergomiPricer(self)

    def _model(self, times: np.ndarray, seed: int):
        """roughvol engine with `xi_tenors` equal to the time grid (for the swap)."""
        return _engine(self.H, self.nu, self.rho, self.xi0, times, times.size - 1, seed)

    def _to_paths(self, raw, times: np.ndarray) -> Paths:
        rbergomi, _ = _roughvol()
        spot = self.s0 * np.exp(self.r * times)[None, :] * raw.Z
        swap = rbergomi.variance_swap(raw)
        tau = np.maximum(self.maturity - times, 0.0)
        swap = np.exp(-self.r * tau)[None, :] * swap
        variance = np.asarray(raw.state["v"], dtype=float)
        instruments = np.empty(spot.shape + (2,))
        instruments[:, :, 0] = spot
        instruments[:, :, 1] = swap
        features = np.empty(spot.shape + (2,))
        features[:, :, 0] = spot
        features[:, :, 1] = variance
        state = {
            "variance": variance,
            "xi": np.asarray(raw.state["xi"], dtype=float),
        }
        return Paths(
            time=np.asarray(times, dtype=float),
            instruments=instruments,
            features=features,
            discount=np.exp(-self.r * times),
            state=state,
            instrument_names=self.instrument_names,
            feature_names=self.feature_names,
            source=raw,
        )

    def simulate(self, m: int, n: int, seed: int) -> Paths:
        """Draw `m` paths on `n` steps. `xi_tenors` is the hedging time grid."""
        if m < 1 or n < 1:
            raise ValueError("m and n must be at least 1")
        times = time_grid(n, self.maturity)
        raw = self._model(times, seed).simulate(m, times, seed=seed)
        return self._to_paths(raw, times)

    def simulate_from(
        self, start: Mapping[str, np.ndarray], tau: float, n: int, n_inner: int, seed: int
    ) -> Paths:
        """Restart from `start["spot"]` and `start["variance"]`, each `(p,)`.

        roughvol's `state0` takes a scalar flat forward variance, so each
        outer start is drawn separately with `n_inner` paths. Spot is
        `spot0 * exp(r t) * Z`. The variance swap is over the remaining
        `[0, tau]` only. Drift is `r`.
        """
        spot0 = np.asarray(start["spot"], dtype=float).reshape(-1)
        var0 = np.asarray(start["variance"], dtype=float).reshape(-1)
        if spot0.shape != var0.shape:
            raise ValueError("start spot and variance must have the same shape")
        if tau <= 0 or n < 1 or n_inner < 1:
            raise ValueError("tau must be positive and n, n_inner at least 1")
        return self._simulate_from(spot0, var0, float(tau), n, n_inner, seed)

    def _simulate_from(
        self,
        spot0: np.ndarray,
        var0: np.ndarray,
        tau: float,
        n: int,
        n_inner: int,
        seed: int,
    ) -> Paths:
        times = time_grid(n, tau)
        rbergomi, _ = _roughvol()
        spots = []
        swaps = []
        variances = []
        xis = []
        raws = []
        for i, (s, v) in enumerate(zip(spot0, np.maximum(var0, 1e-16), strict=True)):
            model = rbergomi.RoughBergomi(
                H=self.H,
                nu=self.nu,
                rho=self.rho,
                xi0=float(v),
                n_steps=n,
                seed=seed + 1_000_003 * i,
                xi_tenors=times,
            )
            raw = model.simulate(n_inner, times, seed=seed + 1_000_003 * i, state0={"v": float(v)})
            spot = float(s) * np.exp(self.r * times)[None, :] * raw.Z
            swap = np.exp(-self.r * np.maximum(tau - times, 0.0))[None, :] * rbergomi.variance_swap(
                raw
            )
            spots.append(spot)
            swaps.append(swap)
            variances.append(np.asarray(raw.state["v"], dtype=float))
            xis.append(np.asarray(raw.state["xi"], dtype=float))
            raws.append(raw)
        spot = np.concatenate(spots, axis=0)
        swap = np.concatenate(swaps, axis=0)
        variance = np.concatenate(variances, axis=0)
        xi = np.concatenate(xis, axis=0)
        instruments = np.empty(spot.shape + (2,))
        instruments[:, :, 0] = spot
        instruments[:, :, 1] = swap
        features = np.empty(spot.shape + (2,))
        features[:, :, 0] = spot
        features[:, :, 1] = variance
        return Paths(
            time=np.asarray(times, dtype=float),
            instruments=instruments,
            features=features,
            discount=np.exp(-self.r * times),
            state={"variance": variance, "xi": xi},
            instrument_names=self.instrument_names,
            feature_names=self.feature_names,
            source=raws,
        )

    def payoff(self, paths: Paths) -> np.ndarray:
        return call_on_spot(paths.instruments, self.strike)

    def premium(self) -> float:
        return float(self.pricer().premium(_call(self.strike)).value)

    def price_along(self, paths: Paths) -> np.ndarray:
        return self.pricer().price_along(paths, _call(self.strike))

    def benchmark_positions(self, paths: Paths) -> np.ndarray:
        """Black-Scholes delta at `sqrt(v_t)` in the spot; variance swap weight 0.

        Shape `(m, n, 2)`. Cheap and free of Monte Carlo error (M11a decision).
        """
        _check_maturity(paths, self.maturity)
        m, n_plus, _ = paths.features.shape
        n = n_plus - 1
        positions = np.zeros((m, n, 2))
        spot = paths.features[:, :, 0]
        variance = np.maximum(paths.features[:, :, 1], 0.0)
        for k in range(n):
            tau = self.maturity - paths.time[k]
            sig = np.sqrt(variance[:, k])
            positions[:, k, 0] = bs_call_delta(spot[:, k], self.strike, tau, sig, self.r)
        return positions


def _call(strike: float):
    from deephedge.contracts import Vanilla

    return Vanilla("call", strike)


class RoughBergomiPricer:
    """Conditional Monte Carlo prices via `roughvol.RoughBergomi.price_with_error`."""

    def __init__(self, generator: RoughBergomi):
        self.generator = generator

    def supports(self, contract) -> bool:
        return getattr(contract, "kind", None) in ("call", "put") and hasattr(contract, "strike")

    def _check(self, contract) -> None:
        if not self.supports(contract):
            raise ValueError(f"RoughBergomiPricer prices vanilla calls and puts, got {contract!r}")

    def premium(self, contract) -> PriceResult:
        self._check(contract)
        g = self.generator
        df = float(np.exp(-g.r * g.maturity))
        forward = g.s0 / df if df > 0 else g.s0
        model = _engine(
            g.H, g.nu, g.rho, g.xi0, None, max(4, int(round(g.maturity * 365))), g.price_seed
        )
        result = model.price_with_error(
            forward,
            contract.strike,
            g.maturity,
            df=df,
            kind=contract.kind,
            n_paths=g.n_price_paths,
            seed=g.price_seed,
        )
        return PriceResult(
            value=float(np.asarray(result.value)), stderr=float(np.asarray(result.stderr))
        )

    def price_along(self, paths: Paths, contract) -> np.ndarray:
        """Conditional Monte Carlo price at each date, shape `(m, n + 1)`.

        Each live state is priced with `n_price_paths` paths. Costly; for
        checks and small evaluation sets, not for training.
        """
        self._check(contract)
        g = self.generator
        _check_maturity(paths, g.maturity)
        m, n_plus = paths.spot.shape
        out = np.empty((m, n_plus))
        for k, t in enumerate(paths.time):
            tau = g.maturity - float(t)
            if tau <= 1e-14:
                out[:, k] = contract.terminal_payoff(paths.spot[:, k])
                continue
            df = float(np.exp(-g.r * tau))
            for i in range(m):
                spot = float(paths.spot[i, k])
                forward = spot / df if df > 0 else spot
                v = float(max(paths.state["variance"][i, k], 1e-16))
                model = _engine(
                    g.H, g.nu, g.rho, v, None, max(4, int(round(tau * 365))), g.price_seed + k
                )
                result = model.price_with_error(
                    forward,
                    contract.strike,
                    tau,
                    df=df,
                    kind=contract.kind,
                    n_paths=g.n_price_paths,
                    seed=g.price_seed + 17 * k + i,
                    state={"v": v},
                )
                out[i, k] = float(np.asarray(result.value))
        return out

    def greeks(
        self, paths: Paths, contract, which: tuple[str, ...] = ("delta",)
    ) -> dict[str, np.ndarray]:
        """Black-Scholes delta at `sqrt(v_t)` in the spot; other instruments 0.

        This matches the classical benchmark. No Monte Carlo error on the delta.
        """
        self._check(contract)
        g = self.generator
        unknown = set(which) - {"delta"}
        if unknown:
            raise ValueError(f"RoughBergomiPricer has no greeks {sorted(unknown)}")
        m, n_plus, d = paths.instruments.shape
        delta = np.zeros((m, n_plus, d))
        spot = paths.spot
        variance = np.maximum(paths.state["variance"], 0.0)
        for k, t in enumerate(paths.time):
            tau = g.maturity - float(t)
            sig = np.sqrt(variance[:, k])
            delta[:, k, 0] = bs_call_delta(spot[:, k], contract.strike, tau, sig, g.r)
        return {"delta": delta, "delta_stderr": np.zeros_like(delta)}
