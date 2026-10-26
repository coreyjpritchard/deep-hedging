"""Heston market: the spot, its variance, and a variance swap.

The scheme is the full truncation Euler method of Lord, Koekkoek and Van
Dijk (2010). The variance used in the drift and the diffusion is
`max(v, 0)`, and the latent variance is allowed to go negative, so the
next step still feels a variance that has crossed zero. The sibling
`roughvol` package uses the same truncation. The parameters of Buehler,
Gonon, Teichmann and Wood (2019), section 5.2, violate the Feller
condition (`2 kappa theta < xi^2`), so a scheme that has to be right
when the variance hits zero is the one shared with later markets.
Quadratic-exponential is more accurate for the variance marginal. It is
a different discretisation from the Euler step rough Bergomi will use,
and the hedging grid below is coarse either way.

Brownian draw. One call,

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((m, n, 2))

in C order. Column 0 is the variance shock. Column 1 is the shock
orthogonal to it. The spot shock is `rho * column 0 + sqrt(1 - rho^2) *
column 1`, applied after the draw. One array, not two sequential calls.

Drifts. The pricing measure uses `r` on the spot and `(kappa, theta)` in
the variance. The simulation uses `mu` on the spot and `(kappa_sim,
theta_sim)` in the variance. `None` means the pricing parameter. The
premium does not read `mu`, `kappa_sim` or `theta_sim`. Defaults are the
2019 experiment, simulated under the pricing measure: `r = mu = 0`,
`kappa = 1`, `theta = v0 = 0.04`, `xi = 2`, `rho = -0.7`, `s0 = K = 100`,
`T = 30 / 365`.

Variance swap. The instrument is the paper's `S^2`, strike 0: the
integrated variance, not that integral divided by `T`. Its time-t price,
in time-t money, is

    exp(-r (T - t)) * (realised integral up to t + L(t, v_t^+))

with

    L(t, v) = (v - theta) / kappa * (1 - exp(-kappa (T - t))) + theta (T - t)

using the pricing `kappa` and `theta`, and `v^+ = max(v, 0)`. The
integral is the left Riemann sum of `v^+`. At maturity `L = 0`, so the
swap equals the realised integral. With `v0 = theta` and `r = 0` the
time-0 price is `theta * T`.

The call price and the benchmark hedge come from the characteristic
function. The premium is an adaptive quadrature of the Heston formula in
the little-trap form (Albrecher, Mayer, Schoutens, Tistaert 2007). Along
a path the price, delta and vega use the COS expansion of Fang and
Oosterlee (2008), which is the same transform evaluated on a cosine
grid. Vega is the derivative with respect to the variance factor, not
with respect to its square root. The benchmark holds delta shares and
`vega / (dP/dv)` variance swaps, the hedge in equation (5.6) of the
paper. `dP/dv` is `exp(-r (T - t)) * dL/dv`.

`xi = 0` is not a limit of that formula (it divides by `xi^2`). The
variance is then deterministic, the log spot is Gaussian with variance
`L`, and the call is the Black-Scholes price with that total variance.


Generator interface (M02). `Heston` is a frozen generator named `"heston"`
with one state, `"variance"` (the truncated `max(v, 0)`, the same column as
feature 1). `params()` is what the graph spec stores, without the strike,
which belongs to the contract. `pricer()` returns a `HestonPricer`, the COS
and quadrature prices for a vanilla call or put. `simulate_from` restarts the
Euler scheme from given spots and variances under the pricing measure, for
`deephedge.pricing.MonteCarloPricer`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from deephedge.generators.base import Market, call_on_spot
from deephedge.generators.base import check_maturity as _check_maturity
from deephedge.markets import time_grid
from deephedge.paths import Paths
from deephedge.pricing import PriceResult, bs_call_delta, bs_call_price

__all__ = [
    "Heston",
    "HestonPricer",
    "cos_call_greeks",
    "d_variance_swap_dv",
    "heston_call_quad",
    "variance_swap_leg",
]


def d_variance_swap_dv(tau, kappa: float):
    """`dL/dv = (1 - exp(-kappa tau)) / kappa`, and `tau` when `kappa` vanishes."""
    tau = np.asarray(tau, dtype=float)
    if abs(kappa) < 1e-12:
        return tau.copy() if isinstance(tau, np.ndarray) else float(tau)
    return (1.0 - np.exp(-kappa * tau)) / kappa


def variance_swap_leg(tau, variance, kappa: float, theta: float):
    """Remaining integrated-variance leg `L(t, v)`, paid notionally at `T`.

    `tau` is time left. `variance` is the current variance factor. This is
    not yet multiplied by the discount from `T` back to `t`.
    """
    variance = np.asarray(variance, dtype=float)
    slope = d_variance_swap_dv(tau, kappa)
    return slope * (variance - theta) + theta * np.asarray(tau, dtype=float)


def _validate(market: Heston) -> None:
    if market.s0 <= 0 or market.strike <= 0:
        raise ValueError("spot and strike must be positive")
    if market.maturity <= 0:
        raise ValueError("maturity must be positive")
    if market.v0 < 0 or market.theta < 0 or market.xi < 0:
        raise ValueError("variance, theta and vol of vol must be non-negative")
    if market.kappa <= 0:
        raise ValueError("pricing kappa must be positive")
    if abs(market.rho) > 1:
        raise ValueError("correlation must lie in [-1, 1]")
    if market.kappa_sim is not None and market.kappa_sim <= 0:
        raise ValueError("simulation kappa must be positive")
    if market.theta_sim is not None and market.theta_sim < 0:
        raise ValueError("simulation theta must be non-negative")
    if market.n_cos < 16:
        raise ValueError("n_cos must be at least 16")


def _little_trap(u, j: int, kappa: float, theta: float, xi: float, rho: float, r: float, tau):
    """`C` and `D` in the little-trap characteristic function, for `P_j`.

    `u` may be complex. Returns `C, D` with the variance and `log S` left
    for the caller, so `phi = exp(C + D v + i u log S)`.
    """
    i = 1j
    a = kappa * theta
    b = kappa - rho * xi if j == 1 else kappa
    u_j = 0.5 if j == 1 else -0.5
    d = np.sqrt((rho * xi * i * u - b) ** 2 - xi**2 * (2 * u_j * i * u - u**2))
    g = (b - rho * xi * i * u - d) / (b - rho * xi * i * u + d)
    e = np.exp(-d * tau)
    log_g = np.log((1.0 - g * e) / (1.0 - g))
    c_coef = b - rho * xi * i * u - d
    capital_c = r * i * u * tau + (a / xi**2) * (c_coef * tau - 2.0 * log_g)
    capital_d = (c_coef / xi**2) * (1.0 - e) / (1.0 - g * e)
    return capital_c, capital_d


def heston_call_quad(
    spot: float,
    variance: float,
    tau: float,
    strike: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    r: float,
) -> tuple[float, float, float]:
    """Call price, delta and vega at one point, by adaptive quadrature.

    Vega is `d(price)/d(variance)`. `xi = 0` returns the Black-Scholes
    values with total variance `L`. Used for the premium and for checks.
    The pathwise hedge uses `cos_call_greeks`, which agrees with this to
    well under a tenth of a volatility point on the 2019 parameters.
    """
    if xi == 0.0:
        price, delta, vega = _black_scholes_from_variance(
            np.asarray(spot),
            np.asarray(variance),
            np.asarray(tau),
            strike,
            kappa,
            theta,
            r,
        )
        return float(price), float(delta), float(vega)
    if tau <= 0.0:
        intrinsic = max(spot - strike, 0.0)
        return intrinsic, float(spot > strike), 0.0

    log_k = np.log(strike)
    log_s = np.log(spot)

    def integrand(u: float, j: int, deriv: bool) -> float:
        capital_c, capital_d = _little_trap(u, j, kappa, theta, xi, rho, r, tau)
        phi = np.exp(capital_c + capital_d * variance + 1j * u * log_s)
        raw = (capital_d * phi if deriv else phi) * np.exp(-1j * u * log_k) / (1j * u)
        return float(np.real(raw))

    probabilities = []
    d_probabilities = []
    for j in (1, 2):
        integral, _ = quad(
            lambda u, j=j: integrand(u, j, False), 1e-8, np.inf, epsabs=1e-8, limit=500
        )
        d_integral, _ = quad(
            lambda u, j=j: integrand(u, j, True), 1e-8, np.inf, epsabs=1e-8, limit=500
        )
        probabilities.append(0.5 + integral / np.pi)
        d_probabilities.append(d_integral / np.pi)
    discount = np.exp(-r * tau)
    price = spot * probabilities[0] - strike * discount * probabilities[1]
    vega = spot * d_probabilities[0] - strike * discount * d_probabilities[1]
    return float(price), float(probabilities[0]), float(vega)


def _black_scholes_from_variance(spot, variance, tau, strike, kappa, theta, r):
    """Black-Scholes call when variance from here on is the deterministic `L`.

    `sigma^2 * tau = L(t, v)`. Delta is `N(d1)`. Vega is the derivative
    with respect to `v`, which is the Black-Scholes vega times `d(sigma)/dv`.
    """
    spot = np.asarray(spot, dtype=float)
    variance = np.maximum(np.asarray(variance, dtype=float), 0.0)
    tau = np.asarray(tau, dtype=float)
    shape = np.broadcast_shapes(spot.shape, variance.shape, tau.shape)
    spot = np.broadcast_to(spot, shape).reshape(-1).copy()
    variance = np.broadcast_to(variance, shape).reshape(-1).copy()
    tau = np.broadcast_to(tau, shape).reshape(-1).copy()
    integrated = np.asarray(variance_swap_leg(tau, variance, kappa, theta), dtype=float).reshape(-1)
    alive = tau > 0
    sigma = np.zeros(spot.shape)
    sigma[alive] = np.sqrt(np.maximum(integrated[alive], 0.0) / tau[alive])
    volatile = alive & (sigma > 1e-12)
    price = np.maximum(spot - strike * np.exp(-r * np.maximum(tau, 0.0)), 0.0)
    delta = (spot > strike).astype(float)
    vega = np.zeros(spot.shape)
    if np.any(volatile):
        sig = sigma[volatile]
        s = spot[volatile]
        t = tau[volatile]
        price[volatile] = bs_call_price(s, strike, t, sig, r)
        delta[volatile] = bs_call_delta(s, strike, t, sig, r)
        sd = sig * np.sqrt(t)
        d1 = (np.log(s / strike) + (r + 0.5 * sig**2) * t) / sd
        slope = d_variance_swap_dv(t, kappa)
        vega[volatile] = s * norm.pdf(d1) * slope / (2.0 * sig * np.sqrt(t))
    return price.reshape(shape), delta.reshape(shape), vega.reshape(shape)


def _cumulants(spot, variance, tau, kappa, theta, xi, rho, r):
    """First two cumulants of `log S`. The `c2` formula corrects Fang and Oosterlee.

    Their published `c2` is wrong, and it can be meaningless when Feller
    fails. This is the expansion in Cui, Kirkby and Nguyen (the correction
    recorded by Le Floc'h). `c1` still carries `log S`.
    """
    variance = np.maximum(variance, 0.0)
    e1 = np.exp(-kappa * tau)
    e2 = np.exp(-2.0 * kappa * tau)
    c1 = (
        np.log(spot) + r * tau + (theta - variance) * (1.0 - e1) / (2.0 * kappa) - 0.5 * theta * tau
    )
    c2 = (variance / (4.0 * kappa**3)) * (
        4.0 * kappa**2 * (1.0 + (rho * xi * tau - 1.0) * e1)
        + kappa * (4.0 * rho * xi * (e1 - 1.0) - 2.0 * xi**2 * tau * e1)
        + xi**2 * (1.0 - e2)
    ) + (theta / (8.0 * kappa**3)) * (
        8.0 * kappa**3 * tau
        - 8.0 * kappa**2 * (1.0 + rho * xi * tau + (rho * xi * tau - 1.0) * e1)
        + 2.0 * kappa * ((1.0 + 2.0 * e1) * xi**2 * tau + 8.0 * (1.0 - e1) * rho * xi)
        + xi**2 * (e2 + 4.0 * e1 - 5.0)
    )
    return c1, c2


def cos_call_greeks(
    spot,
    variance,
    tau,
    strike: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    r: float,
    n_cos: int = 128,
    truncation: float = 12.0,
):
    """Call price, delta and vega on the COS grid. Vectorised in the inputs.

    The truncation interval is `c1 ± L sqrt(max(c2, 0))`, and at least
    `0.3` wide in log price, so a variance sitting on the truncation floor
    still sees the strike. Delta and vega differentiate the characteristic
    function and hold that interval fixed. `xi = 0` is Black-Scholes.
    """
    spot = np.asarray(spot, dtype=float).ravel()
    variance = np.maximum(np.asarray(variance, dtype=float).ravel(), 0.0)
    tau = np.asarray(tau, dtype=float).ravel()
    if spot.shape != variance.shape or spot.shape != tau.shape:
        raise ValueError("spot, variance and tau must share a shape")
    if xi == 0.0:
        return _black_scholes_from_variance(spot, variance, tau, strike, kappa, theta, r)

    m = spot.size
    price = np.empty(m)
    delta = np.empty(m)
    vega = np.empty(m)
    alive = tau > 0
    price[~alive] = np.maximum(spot[~alive] - strike, 0.0)
    delta[~alive] = (spot[~alive] > strike).astype(float)
    vega[~alive] = 0.0
    if not np.any(alive):
        return price, delta, vega

    s = spot[alive]
    v = variance[alive]
    t = tau[alive]
    n_alive = s.size
    c1, c2 = _cumulants(s, v, t, kappa, theta, xi, rho, r)
    half = np.maximum(truncation * np.sqrt(np.maximum(c2, 0.0)), 0.3)
    lower = c1 - half
    upper = c1 + half
    k = np.arange(n_cos)[:, None]
    width = (upper - lower)[None, :]
    frequency = k * np.pi / width
    log_k = np.log(strike)
    left = np.minimum(np.maximum(log_k, lower), upper)
    exp_upper = np.exp(upper)[None, :]
    exp_left = np.exp(left)[None, :]
    span_upper = (upper - lower)[None, :]
    span_left = (left - lower)[None, :]
    chi = (
        np.cos(frequency * span_upper) * exp_upper
        - np.cos(frequency * span_left) * exp_left
        + frequency * np.sin(frequency * span_upper) * exp_upper
        - frequency * np.sin(frequency * span_left) * exp_left
    ) / (1.0 + frequency**2)
    psi = np.empty((n_cos, n_alive))
    psi[0] = upper - left
    if n_cos > 1:
        psi[1:] = (
            np.sin(frequency[1:] * span_upper) - np.sin(frequency[1:] * span_left)
        ) / frequency[1:]
    payoff_coef = (2.0 / (upper - lower))[None, :] * (chi - strike * psi)

    capital_c, capital_d = _little_trap(frequency, 2, kappa, theta, xi, rho, r, t[None, :])
    phi = np.exp(capital_c + capital_d * v[None, :] + 1j * frequency * np.log(s)[None, :])
    series = phi * np.exp(-1j * frequency * lower[None, :])
    weights = np.ones(n_cos)
    weights[0] = 0.5
    discount = np.exp(-r * t)
    weighted = weights[:, None] * payoff_coef
    price[alive] = discount * np.sum(np.real(series) * weighted, axis=0)
    d_series_spot = (1j * frequency / s[None, :]) * series
    delta[alive] = discount * np.sum(np.real(d_series_spot) * weighted, axis=0)
    vega[alive] = discount * np.sum(np.real(capital_d * series) * weighted, axis=0)
    return price, delta, vega


class Heston(Market):
    """Heston market with two instruments, the spot and a variance swap.

    Features are `(spot, variance)`, and the variance column is `max(v, 0)`,
    because the characteristic function is not used on a negative variance.
    The latent variance of the Euler step is not a feature. Feature column
    0 stays the spot, as in Black-Scholes.

    `n_cos` and `truncation` set the COS grid behind `price_along` and
    `benchmark_positions`. The premium is the quadrature price, not the
    COS price. On the 2019 parameters with `n_cos = 128` the two differ
    by about `5e-5` at inception.

    The strike is kept for the deprecated `Market` interface. The generator
    parameters, `params()`, leave it out.
    """

    name: ClassVar[str] = "heston"
    state_names: ClassVar[tuple[str, ...]] = ("variance",)

    def __init__(
        self,
        s0: float = 100.0,
        strike: float = 100.0,
        maturity: float = 30 / 365,
        r: float = 0.0,
        mu: float = 0.0,
        kappa: float = 1.0,
        theta: float = 0.04,
        xi: float = 2.0,
        rho: float = -0.7,
        v0: float = 0.04,
        kappa_sim: float | None = None,
        theta_sim: float | None = None,
        n_cos: int = 128,
        truncation: float = 12.0,
    ):
        self.s0 = float(s0)
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.r = float(r)
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.v0 = float(v0)
        self.kappa_sim = None if kappa_sim is None else float(kappa_sim)
        self.theta_sim = None if theta_sim is None else float(theta_sim)
        self.n_cos = int(n_cos)
        self.truncation = float(truncation)
        _validate(self)

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

    def _kappa_sim(self) -> float:
        return self.kappa if self.kappa_sim is None else self.kappa_sim

    def _theta_sim(self) -> float:
        return self.theta if self.theta_sim is None else self.theta_sim

    def simulate(self, m: int, n: int, seed: int) -> Paths:
        """Draw `m` paths on `n` steps with full truncation Euler.

        The Gaussian sample is one array of shape `(m, n, 2)` from
        `np.random.default_rng(seed)`. Column 0 shocks the variance.
        Column 1 is orthogonal, and the spot uses the correlated mix.
        """
        if m < 1 or n < 1:
            raise ValueError("m and n must be at least 1")
        rng = np.random.default_rng(seed)
        shocks = rng.standard_normal((m, n, 2))
        return self.paths_from_shocks(shocks)

    def paths_from_shocks(self, shocks: np.ndarray) -> Paths:
        """Build paths from a shock array of shape `(m, n, 2)`.

        Column 0 is the variance shock and column 1 the orthogonal shock,
        the same order `simulate` draws. Does not draw.
        """
        z = np.asarray(shocks, dtype=float)
        if z.ndim != 3 or z.shape[-1] != 2:
            raise ValueError(f"shocks must have shape (m, n, 2), got {z.shape}")
        m = z.shape[0]
        return self._euler(
            np.full(m, self.s0),
            np.full(m, self.v0),
            z,
            self.maturity,
            self.mu,
            self._kappa_sim(),
            self._theta_sim(),
        )

    def simulate_from(
        self, start: Mapping[str, np.ndarray], tau: float, n: int, n_inner: int, seed: int
    ) -> Paths:
        """Restart the scheme from `start["spot"]` and `start["variance"]`, each `(p,)`.

        Draws `n_inner` paths per start on `n` steps over `[0, tau]`, under the
        pricing measure (drift `r`, pricing `kappa` and `theta`). One draw
        `default_rng(seed).standard_normal((p * n_inner, n, 2))`, outer path
        major. The variance swap column is the swap over the remaining
        `[0, tau]` only: the variance realised before the restart is not
        known here. It is meant for pricing contracts on the spot.
        """
        spot0 = np.asarray(start["spot"], dtype=float).reshape(-1)
        var0 = np.asarray(start["variance"], dtype=float).reshape(-1)
        if spot0.shape != var0.shape:
            raise ValueError("start spot and variance must have the same shape")
        if tau <= 0 or n < 1 or n_inner < 1:
            raise ValueError("tau must be positive and n, n_inner at least 1")
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((spot0.size * n_inner, n, 2))
        return self._euler(
            np.repeat(spot0, n_inner),
            np.repeat(np.maximum(var0, 0.0), n_inner),
            z,
            float(tau),
            self.r,
            self.kappa,
            self.theta,
        )

    def _euler(self, s0, v0, z, horizon, drift, kappa_s, theta_s) -> Paths:
        """Full truncation Euler from spots `s0 (m,)` and variances `v0 (m,)` over `horizon`."""
        m, n, _ = z.shape
        dt = horizon / n
        time = time_grid(n, horizon)
        log_s = np.empty((m, n + 1))
        log_s[:, 0] = np.log(s0)
        latent = np.array(v0, dtype=float)
        variance = np.empty((m, n + 1))
        variance[:, 0] = v0
        realised = np.zeros(m)
        swap = np.empty((m, n + 1))
        swap[:, 0] = self._swap_price(0.0, v0, horizon)
        orthogonal = float(np.sqrt(max(0.0, 1.0 - self.rho**2)))
        for k in range(n):
            vp = np.maximum(latent, 0.0)
            z_var = z[:, k, 0]
            z_spot = self.rho * z_var + orthogonal * z[:, k, 1]
            log_s[:, k + 1] = log_s[:, k] + (drift - 0.5 * vp) * dt + np.sqrt(vp * dt) * z_spot
            latent = latent + kappa_s * (theta_s - vp) * dt + self.xi * np.sqrt(vp * dt) * z_var
            variance[:, k + 1] = np.maximum(latent, 0.0)
            realised = realised + vp * dt
            left = horizon - time[k + 1]
            swap[:, k + 1] = self._swap_price(realised, variance[:, k + 1], left)
        spot = np.empty_like(log_s)
        spot[:, 0] = s0
        spot[:, 1:] = np.exp(log_s[:, 1:])
        instruments = np.empty((m, n + 1, 2))
        instruments[:, :, 0] = spot
        instruments[:, :, 1] = swap
        features = np.empty((m, n + 1, 2))
        features[:, :, 0] = spot
        features[:, :, 1] = variance
        return Paths(
            time=time,
            instruments=instruments,
            features=features,
            discount=np.exp(-self.r * time),
            state={"variance": variance},
            instrument_names=self.instrument_names,
            feature_names=self.feature_names,
        )

    def params(self) -> dict:
        """Generator parameters for the graph spec. The strike is the contract's."""
        return {
            "s0": self.s0,
            "maturity": self.maturity,
            "r": self.r,
            "mu": self.mu,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "v0": self.v0,
            "kappa_sim": self.kappa_sim,
            "theta_sim": self.theta_sim,
            "n_cos": self.n_cos,
            "truncation": self.truncation,
        }

    def pricer(self) -> HestonPricer:
        return HestonPricer(self)

    def _swap_price(self, realised, variance, tau):
        """Time-t price of the strike-0 variance swap, in time-t money."""
        leg = variance_swap_leg(tau, variance, self.kappa, self.theta)
        return np.exp(-self.r * np.asarray(tau, dtype=float)) * (
            np.asarray(realised, dtype=float) + leg
        )

    def payoff(self, paths: Paths) -> np.ndarray:
        return call_on_spot(paths.instruments, self.strike)

    def greeks(self, spot, variance, tau, strike: float | None = None):
        """Call price, delta and vega. Vectorised. COS, or Black-Scholes if `xi = 0`.

        `strike` defaults to the market's strike.
        """
        spot = np.asarray(spot, dtype=float)
        variance = np.asarray(variance, dtype=float)
        tau = np.asarray(tau, dtype=float)
        shape = np.broadcast_shapes(spot.shape, variance.shape, tau.shape)
        price, delta, vega = cos_call_greeks(
            np.broadcast_to(spot, shape),
            np.broadcast_to(variance, shape),
            np.broadcast_to(tau, shape),
            self.strike if strike is None else float(strike),
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.r,
            n_cos=self.n_cos,
            truncation=self.truncation,
        )
        return price.reshape(shape), delta.reshape(shape), vega.reshape(shape)

    def price_along(self, paths: Paths) -> np.ndarray:
        _check_maturity(paths, self.maturity)
        spot = paths.features[:, :, 0]
        variance = paths.features[:, :, 1]
        price = np.empty(spot.shape)
        for k, t in enumerate(paths.time):
            column, _, _ = self.greeks(spot[:, k], variance[:, k], self.maturity - t)
            price[:, k] = column
        return price

    def benchmark_positions(self, paths: Paths) -> np.ndarray:
        """Delta in the spot and the variance-swap hedge of vega, shape `(m, n, 2)`.

        The swap weight is `vega / (dP/dv)`, with `P` the swap's time-t
        price. `dP/dv = exp(-r tau) * dL/dv`. Near expiry that slope is
        `tau`, so the weight can be large: the swap has little time left
        in which a change in variance can accumulate. Below `1e-12` the
        weight is set to 0.
        """
        _check_maturity(paths, self.maturity)
        m, n_plus, _ = paths.features.shape
        n = n_plus - 1
        positions = np.empty((m, n, 2))
        spot = paths.features[:, :, 0]
        variance = paths.features[:, :, 1]
        for k in range(n):
            tau = self.maturity - paths.time[k]
            _, delta, vega = self.greeks(spot[:, k], variance[:, k], tau)
            slope = np.exp(-self.r * tau) * float(np.asarray(d_variance_swap_dv(tau, self.kappa)))
            positions[:, k, 0] = delta
            if slope < 1e-12:
                positions[:, k, 1] = 0.0
            else:
                positions[:, k, 1] = vega / slope
        return positions

    def premium(self) -> float:
        """Quadrature price of the call at `t = 0`. Does not use the simulation drifts."""
        price, _, _ = heston_call_quad(
            self.s0,
            self.v0,
            self.maturity,
            self.strike,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.r,
        )
        return price

    def quad_greeks(self) -> tuple[float, float, float]:
        """Quadrature price, delta and vega at inception. The check against COS."""
        return heston_call_quad(
            self.s0,
            self.v0,
            self.maturity,
            self.strike,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.r,
        )


class HestonPricer:
    """Closed-form prices for a vanilla call or put on a `Heston` generator.

    The premium is the quadrature price. Along a path the price, delta and
    vega are COS (`n_cos`, `truncation` from the generator); a put comes from
    put-call parity. Greeks per instrument, shape `(m, n + 1, 2)`:

    * `"delta"`: column 0 is the COS delta at fixed variance, column 1 the
      variance-swap weight `vega / (dP/dv)`, the hedge of equation (5.6) in
      Buehler et al. (2019). Rows `0..n-1` equal `Heston.benchmark_positions`;
      at maturity the spot column is the indicator and the swap column 0.
    * `"gamma"`: column 0 is a central difference of the COS delta in the
      spot (relative bump `1e-4`), column 1 is 0.
    * `"vega"`: shape `(m, n + 1)`, the derivative in the variance factor.
    """

    def __init__(self, generator: Heston):
        self.generator = generator

    def supports(self, contract) -> bool:
        return getattr(contract, "kind", None) in ("call", "put") and hasattr(contract, "strike")

    def _check(self, contract) -> None:
        if not self.supports(contract):
            raise ValueError(f"HestonPricer prices vanilla calls and puts, got {contract!r}")

    def premium(self, contract) -> PriceResult:
        self._check(contract)
        g = self.generator
        call, _, _ = heston_call_quad(
            g.s0, g.v0, g.maturity, contract.strike, g.kappa, g.theta, g.xi, g.rho, g.r
        )
        value = call
        if contract.kind == "put":
            value = call - g.s0 + contract.strike * np.exp(-g.r * g.maturity)
        return PriceResult(value=float(value), stderr=0.0)

    def _columns(self, paths: Paths, contract):
        self._check(contract)
        g = self.generator
        _check_maturity(paths, g.maturity)
        spot = paths.features[:, :, 0]
        variance = paths.features[:, :, 1]
        price = np.empty(spot.shape)
        delta = np.empty(spot.shape)
        vega = np.empty(spot.shape)
        for k, t in enumerate(paths.time):
            tau = g.maturity - t
            price[:, k], delta[:, k], vega[:, k] = g.greeks(
                spot[:, k], variance[:, k], tau, contract.strike
            )
        if contract.kind == "put":
            tau = g.maturity - paths.time
            price = price - spot + contract.strike * np.exp(-g.r * tau)[None, :]
            delta = delta - 1.0
        return spot, variance, price, delta, vega

    def price_along(self, paths: Paths, contract) -> np.ndarray:
        return self._columns(paths, contract)[2]

    def greeks(self, paths: Paths, contract, which=("delta",)) -> dict[str, np.ndarray]:
        unknown = set(which) - {"delta", "gamma", "vega"}
        if unknown:
            raise ValueError(f"HestonPricer has no greeks {sorted(unknown)}")
        g = self.generator
        spot, variance, _, delta, vega = self._columns(paths, contract)
        out: dict[str, np.ndarray] = {}
        if "delta" in which:
            per = np.zeros(spot.shape + (2,))
            per[:, :, 0] = delta
            for k, t in enumerate(paths.time):
                tau = g.maturity - t
                slope = np.exp(-g.r * tau) * float(np.asarray(d_variance_swap_dv(tau, g.kappa)))
                if slope >= 1e-12:
                    per[:, k, 1] = vega[:, k] / slope
            out["delta"] = per
        if "gamma" in which:
            per = np.zeros(spot.shape + (2,))
            h = 1e-4 * spot
            for k, t in enumerate(paths.time):
                tau = g.maturity - t
                _, up, _ = g.greeks(spot[:, k] + h[:, k], variance[:, k], tau, contract.strike)
                _, down, _ = g.greeks(spot[:, k] - h[:, k], variance[:, k], tau, contract.strike)
                per[:, k, 0] = (up - down) / (2.0 * h[:, k])
            out["gamma"] = per
        if "vega" in which:
            out["vega"] = vega
        return out
