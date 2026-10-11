"""A market you can edit and upload.

The playground executes this file on your computer. It is not sandboxed.
Upload a file you trust, and use the app locally. The static web page
does not run uploads.

`build_market` returns a `Market`. The class below is a Merton
jump-diffusion: a Black-Scholes diffusion plus a compound Poisson jump.
A regime-switching volatility is the same kind of file. Draw the regime
from the generator, write down the order of the draws, and implement the
same methods. On upload the playground checks the shapes and a
martingale condition on the discounted spot.

Draw order, from `np.random.default_rng(seed)`, and no other draws:

1. `standard_normal((m, n))`, the Brownian increments.
2. `poisson(intensity * dt, size=(m, n))`, the jump counts.
3. `standard_normal((m, n))`, the jump-size shocks.

Under the pricing measure the spot drift is `r` after the jump
compensator, so the discounted spot is a martingale (up to sampling
error). The line marked `sim_drift` is where a physical drift would go.
A drift different from `r` fails the martingale check, which is the
right outcome.

The playground's Delta button is a single Black-Scholes delta, using
`sigma` below, so the same formula is used on every card.
`benchmark_positions` here is the Merton mixture of Black-Scholes
deltas. The upload check only looks at its shape.
"""

from __future__ import annotations

import math

import numpy as np

from deephedge.env import Market, Paths, bs_call_delta, bs_call_price, call_on_spot
from deephedge.markets import time_grid

# How many Poisson terms to keep in the call price. The check below
# raises if the leftover tail is still large.
N_TERMS = 40


def mean_jump_excess(mu_j: float, sigma_j: float) -> float:
    """E[J - 1] for J = exp(mu_j + sigma_j Z), Z standard normal."""
    return math.exp(mu_j + 0.5 * sigma_j**2) - 1.0


def poisson_weights(mean: float, n_terms: int) -> np.ndarray:
    """Poisson probabilities for 0, ..., n_terms - 1. The tail must be tiny."""
    if mean < 0.0:
        raise ValueError("Poisson mean must be non-negative.")
    weights = np.zeros(n_terms)
    if mean == 0.0:
        weights[0] = 1.0
        return weights
    log_term = -mean
    total = 0.0
    for n in range(n_terms):
        weights[n] = math.exp(log_term)
        total += weights[n]
        log_term += math.log(mean) - math.log(n + 1)
    if total < 1.0 - 1e-8:
        raise ValueError(
            f"Merton series kept probability {total:.6f} with {n_terms} terms. "
            "Raise N_TERMS, or lower the jump intensity."
        )
    return weights


def merton_call(spot, strike: float, tau: float, r: float, sigma: float,
                 intensity: float, mu_j: float, sigma_j: float):
    """Merton (1976) call price. `spot` may be an array. `tau` is one number.

    The sum is Black-Scholes prices under a Poisson number of jumps.
    lambda' = intensity * (1 + m), with m = E[J - 1].
    sigma_n^2 = sigma^2 + n * sigma_j^2 / tau.
    r_n = r - intensity * m + n * log(1 + m) / tau.
    """
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return np.maximum(spot - strike, 0.0)
    excess = mean_jump_excess(mu_j, sigma_j)
    tilted = intensity * (1.0 + excess)
    weights = poisson_weights(tilted * tau, N_TERMS)
    total = np.zeros_like(spot, dtype=float)
    for n, weight in enumerate(weights):
        if weight == 0.0:
            continue
        sigma_n = math.sqrt(sigma**2 + n * sigma_j**2 / tau)
        r_n = r - intensity * excess + n * math.log(1.0 + excess) / tau
        total = total + weight * bs_call_price(spot, strike, tau, sigma_n, r_n)
    return total


def merton_delta(spot, strike: float, tau: float, r: float, sigma: float,
                  intensity: float, mu_j: float, sigma_j: float):
    """Spot derivative of `merton_call`. Same weights, Black-Scholes deltas."""
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return (spot > strike).astype(float)
    excess = mean_jump_excess(mu_j, sigma_j)
    tilted = intensity * (1.0 + excess)
    weights = poisson_weights(tilted * tau, N_TERMS)
    total = np.zeros_like(spot, dtype=float)
    for n, weight in enumerate(weights):
        if weight == 0.0:
            continue
        sigma_n = math.sqrt(sigma**2 + n * sigma_j**2 / tau)
        r_n = r - intensity * excess + n * math.log(1.0 + excess) / tau
        total = total + weight * bs_call_delta(spot, strike, tau, sigma_n, r_n)
    return total


class MertonJump(Market):
    """Jump-diffusion call. Edit the numbers in `build_market`."""

    def __init__(
        self,
        s0: float = 100.0,
        strike: float = 100.0,
        maturity: float = 30 / 365,
        r: float = 0.0,
        sigma: float = 0.2,
        intensity: float = 1.0,
        mu_j: float = -0.1,
        sigma_j: float = 0.2,
    ):
        self.s0 = float(s0)
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.r = float(r)
        self.sigma = float(sigma)
        self.intensity = float(intensity)
        self.mu_j = float(mu_j)
        self.sigma_j = float(sigma_j)
        if self.s0 <= 0.0 or self.strike <= 0.0:
            raise ValueError("spot and strike must be positive.")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if self.sigma < 0.0 or self.sigma_j < 0.0 or self.intensity < 0.0:
            raise ValueError("sigma, sigma_j and intensity must be non-negative.")

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
        """Exact step of the compensated jump-diffusion on a uniform grid.

        The continuous part is the usual log-Euler, which is exact for a
        diffusion between jumps. The jump in the log is
        `count * mu_j + sigma_j * sqrt(count) * z_j`. A count of zero
        adds nothing. `sim_drift` equals `r`, so the mean of the
        discounted spot is `s0`.
        """
        if m < 1 or n < 1:
            raise ValueError("m and n must be at least 1.")
        rng = np.random.default_rng(seed)
        dt = self.maturity / n
        brownian = rng.standard_normal((m, n))
        counts = rng.poisson(self.intensity * dt, size=(m, n))
        jump_shock = rng.standard_normal((m, n))
        excess = mean_jump_excess(self.mu_j, self.sigma_j)
        sim_drift = self.r
        increments = (
            (sim_drift - self.intensity * excess - 0.5 * self.sigma**2) * dt
            + self.sigma * math.sqrt(dt) * brownian
            + counts * self.mu_j
            + self.sigma_j * np.sqrt(counts) * jump_shock
        )
        log_s = np.empty((m, n + 1))
        log_s[:, 0] = np.log(self.s0)
        np.cumsum(increments, axis=1, out=log_s[:, 1:])
        log_s[:, 1:] += np.log(self.s0)
        spot = np.exp(log_s)
        spot[:, 0] = self.s0
        time = time_grid(n, self.maturity)
        column = spot[:, :, None]
        return Paths(
            time=time,
            instruments=column,
            features=column,
            discount=np.exp(-self.r * time),
        )

    def payoff(self, paths: Paths) -> np.ndarray:
        return call_on_spot(paths.instruments, self.strike)

    def price_along(self, paths: Paths) -> np.ndarray:
        tau = self.maturity - paths.time
        out = np.empty(paths.spot.shape)
        for k in range(paths.spot.shape[1]):
            out[:, k] = merton_call(
                paths.spot[:, k], self.strike, float(tau[k]), self.r, self.sigma,
                self.intensity, self.mu_j, self.sigma_j,
            )
        return out

    def benchmark_positions(self, paths: Paths) -> np.ndarray:
        """Merton mixture delta at each trading date, shape (m, n, 1)."""
        tau = self.maturity - paths.time[:-1]
        out = np.empty((paths.spot.shape[0], paths.n_steps))
        for k in range(paths.n_steps):
            out[:, k] = merton_delta(
                paths.spot[:, k], self.strike, float(tau[k]), self.r, self.sigma,
                self.intensity, self.mu_j, self.sigma_j,
            )
        return out[:, :, None]

    def premium(self) -> float:
        price = merton_call(
            self.s0, self.strike, self.maturity, self.r, self.sigma,
            self.intensity, self.mu_j, self.sigma_j,
        )
        return float(np.asarray(price))


def build_market() -> MertonJump:
    """The market the playground constructs from this file.

    Change these numbers, or return a different `Market`. Keep `s0`,
    `strike`, `maturity` and `r`: the playground reads them.
    """
    return MertonJump(
        s0=100.0,
        strike=100.0,
        maturity=30 / 365,
        r=0.0,
        sigma=0.2,
        intensity=1.0,
        mu_j=-0.1,
        sigma_j=0.2,
    )


if __name__ == "__main__":
    from deephedge.upload import validate_market

    print(validate_market(build_market()))
