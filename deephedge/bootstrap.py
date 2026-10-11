"""Block bootstrap of a historical price series, as a `Market`.

Closes become log returns. Each simulated path is a concatenation of
contiguous blocks of those returns, then rescaled so the path starts at
the contract spot. The price and the delta are Black-Scholes with the
annualised volatility of the returns. The paths are not Black-Scholes
paths. This is a data-driven generator, the simplest stand-in for the
market generators of Buehler, Horvath, Lyons, Perez Arribas and Wood
(2020).

Draw. One call, and no others:

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_returns - block + 1, size=(m, n_blocks))

`n_blocks` is the number of blocks needed to cover `n` steps. Block `j`
of a path is the historical returns starting at `starts[i, j]`.

Time. Each step consumes one historical return. The time grid is still
the contract maturity, with `n` equal steps. When the closes are daily,
`n = 30` and maturity is `30/365`, thirty daily returns sit on a
30/365-year grid. The volatility used in the price and the delta is
`std(log returns) * sqrt(trading_days)` with `trading_days` 252, so the
path variance and the Black-Scholes variance differ by the 365-versus-252
day count. `demean` subtracts the mean log return before stitching.
Leave it false to keep the historical drift. With the drift left in, the
discounted spot is not a martingale.
"""

from __future__ import annotations

import math

import numpy as np

from deephedge.env import BlackScholes, Market, Paths

__all__ = ["BlockBootstrap", "parse_closes"]


def parse_closes(text: str) -> tuple[float, ...]:
    """Positive closes from CSV text.

    A line may be `close` or `date,close`. A header that contains no
    number is skipped, and so is a line starting with `#`.
    """
    closes: list[float] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        value: float | None = None
        for part in reversed(line.split(",")):
            try:
                value = float(part.strip())
            except ValueError:
                continue
            break
        if value is None:
            continue
        if value <= 0.0:
            raise ValueError(f"Line {line_number}: close must be positive, got {value}.")
        closes.append(value)
    if len(closes) < 3:
        raise ValueError(f"Need at least three closes, found {len(closes)}.")
    return tuple(closes)


def _returns(closes: tuple[float, ...], demean: bool) -> np.ndarray:
    rets = np.diff(np.log(np.asarray(closes, dtype=float)))
    if demean:
        rets = rets - rets.mean()
    return rets


class BlockBootstrap(Market):
    """European call on paths stitched from blocks of historical log returns."""

    def __init__(
        self,
        closes: tuple[float, ...],
        s0: float = 100.0,
        strike: float = 100.0,
        maturity: float = 30 / 365,
        r: float = 0.0,
        block: int = 5,
        trading_days: float = 252.0,
        demean: bool = False,
    ):
        self.closes = tuple(float(c) for c in closes)
        self.s0 = float(s0)
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.r = float(r)
        self.block = int(block)
        self.trading_days = float(trading_days)
        self.demean = bool(demean)
        if len(self.closes) < 3:
            raise ValueError("Need at least three closes.")
        if any(close <= 0.0 for close in self.closes):
            raise ValueError("Closes must be positive.")
        if self.block < 1:
            raise ValueError("block must be at least 1.")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if self.trading_days <= 0.0:
            raise ValueError("trading_days must be positive.")
        if self.s0 <= 0.0 or self.strike <= 0.0:
            raise ValueError("spot and strike must be positive.")

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

    def log_returns(self) -> np.ndarray:
        """Log returns actually stitched. Demeaned when `demean` is set."""
        return _returns(self.closes, self.demean)

    @property
    def sigma(self) -> float:
        """Annualised volatility, `std(log returns) * sqrt(trading_days)`."""
        rets = self.log_returns()
        if rets.size < 2:
            raise ValueError("Need at least three closes to estimate a volatility.")
        return float(rets.std(ddof=1) * math.sqrt(self.trading_days))

    def _carrier(self) -> BlackScholes:
        """Black-Scholes market used for the price, the delta and the discount."""
        return BlackScholes(
            s0=self.s0,
            strike=self.strike,
            maturity=self.maturity,
            sigma=self.sigma,
            r=self.r,
            mu=self.r,
        )

    def simulate(self, m: int, n: int, seed: int) -> Paths:
        """Draw `m` paths of `n` steps. One `rng.integers` call, documented above."""
        if m < 1 or n < 1:
            raise ValueError("m and n must be at least 1.")
        rets = self.log_returns()
        if self.block > rets.size:
            raise ValueError(
                f"block length {self.block} is longer than the {rets.size} log returns."
            )
        n_blocks = math.ceil(n / self.block)
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, rets.size - self.block + 1, size=(m, n_blocks))
        offsets = np.arange(self.block)
        windows = rets[starts[:, :, None] + offsets[None, None, :]]
        increments = windows.reshape(m, -1)[:, :n]
        log_s = np.empty((m, n + 1))
        log_s[:, 0] = np.log(self.s0)
        np.cumsum(increments, axis=1, out=log_s[:, 1:])
        log_s[:, 1:] += np.log(self.s0)
        spot = np.exp(log_s)
        spot[:, 0] = self.s0
        return self._carrier().paths_from_spot(spot)

    def payoff(self, paths: Paths) -> np.ndarray:
        return self._carrier().payoff(paths)

    def price_along(self, paths: Paths) -> np.ndarray:
        return self._carrier().price_along(paths)

    def benchmark_positions(self, paths: Paths) -> np.ndarray:
        return self._carrier().benchmark_positions(paths)

    def premium(self) -> float:
        return self._carrier().premium()
