"""Market simulators. Paths have shape (m, n + 1) and include the t = 0 column."""

from __future__ import annotations

import numpy as np
import torch


def time_grid(n: int, T: float) -> np.ndarray:
    """Uniform grid t_k = k T / n for k = 0, ..., n."""
    return np.linspace(0.0, T, n + 1)


def bs_log_paths(
    m: int,
    n: int,
    T: float,
    sigma: float,
    s0: float = 100.0,
    mu: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Log-price paths of Black-Scholes, dS = mu S dt + sigma S dW.

    Returns log S on the uniform grid of n steps over [0, T], shape (m, n + 1).
    The log price is simulated exactly, so there is no discretisation error.
    """
    rng = np.random.default_rng(seed)
    dt = T / n
    z = rng.standard_normal((m, n))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_s = np.empty((m, n + 1))
    log_s[:, 0] = np.log(s0)
    np.cumsum(increments, axis=1, out=log_s[:, 1:])
    log_s[:, 1:] += np.log(s0)
    return log_s


def bs_paths(
    m: int,
    n: int,
    T: float,
    sigma: float,
    s0: float = 100.0,
    mu: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Price paths of Black-Scholes, exp of `bs_log_paths`. Shape (m, n + 1)."""
    return np.exp(bs_log_paths(m, n, T, sigma, s0=s0, mu=mu, seed=seed))


def to_torch(paths: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The same paths as a torch tensor, so numpy and torch hedges see identical data."""
    return torch.as_tensor(np.asarray(paths), dtype=dtype)
