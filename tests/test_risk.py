"""Risk measures on X ~ N(mu, s^2), where closed forms exist.

entropic: (1/lam) log E exp(-lam X) = -mu + lam s^2 / 2.
CVaR_alpha of the loss -X: -mu + s phi(Phi^{-1}(alpha)) / (1 - alpha).
Each is tested to 4 standard errors at m = 200 000.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

MU, S, M = 0.3, 1.5, 200_000


@pytest.fixture(scope="module")
def sample():
    return np.random.default_rng(11).normal(MU, S, M)


@pytest.mark.parametrize("lam", [0.1, 0.5, 1.0])
def test_entropic_normal(impl, sample, lam):
    exact = -MU + lam * S**2 / 2
    w = np.exp(-lam * (sample - MU))
    se = w.std() / w.mean() / np.sqrt(M) / lam  # delta method on log of a sample mean
    assert abs(impl.entropic(sample, lam) - exact) < 4 * se


@pytest.mark.parametrize("alpha", [0.5, 0.9, 0.95, 0.99])
def test_cvar_normal(impl, sample, alpha):
    exact = -MU + S * norm.pdf(norm.ppf(alpha)) / (1 - alpha)
    var = -MU + S * norm.ppf(alpha)
    tail = np.maximum(-sample - var, 0.0)
    se = tail.std() / (1 - alpha) / np.sqrt(M)  # Rockafellar-Uryasev representation
    assert abs(impl.cvar(sample, alpha) - exact) < 4 * se


def test_monetary_and_monotone(impl, sample):
    x = sample[:10_000]
    for rho in (lambda y: impl.entropic(y, 0.5), lambda y: impl.cvar(y, 0.9)):
        assert rho(x + 2.0) == pytest.approx(rho(x) - 2.0)
        assert rho(x + np.abs(x)) <= rho(x)


def test_torch_matches_numpy(impl, sample):
    torch = pytest.importorskip("torch")
    x = sample[:10_000]
    t = torch.as_tensor(x)
    assert float(impl.entropic_torch(t, 0.7)) == pytest.approx(impl.entropic(x, 0.7))
    assert float(impl.cvar_torch(t, 0.95)) == pytest.approx(impl.cvar(x, 0.95))
