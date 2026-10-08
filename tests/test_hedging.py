"""Delta hedging P&L: no-cost limit, discretisation error, and cost scaling.

Setting: S0 = K = 100, sigma = 0.2, T = 30 days, r = mu = 0, m = 20 000 paths. Means are
tested to 4 standard errors, so a correct implementation fails about once in 16 000 runs.
"""

from __future__ import annotations

import numpy as np
import pytest

S0, K, T, SIGMA = 100.0, 100.0, 30 / 365, 0.2
M = 20_000


def paths(impl, n, seed=0, m=M):
    return impl.bs_paths(m, n, T, SIGMA, s0=S0, seed=seed)


def test_paths_shape_and_start(impl):
    log_s = impl.bs_log_paths(50, 7, T, SIGMA, s0=S0, seed=1)
    assert log_s.shape == (50, 8)
    assert np.allclose(log_s[:, 0], np.log(S0))


def test_paths_are_seeded(impl):
    assert np.array_equal(paths(impl, 5, seed=3, m=10), paths(impl, 5, seed=3, m=10))
    assert not np.array_equal(paths(impl, 5, seed=3, m=10), paths(impl, 5, seed=4, m=10))


def test_terminal_log_price_moments(impl):
    """log S_T ~ N(log S0 - sigma^2 T / 2, sigma^2 T). Mean to 4 SE at m = 20 000."""
    x = impl.bs_log_paths(M, 10, T, SIGMA, s0=S0, seed=2)[:, -1]
    sd = SIGMA * np.sqrt(T)
    se = sd / np.sqrt(M)
    assert abs(x.mean() - (np.log(S0) - 0.5 * SIGMA**2 * T)) < 4 * se
    assert abs(x.std() / sd - 1) < 4 / np.sqrt(2 * M)


def test_zero_cost_delta_hedge_is_unbiased(impl):
    """Without costs, mean P&L is zero: premium = E[payoff] and gains are a martingale."""
    pnl = impl.delta_hedge_pnl(paths(impl, 30), K, T, SIGMA, 0.0, 0.0)
    se = pnl.std() / np.sqrt(pnl.size)
    assert abs(pnl.mean()) < 4 * se


def test_zero_cost_hedging_error_falls_like_inverse_sqrt_n(impl):
    """std(P&L) ~ C / sqrt(n), so std(n=20) / std(n=80) is close to 2.

    The SE of each std is about std / sqrt(2m), under 1 per cent at m = 20 000, so the
    band [1.8, 2.2] is many standard errors wide and leaves room for the finite-n bias.
    """
    s20 = impl.delta_hedge_pnl(paths(impl, 20, seed=5), K, T, SIGMA, 0.0, 0.0).std()
    s80 = impl.delta_hedge_pnl(paths(impl, 80, seed=6), K, T, SIGMA, 0.0, 0.0).std()
    assert 1.8 < s20 / s80 < 2.2


def test_costs_make_mean_pnl_negative(impl):
    pnl = impl.delta_hedge_pnl(paths(impl, 30), K, T, SIGMA, 0.0, 0.005)
    se = pnl.std() / np.sqrt(pnl.size)
    assert pnl.mean() < -10 * se


def test_rebalancing_cost_grows_like_sqrt_n(impl):
    """E|change in delta| per step ~ sqrt(dt), so the total over n steps ~ sqrt(n).

    The opening and closing trades do not depend on n, so they are removed and the
    remaining rebalancing cost is compared at n = 20 and n = 80: ratio close to 2.
    """
    cost = 0.01

    def rebalancing_cost(n, seed):
        p = paths(impl, n, seed=seed)
        delta = impl.delta_positions(p, K, T, SIGMA)
        no_cost = impl.delta_hedge_pnl(p, K, T, SIGMA, 0.0, 0.0)
        with_cost = impl.delta_hedge_pnl(p, K, T, SIGMA, 0.0, cost)
        total = no_cost - with_cost
        ends = cost * (delta[:, 0] * p[:, 0] + np.abs(delta[:, -1]) * p[:, -1])
        return (total - ends).mean()

    ratio = rebalancing_cost(80, 8) / rebalancing_cost(20, 7)
    assert 1.75 < ratio < 2.25


def test_pnl_from_positions_matches_delta_hedge(impl):
    p = paths(impl, 15, m=2_000)
    for r in (0.0, 0.03):
        for cost in (0.0, 0.01):
            direct = impl.delta_hedge_pnl(p, K, T, SIGMA, r, cost)
            t = np.linspace(0, T, 16)
            disc = np.exp(-r * t)
            via = impl.pnl_from_positions(
                p * disc,
                impl.delta_positions(p, K, T, SIGMA, r),
                disc[-1] * np.maximum(p[:, -1] - K, 0.0),
                cost,
                premium=float(impl.bs_call_price(S0, K, T, SIGMA, r)),
            )
            np.testing.assert_allclose(direct, via, rtol=0, atol=1e-12)


def test_no_hedge_pnl_is_premium_minus_payoff(impl):
    p = paths(impl, 10, m=1_000)
    pnl = impl.pnl_from_positions(p, np.zeros((1_000, 10)), impl.call_payoff(K), 0.01, 2.0)
    np.testing.assert_allclose(pnl, 2.0 - np.maximum(p[:, -1] - K, 0.0))


def test_torch_pnl_matches_numpy(impl):
    torch = pytest.importorskip("torch")
    p = paths(impl, 12, m=500)
    pos = impl.delta_positions(p, K, T, SIGMA)
    expected = impl.pnl_from_positions(p, pos, impl.call_payoff(K), 0.004, 1.5)
    got = impl.pnl_from_positions_torch(
        torch.as_tensor(p), torch.as_tensor(pos), impl.call_payoff(K), 0.004, 1.5
    )
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-10)
