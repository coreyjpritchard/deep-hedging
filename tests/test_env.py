"""Black-Scholes market behind the environment interface.

Setting, unless a test says otherwise: S0 = K = 100, sigma = 0.2, T = 30 days,
m = 20 000 paths. Monte Carlo checks use 4 standard errors, so a correct
implementation fails a single check about once in 16 000 runs. Martingale and
price checks are under the pricing measure, mu = r.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from deephedge.hedging import call_payoff, delta_hedge_pnl
from deephedge.markets import bs_paths
from deephedge.policy import DeltaPolicy, rollout_numpy, rollout_paths
from deephedge.risk import entropic
from deephedge.train import DEFAULT, cached_policy

S0, K, T, SIGMA = 100.0, 100.0, 30 / 365, 0.2
M = 20_000


def market(impl, r=0.0, mu=0.0):
    return impl.BlackScholes(s0=S0, strike=K, maturity=T, sigma=SIGMA, r=r, mu=mu)


def test_shapes_names_and_seed(impl):
    mkt = market(impl, r=0.01, mu=0.01)
    paths = mkt.simulate(32, 10, seed=1)
    assert paths.instruments.shape == (32, 11, 1)
    assert paths.features.shape == (32, 11, 1)
    assert paths.time.shape == (11,)
    assert paths.discount.shape == (11,)
    assert paths.n_steps == 10
    assert mkt.n_instruments == 1 and mkt.n_features == 1
    assert mkt.instrument_names == ("spot",)
    assert mkt.feature_names == ("spot",)
    assert paths.time[0] == 0.0
    assert paths.time[-1] == pytest.approx(T)
    np.testing.assert_allclose(paths.discount, np.exp(-0.01 * paths.time))
    np.testing.assert_array_equal(paths.features, paths.instruments)
    # S_0 = exp(log S0), which is one ulp off 100. The simulator stores the log.
    np.testing.assert_allclose(paths.spot[:, 0], S0, rtol=0, atol=1e-12)
    again = mkt.simulate(32, 10, seed=1)
    np.testing.assert_array_equal(paths.spot, again.spot)


def test_simulate_matches_bs_paths(impl):
    mkt = market(impl, mu=0.05)
    paths = mkt.simulate(20, 7, seed=3)
    expected = impl.bs_paths(20, 7, T, SIGMA, s0=S0, mu=0.05, seed=3)
    np.testing.assert_array_equal(paths.spot, expected)


def test_premium_and_expiry_price(impl):
    mkt = market(impl, r=0.02)
    np.testing.assert_allclose(mkt.premium(), float(impl.bs_call_price(S0, K, T, SIGMA, 0.02)))
    spot = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(mkt.price(spot, 0.0), [0.0, 0.0, 20.0])
    np.testing.assert_allclose(mkt.delta(spot, 0.0), [0.0, 0.0, 1.0])


def test_price_along_matches_payoff_at_maturity(impl):
    mkt = market(impl, r=0.03, mu=0.03)
    paths = mkt.simulate(100, 12, seed=4)
    price = mkt.price_along(paths)
    np.testing.assert_allclose(price[:, 0], mkt.premium())
    np.testing.assert_allclose(price[:, -1], mkt.payoff(paths))
    assert mkt.benchmark_positions(paths).shape == (100, 12, 1)


def test_discounted_price_is_a_martingale(impl):
    """E[e^{-rt} C(t, S_t)] = C(0) at three dates. m = 20 000, 4 SE.

    Drift mu equals the rate r = 0.05, so the paths are under the pricing
    measure. The price is the analytic Black-Scholes call.
    """
    r = 0.05
    mkt = market(impl, r=r, mu=r)
    paths = mkt.simulate(M, 30, seed=11)
    discounted = mkt.price_along(paths) * paths.discount
    c0 = mkt.premium()
    np.testing.assert_allclose(discounted[:, 0], c0)
    for k in (1, 15, 30):
        x = discounted[:, k]
        se = x.std() / np.sqrt(x.size)
        assert abs(x.mean() - c0) < 4 * se


def test_analytic_price_matches_monte_carlo(impl):
    """E[e^{-rT} (S_T - K)+] equals the analytic price. m = 20 000, 4 SE.

    mu = r = 0.04. The sample is the discounted payoff, not the price process.
    """
    r = 0.04
    mkt = market(impl, r=r, mu=r)
    paths = mkt.simulate(M, 20, seed=17)
    discounted_payoff = paths.discount[-1] * mkt.payoff(paths)
    se = discounted_payoff.std() / np.sqrt(M)
    assert abs(discounted_payoff.mean() - mkt.premium()) < 4 * se


def test_zero_cost_delta_variance_shrinks_with_rebalancing(impl):
    """With zero cost, std of the benchmark-hedged P&L scales as 1/sqrt(n).

    std(n=20) / std(n=80) is near 2. m = 20 000. The band [1.8, 2.2] is the
    same one as the legacy delta-hedge test: the SE of each std is under 1
    per cent, and the band leaves room for the finite-n bias.
    """
    mkt = market(impl)
    env = impl.Environment(mkt, impl.ProportionalCost(0.0))

    def hedged_std(n, seed):
        paths = mkt.simulate(M, n, seed)
        return env.pnl(paths, mkt.benchmark_positions(paths)).std()

    s20, s80 = hedged_std(20, 5), hedged_std(80, 6)
    assert s80 < s20
    assert 1.8 < s20 / s80 < 2.2


def test_environment_pnl_matches_delta_hedge(impl):
    """The environment reproduces delta_hedge_pnl, including a nonzero rate."""
    spot = bs_paths(500, 15, T, SIGMA, s0=S0, seed=8)
    # Premium follows the market's s0. Match it to the simulated column, which
    # is exp(log S0) rather than S0.
    s0 = float(spot[0, 0])
    for r in (0.0, 0.03):
        for cost in (0.0, 0.01):
            mkt = impl.BlackScholes(s0=s0, strike=K, maturity=T, sigma=SIGMA, r=r, mu=0.0)
            paths = mkt.paths_from_spot(spot)
            via = impl.Environment(mkt, cost).pnl(paths, mkt.benchmark_positions(paths))
            direct = impl.delta_hedge_pnl(spot, K, T, SIGMA, r, cost)
            np.testing.assert_array_equal(via, direct)


def test_pnl_adds_across_instruments(impl):
    """Gains and proportional costs sum over the instrument axis.

    Two instruments, two rates. The payoff is charged once.
    """
    rng = np.random.default_rng(0)
    m, n = 8, 5
    prices = rng.random((m, n + 1, 2)) + 5.0
    positions = rng.normal(size=(m, n, 2))
    time = np.linspace(0.0, 1.0, n + 1)
    paths = impl.Paths(
        time=time,
        instruments=prices,
        features=prices[:, :, :1],
        discount=np.exp(-0.01 * time),
    )
    payoff = rng.random(m)
    total = impl.terminal_pnl(paths, positions, payoff, cost=(0.01, 0.03), premium=1.5)
    disc = paths.discounted_instruments()
    legs = [
        impl.pnl_from_positions(disc[:, :, j], positions[:, :, j], 0.0, rate, 0.0)
        for j, rate in enumerate((0.01, 0.03))
    ]
    expected = 1.5 - payoff * float(paths.discount[-1]) + legs[0] + legs[1]
    np.testing.assert_allclose(total, expected, rtol=0, atol=1e-12)


def test_rollout_positions_keeps_feature_and_instrument_axes(impl):
    """A policy may read f features and return d instrument weights, f != d."""

    class HoldTwo(nn.Module):
        def forward(self, t, features, prev):
            return features[:, :2]

    m, n, f = 5, 4, 3
    features = torch.arange(m * (n + 1) * f, dtype=torch.float32).reshape(m, n + 1, f)
    time = np.linspace(0.0, 1.0, n + 1)
    pos = impl.rollout_positions(HoldTwo(), features, time, n_instruments=2)
    assert tuple(pos.shape) == (m, n, 2)
    torch.testing.assert_close(pos, features[:, :-1, :2])


def test_rollout_paths_matches_spot_rollout(impl):
    mkt = market(impl)
    paths = mkt.simulate(40, 6, seed=9)
    policy = DeltaPolicy(K, T, SIGMA)
    via = impl.rollout_paths(policy, paths)
    direct = rollout_numpy(policy, paths.spot, T)
    np.testing.assert_array_equal(via, direct)


def test_session_00_logged_risks_unchanged():
    """Logged session 00 row at 50 bp, seed 12345, m = 20 000.

    The log reports network entropic risk 1.05 and delta 1.54, and means
    -0.83 and -1.37. Half a unit in the last printed digit is the tolerance.
    """
    from dataclasses import replace

    setup = replace(DEFAULT, cost=0.005)
    env = setup.environment()
    sim = env.market.simulate(setup.m_eval, setup.n_steps, seed=12345)
    policy, trained_for = cached_policy(setup)
    assert trained_for.cost == 0.005
    net = env.pnl(sim, rollout_paths(policy, sim))
    delta = env.pnl(sim, env.market.benchmark_positions(sim))
    legacy = delta_hedge_pnl(
        sim.spot, setup.strike, setup.maturity, setup.sigma, setup.r, setup.cost
    )
    # Constant gap: the environment prices at S0 = 100, delta_hedge_pnl prices
    # at the simulated S_0 = exp(log 100), one ulp higher.
    np.testing.assert_allclose(delta, legacy, rtol=0, atol=1e-12)
    assert entropic(net, setup.lam) == pytest.approx(1.05, abs=0.005)
    assert entropic(delta, setup.lam) == pytest.approx(1.54, abs=0.005)
    assert net.mean() == pytest.approx(-0.83, abs=0.005)
    assert delta.mean() == pytest.approx(-1.37, abs=0.005)
    # No hedge pays the call and nothing else. call_payoff is the torch/numpy one.
    flat = env.pnl(sim, np.zeros((setup.m_eval, setup.n_steps)))
    np.testing.assert_allclose(flat, setup.premium - call_payoff(setup.strike)(sim.spot[:, -1]))
