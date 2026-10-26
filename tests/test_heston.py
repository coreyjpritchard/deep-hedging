"""Heston market: price, variance swap, Feller, Black-Scholes limit, multi-instrument P&L."""

from dataclasses import replace

import numpy as np
import pytest

from deephedge.env import bs_call_price

M = 20_000
# Fine grid. The 30-step hedging grid is biased against the continuous price.
N_FINE = 480
PAPER_PRICE = 1.6918339240921085


def _paper(impl, **kwargs):
    return impl.Heston(**kwargs)


def test_shapes_draw_and_swap_at_inception(impl):
    mkt = _paper(impl)
    assert mkt.n_instruments == 2
    assert mkt.n_features == 2
    assert mkt.instrument_names == ("spot", "variance_swap")
    assert mkt.feature_names == ("spot", "variance")
    paths = mkt.simulate(8, 5, seed=4)
    assert paths.instruments.shape == (8, 6, 2)
    assert paths.features.shape == (8, 6, 2)
    np.testing.assert_array_equal(paths.spot, paths.features[:, :, 0])
    np.testing.assert_array_equal(paths.spot[:, 0], mkt.s0)
    # v0 = theta and r = 0, so the strike-0 variance swap starts at theta * T.
    np.testing.assert_allclose(paths.instruments[:, 0, 1], mkt.theta * mkt.maturity)
    again = mkt.simulate(8, 5, seed=4)
    np.testing.assert_array_equal(paths.instruments, again.instruments)


def test_brownian_column_zero_is_the_variance_shock(impl):
    """One step, hand-rolled from a fresh draw of shape (m, 1, 2).

    Column 0 shocks variance. Column 1 is orthogonal. The spot shock is the
    correlated mix, applied after the draw.
    """
    mkt = _paper(impl)
    paths = mkt.simulate(5, 1, seed=7)
    z = np.random.default_rng(7).standard_normal((5, 1, 2))
    dt = mkt.maturity
    vp = mkt.v0
    orthogonal = np.sqrt(1.0 - mkt.rho**2)
    z_spot = mkt.rho * z[:, 0, 0] + orthogonal * z[:, 0, 1]
    log_next = np.log(mkt.s0) + (mkt.mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * z_spot
    latent = mkt.v0 + mkt.kappa * (mkt.theta - vp) * dt + mkt.xi * np.sqrt(vp * dt) * z[:, 0, 0]
    np.testing.assert_allclose(paths.spot[:, 1], np.exp(log_next))
    np.testing.assert_allclose(paths.features[:, 1, 1], np.maximum(latent, 0.0))


def test_premium_ignores_the_simulation_drifts(impl):
    base = _paper(impl).premium()
    shifted = _paper(impl, mu=0.4, kappa_sim=4.0, theta_sim=0.2).premium()
    assert shifted == base
    calm = _paper(impl, kappa_sim=4.0, theta_sim=0.2).simulate(300, 10, seed=1)
    wild = _paper(impl).simulate(300, 10, seed=1)
    assert not np.allclose(calm.features[:, -1, 1], wild.features[:, -1, 1])


def test_quadrature_price_is_the_2019_call(impl):
    """The little-trap premium at the section 5.2 parameters.

    The paper prints 1.69. The quadrature value is the longer one below.
    """
    price = _paper(impl).premium()
    assert abs(price - PAPER_PRICE) < 1e-9
    assert abs(price - 1.69) < 0.005


def test_cos_greeks_match_quadrature_at_inception(impl):
    """The pathwise COS hedge agrees with the quadrature at t = 0.

    Default grid, 512 cosines. Price within 1e-3, delta within 2e-3,
    vega within 0.2.
    """
    mkt = _paper(impl)
    quad_price, quad_delta, quad_vega = mkt.quad_greeks()
    price, delta, vega = mkt.greeks(mkt.s0, mkt.v0, mkt.maturity)
    assert abs(float(price) - quad_price) < 1e-3
    assert abs(float(delta) - quad_delta) < 2e-3
    assert abs(float(vega) - quad_vega) < 0.2
    paths = mkt.simulate(3, 4, seed=2)
    bench = mkt.benchmark_positions(paths)
    assert bench.shape == (3, 4, 2)
    slope = (1.0 - np.exp(-mkt.kappa * mkt.maturity)) / mkt.kappa
    np.testing.assert_allclose(bench[:, 0, 0], quad_delta, atol=2e-3)
    np.testing.assert_allclose(bench[:, 0, 1], quad_vega / slope, atol=2.0)


@pytest.mark.parametrize(
    "spot, variance, days",
    [
        (99.945, 0.0, 1),
        (100.046, 0.0, 2),
        (100.046, 0.0, 5),
        (100.95445, 0.0, 15),
        (100.0, 1e-4, 1),
        (100.0, 1e-3, 3),
        (98.0, 0.0, 10),
    ],
)
def test_cos_delta_matches_quadrature_at_zero_variance(impl, spot, variance, days):
    """Near expiry at zero variance the COS delta still matches the quadrature.

    On the 2019 parameters the truncated variance is 0 on 45 per cent of hedging
    dates. With a 0.3 floor on the truncation half-width, the delta one day out
    at 99.945 was 0.38 against a quadrature 0.012. Delta within 2e-3, price
    within 1e-4.
    """
    mkt = _paper(impl)
    tau = days / 365
    quad_price, quad_delta, _ = _paper(impl, s0=spot, v0=variance, maturity=tau).quad_greeks()
    price, delta, _ = mkt.greeks(spot, variance, tau)
    assert abs(float(delta) - quad_delta) < 2e-3
    assert abs(float(price) - quad_price) < 1e-4


def test_cos_deep_in_the_money_delta_at_zero_variance(impl):
    """13% in the money, two days out, zero variance: delta is 1 to 1e-3.

    The quadrature does not converge here, so the check is the bound. Twelve
    standard deviations of truncation gave 0.998: the tails are fat at zero variance.
    """
    _, delta, _ = _paper(impl).greeks(113.184, 0.0, 2 / 365)
    assert 0.999 < float(delta) <= 1.0 + 1e-6


def test_price_matches_monte_carlo(impl):
    """E[e^{-rT} (S_T - K)+] equals the quadrature price. m = 20 000, n = 480, 4 SE.

    mu = r and the variance is simulated at the pricing kappa and theta, so
    the paths are under the pricing measure. n = 480, not the 30-step
    hedging grid: full truncation on 30 steps prices the call about 0.1
    above the continuous value.
    """
    mkt = _paper(impl)
    paths = mkt.simulate(M, N_FINE, seed=1)
    discounted = float(paths.discount[-1]) * mkt.payoff(paths)
    se = discounted.std() / np.sqrt(M)
    assert abs(discounted.mean() - mkt.premium()) < 4 * se


def test_variance_swap_is_a_martingale(impl):
    """Discounted variance-swap price has mean equal to its time-0 price.

    m = 20 000, n = 480, 4 SE, at the middle date and at maturity. r = 0,
    so the discount is 1 and the time-0 price is theta * T.
    """
    mkt = _paper(impl)
    paths = mkt.simulate(M, N_FINE, seed=3)
    swap = paths.instruments[:, :, 1]
    target = float(swap[0, 0])
    np.testing.assert_allclose(target, mkt.theta * mkt.maturity)
    for k in (N_FINE // 2, N_FINE):
        x = swap[:, k] * paths.discount[k]
        se = x.std() / np.sqrt(M)
        assert abs(x.mean() - target) < 4 * se, (k, x.mean(), se, target)


def test_feller_violation_stays_finite_and_truncates(impl):
    """Paper parameters have 2 kappa theta = 0.08 < xi^2 = 4. Variance features stay non-negative.

    On 2 000 paths of 30 steps with seed 0 the truncation binds: some
    feature entries after t = 0 are exactly zero, which is the full
    truncation floor, and the spot stays positive and finite.
    """
    mkt = _paper(impl)
    assert 2 * mkt.kappa * mkt.theta < mkt.xi**2
    paths = mkt.simulate(2_000, 30, seed=0)
    variance = paths.features[:, :, 1]
    assert np.isfinite(paths.instruments).all()
    assert np.isfinite(variance).all()
    assert (variance >= 0).all()
    assert (paths.spot > 0).all()
    assert (variance[:, 1:] == 0).any()


def test_zero_vol_of_vol_is_black_scholes(impl):
    """xi = 0 with v0 = theta is a Black-Scholes call with sigma = sqrt(theta).

    The premium matches the closed form. The variance feature stays at
    theta. The Monte Carlo call, m = 20 000 and n = 30, is within 4 SE of
    that price: with xi = 0 the Euler step is an exact Gaussian.
    """
    mkt = _paper(impl, xi=0.0)
    sigma = np.sqrt(mkt.theta)
    closed = float(bs_call_price(mkt.s0, mkt.strike, mkt.maturity, sigma, mkt.r))
    assert abs(mkt.premium() - closed) < 1e-12
    paths = mkt.simulate(M, 30, seed=5)
    np.testing.assert_allclose(paths.features[:, :, 1], mkt.theta)
    discounted = mkt.payoff(paths)  # r = 0
    se = discounted.std() / np.sqrt(M)
    assert abs(discounted.mean() - closed) < 4 * se

    # A different starting variance: total variance is L(0, v0), not theta * T.
    moved = _paper(impl, xi=0.0, v0=0.09)
    tau = moved.maturity
    integrated = (0.09 - moved.theta) * (1.0 - np.exp(-moved.kappa * tau)) / moved.kappa
    integrated += moved.theta * tau
    closed_moved = float(
        bs_call_price(moved.s0, moved.strike, tau, np.sqrt(integrated / tau), moved.r)
    )
    assert abs(moved.premium() - closed_moved) < 1e-12


def test_small_vol_of_vol_stays_near_black_scholes(impl):
    """xi = 0.05, v0 = theta. The call is within 0.01 of Black-Scholes at sigma = sqrt(theta)."""
    mkt = _paper(impl, xi=0.05)
    closed = float(bs_call_price(mkt.s0, mkt.strike, mkt.maturity, np.sqrt(mkt.theta), mkt.r))
    assert abs(mkt.premium() - closed) < 0.01


def test_torch_multi_instrument_pnl_matches_numpy(impl):
    """Two instruments, two rates, payoff charged once. Float64, atol 1e-10.

    Prices are already the discounted ones `terminal_pnl` consumes, so the
    torch contract (discount 1) is the numpy contract.
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    m, n = 16, 7
    prices = rng.random((m, n + 1, 2)) + 2.0
    positions = rng.normal(size=(m, n, 2))
    payoff = rng.random(m)
    rates = (0.01, 0.04)
    paths = impl.Paths(
        time=np.linspace(0.0, 1.0, n + 1),
        instruments=prices,
        features=prices[:, :, :1],
        discount=np.ones(n + 1),
    )
    expected = impl.terminal_pnl(paths, positions, payoff, cost=rates, premium=1.25)
    got = impl.pnl_from_positions_torch(
        torch.as_tensor(prices, dtype=torch.float64),
        torch.as_tensor(positions, dtype=torch.float64),
        torch.as_tensor(payoff, dtype=torch.float64),
        rates,
        1.25,
    )
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-10)

    # A callable payoff is applied to the spot column only, once.
    via_call = impl.pnl_from_positions_torch(
        torch.as_tensor(prices, dtype=torch.float64),
        torch.as_tensor(positions, dtype=torch.float64),
        lambda s: s * 0.0,
        0.02,
        0.0,
    )
    zeros = impl.terminal_pnl(paths, positions, np.zeros(m), cost=0.02, premium=0.0)
    np.testing.assert_allclose(via_call.numpy(), zeros, atol=1e-10)

    leaf = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    loss = impl.pnl_from_positions_torch(
        torch.as_tensor(prices, dtype=torch.float64), leaf, torch.as_tensor(payoff), rates, 1.25
    )
    loss.sum().backward()
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()
    assert leaf.grad.abs().sum() > 0


def test_heston_environment_pnl_matches_torch(impl):
    """The environment P&L on a Heston book equals the torch P&L on discounted prices."""
    torch = pytest.importorskip("torch")
    mkt = _paper(impl, r=0.02, mu=0.02)
    paths = mkt.simulate(32, 6, seed=9)
    positions = mkt.benchmark_positions(paths)
    expected = impl.Environment(mkt, (0.001, 0.004)).pnl(paths, positions)
    discounted = paths.discounted_instruments()
    payoff = mkt.payoff(paths) * paths.discount_terminal()
    got = impl.pnl_from_positions_torch(
        torch.as_tensor(discounted, dtype=torch.float64),
        torch.as_tensor(positions, dtype=torch.float64),
        torch.as_tensor(payoff, dtype=torch.float64),
        (0.001, 0.004),
        mkt.premium(),
    )
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-8)


def test_heston_training_step_is_finite(impl):
    """One epoch on a tiny book. The loss is finite and the graph runs backward."""
    pytest.importorskip("torch")
    setup = replace(
        impl.HestonSetup(),
        m_train=256,
        n_steps=4,
        epochs=1,
        batch_size=128,
        width=8,
        depth=1,
        seed=1,
    )
    result = impl.train_heston(setup)
    assert len(result.history) == 1
    assert np.isfinite(result.history[0])
