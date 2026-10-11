"""Cost models: zero cost, monotonicity, which instrument is charged, torch agreement.

Books are synthetic and small. No Monte Carlo tolerance is required: the
charges are closed form. Torch matches numpy in float64 to 1e-10.
"""

from __future__ import annotations

import numpy as np
import pytest


def _prices_and_book():
    """Two instruments, four paths, three holding dates. Both legs trade."""
    prices = np.array(
        [
            [[100.0, 0.003], [101.0, 0.004], [99.0, 0.002], [103.0, 0.005]],
            [[100.0, 0.003], [98.0, 0.003], [102.0, 0.006], [97.0, 0.001]],
            [[100.0, 0.010], [105.0, 0.012], [110.0, 0.008], [108.0, 0.009]],
            [[100.0, 0.003], [100.0, 0.003], [100.0, 0.003], [100.0, 0.003]],
        ],
        dtype=float,
    )
    positions = np.array(
        [
            [[0.5, 20.0], [0.2, -10.0], [0.0, 5.0]],
            [[-0.3, 0.0], [-0.3, 15.0], [0.1, 15.0]],
            [[0.8, 40.0], [0.8, 40.0], [0.4, 0.0]],
            [[0.0, 0.0], [0.25, -8.0], [0.25, -8.0]],
        ],
        dtype=float,
    )
    return prices, positions


def _paths(impl, prices, discount=None):
    m, n_plus, d = prices.shape
    if discount is None:
        discount = np.ones(n_plus)
    features = prices[:, :, :1]
    return impl.Paths(
        time=np.linspace(0.0, 1.0, n_plus),
        instruments=prices,
        features=np.repeat(features, 1, axis=2) if features.shape[-1] == 1 else features,
        discount=discount,
    )


def _pnl(impl, prices, positions, cost, premium=1.5):
    payoff = np.zeros(prices.shape[0])
    return impl.terminal_pnl(_paths(impl, prices), positions, payoff, cost, premium)


def test_zero_cost_models_match_frictionless_pnl(impl):
    prices, positions = _prices_and_book()
    base = _pnl(impl, prices, positions, 0.0)
    models = [
        impl.ProportionalCost(0.0),
        impl.ProportionalCost((0.0, 0.0)),
        impl.ProportionalCost(rate=0.0, unit=1.0),
        impl.ProportionalCost(rate=(0.0, 0.0), unit=(None, 1.0)),
        impl.FixedPlusProportional(rate=0.0, fixed=0.0),
        impl.FixedPlusProportional(rate=(0.0, 0.0), fixed=(0.0, 0.0)),
        impl.QuadraticCost(0.0),
        impl.QuadraticCost(impact=(0.0, 0.0), unit=(None, 1.0)),
    ]
    for model in models:
        got = _pnl(impl, prices, positions, model)
        np.testing.assert_allclose(got, base, atol=1e-12, err_msg=type(model).__name__)


def test_proportional_model_matches_a_plain_rate(impl):
    prices, positions = _prices_and_book()
    via_rate = _pnl(impl, prices, positions, (0.01, 0.02))
    via_model = _pnl(impl, prices, positions, impl.ProportionalCost((0.01, 0.02)))
    np.testing.assert_allclose(via_model, via_rate, atol=1e-12)


def test_liquidation_is_charged(impl):
    """A constant holding pays the open and the close, and nothing between."""
    prices = np.array([[[100.0], [110.0], [90.0], [105.0]]], dtype=float)
    positions = np.full((1, 3, 1), 0.4)
    proportional = impl.ProportionalCost(0.01).amount(prices, positions)
    np.testing.assert_allclose(proportional, 0.01 * (0.4 * 100.0 + 0.4 * 105.0))
    fixed = impl.FixedPlusProportional(rate=0.0, fixed=0.05).amount(prices, positions)
    np.testing.assert_allclose(fixed, 2 * 0.05)
    quadratic = impl.QuadraticCost(0.02).amount(prices, positions)
    np.testing.assert_allclose(quadratic, 0.02 * (0.4**2 * 100.0 + 0.4**2 * 105.0))


def test_costs_are_monotone_and_pnl_falls(impl):
    prices, positions = _prices_and_book()
    pairs = [
        (impl.ProportionalCost(0.001), impl.ProportionalCost(0.01)),
        (impl.ProportionalCost((0.001, 0.0)), impl.ProportionalCost((0.004, 0.002))),
        (
            impl.FixedPlusProportional(rate=0.001, fixed=0.0),
            impl.FixedPlusProportional(rate=0.001, fixed=0.02),
        ),
        (
            impl.FixedPlusProportional(fixed=(0.01, 0.0)),
            impl.FixedPlusProportional(fixed=(0.01, 0.05)),
        ),
        (impl.QuadraticCost(0.001), impl.QuadraticCost(0.02)),
        (impl.QuadraticCost(impact=(0.0, 0.01)), impl.QuadraticCost(impact=(0.002, 0.01))),
    ]
    for low, high in pairs:
        cost_low = low.amount(prices, positions)
        cost_high = high.amount(prices, positions)
        assert np.all(cost_high + 1e-12 >= cost_low)
        assert np.any(cost_high > cost_low + 1e-8)
        pnl_low = _pnl(impl, prices, positions, low)
        pnl_high = _pnl(impl, prices, positions, high)
        assert np.all(pnl_high <= pnl_low + 1e-12)


def test_per_instrument_rate_charges_only_that_leg(impl):
    prices, _positions = _prices_and_book()
    spot_only = np.zeros_like(_positions)
    spot_only[:, :, 0] = _positions[:, :, 0]
    swap_only = np.zeros_like(_positions)
    swap_only[:, :, 1] = _positions[:, :, 1]
    spot_rate = impl.ProportionalCost(rate=(0.2, 0.0))
    swap_rate = impl.ProportionalCost(rate=(0.0, 0.2))
    np.testing.assert_allclose(spot_rate.amount(prices, swap_only), 0.0, atol=1e-12)
    np.testing.assert_allclose(swap_rate.amount(prices, spot_only), 0.0, atol=1e-12)
    assert np.all(spot_rate.amount(prices, spot_only) > 0)
    assert np.all(swap_rate.amount(prices, swap_only) > 0)
    both = impl.ProportionalCost(rate=(0.2, 0.2)).amount(prices, _positions)
    parts = spot_rate.amount(prices, _positions) + swap_rate.amount(prices, _positions)
    np.testing.assert_allclose(both, parts, atol=1e-12)


def test_variance_swap_unit_ignores_the_mark(impl):
    """A cash unit charges per contract. The swap's price column does not enter."""
    prices = np.array(
        [
            [[100.0, 0.003], [102.0, 0.004]],
            [[100.0, 0.003], [97.0, 0.001]],
        ],
        dtype=float,
    )
    positions = np.array([[[0.0, 10.0]], [[0.5, -4.0]]], dtype=float)
    unit = impl.ProportionalCost(rate=(0.0, 0.2), unit=(None, 1.0))
    marked_up = prices.copy()
    marked_up[:, :, 1] *= 50.0
    np.testing.assert_allclose(unit.amount(prices, positions), unit.amount(marked_up, positions))
    # Open and close of a constant 10, and of a constant -4. Spot rate is 0.
    expected = np.array([0.2 * (10.0 + 10.0), 0.2 * (4.0 + 4.0)])
    np.testing.assert_allclose(unit.amount(prices, positions), expected)
    on_the_mark = impl.ProportionalCost(rate=(0.0, 0.2))
    assert np.all(
        np.abs(on_the_mark.amount(prices, positions) - on_the_mark.amount(marked_up, positions))
        > 1e-6
    )


def test_scaled_multiplies_the_charge(impl):
    prices, positions = _prices_and_book()
    base = impl.ProportionalCost((0.01, 0.02), unit=(None, 1.0))
    np.testing.assert_allclose(
        impl.scaled(base, 2.0).amount(prices, positions),
        2.0 * base.amount(prices, positions),
    )
    np.testing.assert_allclose(impl.scaled(base, 0.0).amount(prices, positions), 0.0, atol=1e-12)


def test_torch_pnl_matches_numpy_for_every_cost_model(impl):
    torch = pytest.importorskip("torch")
    prices, positions = _prices_and_book()
    payoff = np.maximum(prices[:, -1, 0] - 100.0, 0.0)
    models = [
        impl.ProportionalCost(0.01),
        impl.ProportionalCost(rate=(0.01, 0.02), unit=(None, 1.0)),
        impl.FixedPlusProportional(rate=(0.005, 0.0), fixed=(0.02, 0.1), unit=(None, 1.0)),
        impl.QuadraticCost(impact=(0.01, 0.002), unit=(None, 1.0)),
    ]
    price_t = torch.tensor(prices, dtype=torch.float64)
    pos_t = torch.tensor(positions, dtype=torch.float64)
    pay_t = torch.tensor(payoff, dtype=torch.float64)
    for model in models:
        expected = impl.terminal_pnl(_paths(impl, prices), positions, payoff, model, 1.25)
        got = impl.pnl_from_positions_torch(price_t, pos_t, pay_t, model, 1.25, exact=True)
        np.testing.assert_allclose(got.detach().numpy(), expected, atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            model.amount_torch(price_t, pos_t, exact=True).detach().numpy(),
            model.amount(prices, positions),
            atol=1e-10,
            rtol=0,
        )


def test_one_instrument_cost_model_matches_the_float_torch_branch(impl):
    """The 2d training branch stays the proportional formula. A cost model agrees with it."""
    torch = pytest.importorskip("torch")
    prices, positions = _prices_and_book()
    spot = prices[:, :, 0]
    held = positions[:, :, 0]
    payoff = np.maximum(spot[:, -1] - 100.0, 0.0)
    expected = impl.pnl_from_positions(spot, held, payoff, 0.004, 1.5)
    spot_t = torch.tensor(spot, dtype=torch.float64)
    held_t = torch.tensor(held, dtype=torch.float64)
    pay_t = torch.tensor(payoff, dtype=torch.float64)
    via_float = impl.pnl_from_positions_torch(spot_t, held_t, pay_t, 0.004, 1.5)
    via_model = impl.pnl_from_positions_torch(
        spot_t, held_t, pay_t, impl.ProportionalCost(0.004), 1.5, exact=True
    )
    np.testing.assert_allclose(via_float.detach().numpy(), expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(via_model.detach().numpy(), expected, atol=1e-10, rtol=0)


def test_smooth_fixed_cost_is_exact_for_a_real_trade_and_has_a_gradient(impl):
    torch = pytest.importorskip("torch")
    prices = torch.tensor([[[100.0], [101.0], [99.0]]], dtype=torch.float64)
    held = torch.tensor([[[1.0], [1.0]]], dtype=torch.float64)
    model = impl.FixedPlusProportional(rate=0.0, fixed=0.4, width=1e-2)
    exact = model.amount_torch(prices, held, exact=True)
    smooth = model.amount_torch(prices, held, exact=False)
    np.testing.assert_allclose(exact.detach().numpy(), 2 * 0.4)
    np.testing.assert_allclose(smooth.detach().numpy(), exact.detach().numpy(), atol=1e-6)

    dust = torch.tensor([[[1e-4], [1e-4]]], dtype=torch.float64)
    exact_dust = model.amount_torch(prices, dust, exact=True)
    smooth_dust = model.amount_torch(prices, dust, exact=False)
    assert float(exact_dust) == pytest.approx(2 * 0.4)
    assert float(smooth_dust) < 0.5 * float(exact_dust)

    small = torch.tensor([[[0.03], [0.03]]], dtype=torch.float64, requires_grad=True)
    impl_loss = model.amount_torch(prices, small, exact=False).sum()
    impl_loss.backward()
    assert small.grad is not None
    assert torch.isfinite(small.grad).all()
    assert float(small.grad.abs().sum()) > 0

    frozen = small.detach().clone().requires_grad_(True)
    model.amount_torch(prices, frozen, exact=True).sum().backward()
    assert float(frozen.grad.abs().sum()) == 0.0

    zero = torch.zeros(1, 2, 1, dtype=torch.float64)
    np.testing.assert_allclose(
        model.amount_torch(prices, zero, exact=False).detach().numpy(), 0.0, atol=1e-12
    )
