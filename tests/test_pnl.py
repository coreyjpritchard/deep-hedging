"""M02b: `Environment.pnl`, `Environment.pnl_torch` and `Environment.ledger`.

The card: the torch P&L equals the numpy P&L to 1e-10 on two instruments at
`r = 0.03`; the ledger's final value equals the P&L per path; the session 00
table is reproduced through `Environment.pnl`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from deephedge.costs import FixedPlusProportional, ProportionalCost, QuadraticCost, as_cost
from deephedge.env import Environment, Ledger, terminal_pnl
from deephedge.generators import BlackScholes, Heston
from deephedge.hedging import pnl_from_positions, pnl_from_positions_torch

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "results.csv"

COSTS = [
    0.0,
    (0.004, 0.01),
    ProportionalCost((0.002, 0.0), unit=(None, 0.5)),
    FixedPlusProportional(rate=(0.001, 0.003), fixed=0.02),
    QuadraticCost(impact=(0.05, 0.1)),
]


def _heston_book(seed: int, m: int = 400, n: int = 12):
    """Two instruments (spot, variance swap) at r = 0.03, with noisy positions."""
    mkt = Heston(r=0.03, mu=0.03)
    paths = mkt.simulate(m, n, seed=seed)
    rng = np.random.default_rng(seed)
    positions = mkt.benchmark_positions(paths) + 0.1 * rng.standard_normal((m, n, 2))
    # A few positions held still, so the fixed charge's indicator sees zeros.
    positions[:, 5:7] = positions[:, 4:5]
    return mkt, paths, positions


@pytest.mark.parametrize("cost", COSTS, ids=lambda c: type(c).__name__)
def test_pnl_torch_matches_pnl_two_instruments_r003(cost):
    mkt, paths, positions = _heston_book(seed=3)
    assert paths.discount[-1] < 1.0
    env = Environment(mkt, cost)
    expected = env.pnl(paths, positions)
    pos_t = torch.as_tensor(positions, dtype=torch.float64)
    via_paths = env.pnl_torch(paths, pos_t, exact=True)
    np.testing.assert_allclose(via_paths.numpy(), expected, rtol=0, atol=1e-10)
    prices = torch.as_tensor(paths.discounted_instruments(), dtype=torch.float64)
    payoff = torch.as_tensor(mkt.payoff(paths) * paths.discount_terminal(), dtype=torch.float64)
    via_tensor = env.pnl_torch(prices, pos_t, payoff, mkt.premium(), exact=True)
    np.testing.assert_allclose(via_tensor.numpy(), expected, rtol=0, atol=1e-10)


def test_pnl_torch_is_differentiable_in_positions():
    mkt, paths, positions = _heston_book(seed=4, m=50)
    env = Environment(mkt, FixedPlusProportional(rate=0.002, fixed=0.01))
    pos_t = torch.as_tensor(positions, dtype=torch.float64).requires_grad_(True)
    env.pnl_torch(paths, pos_t, exact=False).mean().backward()
    assert pos_t.grad is not None and torch.isfinite(pos_t.grad).all()
    assert pos_t.grad.abs().sum() > 0


@pytest.mark.parametrize("cost", COSTS, ids=lambda c: type(c).__name__)
def test_ledger_final_value_equals_pnl(cost):
    mkt, paths, positions = _heston_book(seed=5)
    env = Environment(mkt, cost)
    book = env.ledger(paths, positions)
    assert isinstance(book, Ledger)
    m, n, d = positions.shape
    assert book.value.shape == (m, n + 1) and book.position.shape == (m, n + 1, d)
    np.testing.assert_allclose(book.final_value, env.pnl(paths, positions), rtol=0, atol=1e-10)
    # Before maturity the value is cash plus what the book holds, marked at the price.
    np.testing.assert_allclose(book.value[:, :-1], (book.cash + book.holdings)[:, :-1], atol=1e-10)
    # At maturity the book is flat and the cash, less the payoff, is the P&L.
    np.testing.assert_array_equal(book.position[:, -1], 0.0)
    np.testing.assert_allclose(book.cash[:, -1] - book.payoff, book.final_value, atol=1e-10)
    np.testing.assert_allclose(book.trade.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        book.cost.sum(axis=1),
        as_cost(cost).amount(paths.discounted_instruments(), positions),
        rtol=0,
        atol=1e-12,
    )


def test_ledger_one_instrument_black_scholes():
    mkt = BlackScholes(r=0.03)
    paths = mkt.simulate(300, 20, seed=6)
    env = Environment(mkt, 0.005)
    positions = mkt.benchmark_positions(paths)[:, :, 0]
    book = env.ledger(paths, positions)
    np.testing.assert_allclose(book.final_value, env.pnl(paths, positions), rtol=0, atol=1e-10)
    assert book.gain[:, 0].tolist() == [0.0] * 300


def test_aliases_are_the_environment():
    """`terminal_pnl` and the PR 2 functions give the environment's numbers to the bit."""
    mkt, paths, positions = _heston_book(seed=7)
    cost = FixedPlusProportional(rate=0.001, fixed=0.01)
    env = Environment(mkt, cost)
    pay = mkt.payoff(paths)
    np.testing.assert_array_equal(
        terminal_pnl(paths, positions, pay, cost, mkt.premium()), env.pnl(paths, positions)
    )
    spot = np.exp(np.cumsum(np.full((5, 4), 0.01), axis=1))
    held = np.full((5, 3), 0.5)
    pay1 = np.maximum(spot[:, -1] - 1.0, 0.0)
    pnl1 = pnl_from_positions(spot, held, pay1, 0.004, 0.2)
    spot_t = torch.as_tensor(spot, dtype=torch.float64)
    held_t = torch.as_tensor(held, dtype=torch.float64)
    pay_t = torch.as_tensor(pay1, dtype=torch.float64)
    torch1 = pnl_from_positions_torch(spot_t, held_t, pay_t, 0.004, 0.2)
    np.testing.assert_allclose(torch1.numpy(), pnl1, rtol=0, atol=1e-12)
    direct = Environment(None, 0.004).pnl_torch(spot_t, held_t, pay_t, 0.2)
    np.testing.assert_array_equal(torch1.numpy(), direct.numpy())


def test_environment_without_market_needs_premium():
    _, paths, positions = _heston_book(seed=8, m=10)
    with pytest.raises(ValueError, match="no market"):
        Environment(None, 0.0).pnl(paths, positions)


def test_session_00_table_reproduced_through_environment_pnl():
    """Every session 00 row of `docs/results.csv`, to the printed six decimals.

    `figures.session00_numbers` scores the cached networks and delta with
    `Environment.pnl`.
    """
    from deephedge.figures import session00_numbers

    with RESULTS.open() as fh:
        logged = [r for r in csv.DictReader(fh) if r["session"] == "00"]
    rows = session00_numbers()
    assert [(r["case"], r["hedger"]) for r in rows] == [(r["case"], r["hedger"]) for r in logged]
    for got, want in zip(rows, logged, strict=True):
        for key in ("mean", "std", "entropic"):
            if want[key] == "":
                continue
            assert got[key] == pytest.approx(float(want[key]), abs=5e-7), (got, key)
