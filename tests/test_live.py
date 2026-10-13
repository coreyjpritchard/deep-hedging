"""The live ledger matches terminal P&L, and the band sits still inside its edges."""

from __future__ import annotations

import numpy as np
import pytest

from deephedge.band import band_positions
from deephedge.env import BlackScholes
from deephedge.heston import Heston
from deephedge.live import (
    LIVE_HESTON_XI,
    _greeks,
    clip_call_delta,
    clip_positions,
    describe,
    draw_batch,
    draw_live,
    market_by_name,
    score_books,
    simulate_live,
    terminal_of,
)
from deephedge.risk import entropic


def _wealth(path, book, k: int) -> float:
    return float(book.cash[k] + book.position[k] * path.spot_disc[k])


def test_position_stays_inside_and_is_pushed_to_the_edge():
    calm = clip_positions(np.array([0.50, 0.48, 0.47, 0.45]), 0.10)
    assert calm == pytest.approx([0.40, 0.40, 0.40, 0.40])
    pushed = clip_positions(np.array([0.50, 0.48, 0.70]), 0.10)
    assert pushed == pytest.approx([0.40, 0.40, 0.60])
    assert clip_positions(np.array([0.2, 0.5, 0.9]), 0.0) == pytest.approx([0.2, 0.5, 0.9])
    rows = np.vstack([np.array([0.50, 0.48, 0.47]), np.array([0.20, 0.80, 0.10])])
    both = clip_positions(rows, 0.10)
    assert both[0] == pytest.approx(clip_positions(rows[0], 0.10))
    assert both[1] == pytest.approx(clip_positions(rows[1], 0.10))
    with pytest.raises(ValueError):
        clip_positions(np.array([0.5]), -0.1)


def test_black_scholes_band_matches_band_positions_and_terminal_pnl():
    market = BlackScholes(r=0.02, mu=0.02)
    n, seed, width, cost = 8, 4, 0.05, 0.005
    paths = market.simulate(1, n, seed)
    expected = band_positions(
        paths.spot, paths.time, market.strike, market.sigma, market.r, width
    )
    path = simulate_live(market, n, seed, width, cost)
    assert path.band.position[:-1] == pytest.approx(expected[0])
    bench = market.benchmark_positions(paths)[0, :, 0]
    assert path.delta_hedge.position[:-1] == pytest.approx(bench)
    for book, held in (
        (path.band, path.band.position[:-1]),
        (path.delta_hedge, path.delta_hedge.position[:-1]),
    ):
        pnl = terminal_of(market, paths, held[None, :], cost)
        assert book.pnl[-1] == pytest.approx(float(pnl[0]), abs=1e-10)
        assert book.position[-1] == pytest.approx(0.0)
        for k in range(n):
            wealth = path.premium + book.gains[k] - book.cost[k]
            assert _wealth(path, book, k) == pytest.approx(wealth, abs=1e-10)
            marked = wealth - path.discount[k] * path.price[k]
            assert book.pnl[k] == pytest.approx(marked, abs=1e-10)
        closed = path.premium + book.gains[-1] - book.cost[-1] - path.payoff_disc
        assert book.cash[-1] == pytest.approx(closed, abs=1e-10)
        assert book.pnl[-1] == pytest.approx(book.cash[-1], abs=1e-10)


def test_heston_ledger_matches_terminal_pnl():
    market = Heston()
    n, seed, width, cost = 5, 2, 0.04, 0.01
    path = simulate_live(market, n, seed, width, cost)
    paths = market.simulate(1, n, seed)
    assert path.variance is not None
    assert path.variance[0] == pytest.approx(market.v0)
    for book in (path.band, path.delta_hedge):
        pnl = terminal_of(market, paths, book.position[:-1][None, :], cost)
        assert book.pnl[-1] == pytest.approx(float(pnl[0]), abs=1e-9)
        assert np.all(np.diff(book.cost) >= -1e-12)


def test_a_wide_band_does_not_trade_and_zero_width_tracks_delta():
    market = market_by_name("Black-Scholes")
    wide = simulate_live(market, 4, 0, half_width=5.0, cost=0.01)
    assert not wide.band.rebalanced[0]
    assert "does not trade" in describe(wide, 0)
    tight = simulate_live(market, 4, 0, half_width=0.0, cost=0.01)
    assert "half-width is zero" in describe(tight, 0)
    maturity = describe(tight, tight.n)
    assert maturity.startswith("Maturity.")
    assert f"{tight.band.pnl[-1]:.3f}" in maturity
    assert f"{tight.delta_hedge.pnl[-1]:.3f}" in maturity


def test_batch_score_uses_the_same_terminal_pnl():
    market = market_by_name("Black-Scholes")
    path = simulate_live(market, 6, 9, 0.08, 0.005)
    paths = market.simulate(40, 6, 11)
    _price, delta = _greeks(market, paths)
    held = clip_positions(clip_call_delta(delta)[:, :-1], path.half_width)
    pnl = terminal_of(market, paths, held, path.cost)
    score = score_books(
        market, 6, 40, 11, path.half_width, path.cost, token=path.token, lam=1.0
    )
    assert score.pnl_band == pytest.approx(pnl)
    assert score.entropic_band == pytest.approx(entropic(pnl, 1.0))
    assert score.m == 40
    fig = draw_live(path, 3)
    spot = next(trace for trace in fig.data if trace.name == "spot")
    assert list(spot.x) == pytest.approx(path.time[:4])
    y0, y1 = fig.layout.yaxis.range
    assert y0 < float(np.min(path.spot))
    assert y1 > float(np.max(path.spot))
    batch = draw_batch(path, score)
    names = {trace.name for trace in batch.data}
    assert names == {"band", "delta hedging"}


def test_clip_call_delta_bounds_a_vanilla_call():
    out = clip_call_delta(np.array([-0.01, 0.4, 1.016, 1.0]))
    assert out == pytest.approx([0.0, 0.4, 1.0, 1.0])


def test_research_heston_pins_variance_and_the_cosine_delta_is_clipped():
    market = Heston()
    assert market.xi == pytest.approx(2.0)
    n, seed = 30, 7
    paths = market.simulate(1, n, seed)
    variance = paths.features[0, :, 1]
    assert float(np.mean(variance <= 1e-12)) > 0.5
    _price, raw = _greeks(market, paths)
    assert float(raw[0, 16]) > 1.01
    assert float(clip_call_delta(raw)[0, 16]) == pytest.approx(1.0)
    path = simulate_live(market, n, seed, 0.10, 0.005)
    assert path.variance is not None
    assert path.variance[16] == pytest.approx(0.0)
    assert path.delta[16] == pytest.approx(1.0)
    assert path.delta_raw[16] > 1.01
    text = describe(path, 16)
    assert "truncation floor" in text
    assert "cosine delta" in text


def test_live_heston_preset_keeps_variance_off_the_floor():
    market = market_by_name("Heston")
    assert market.xi == pytest.approx(LIVE_HESTON_XI)
    assert market.xi == pytest.approx(0.5)
    path = simulate_live(market, 30, 7, 0.10, 0.005)
    assert path.variance is not None
    assert path.variance[16] > 1e-4
    assert np.all(path.delta >= -1e-12)
    assert np.all(path.delta <= 1.0 + 1e-12)
    assert "xi = 0.5" in path.delta_note
