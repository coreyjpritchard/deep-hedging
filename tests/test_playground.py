"""Playground markets, the upload check, the recurrent hedge, and the fit loop."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from deephedge.bootstrap import BlockBootstrap, parse_closes
from deephedge.env import BlackScholes, Environment, terminal_pnl
from deephedge.hedging import call_payoff, pnl_from_positions_torch
from deephedge.heston import Heston
from deephedge.markets import bs_paths, to_torch
from deephedge.playground import (
    PlaygroundError,
    PlaygroundSpec,
    RoughBergomiStub,
    fit_policy,
    run_experiment,
    spec_id,
    spot_only,
)
from deephedge.policy import GRUPolicy, MLPPolicy, position_grid, rollout_numpy
from deephedge.risk import entropic_torch
from deephedge.train import train
from deephedge.upload import MarketUploadError, load_uploaded_market, validate_market

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "playground" / "templates" / "merton_jump.py"
EXAMPLE = ROOT / "playground" / "examples" / "daily_closes.csv"
T = 30 / 365


def test_block_bootstrap_repeats_a_historical_window():
    """One integers draw, and each block is a window of the historical returns."""
    rets = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.003])
    log_c = np.concatenate([[np.log(100.0)], np.log(100.0) + np.cumsum(rets)])
    closes = tuple(np.exp(log_c))
    market = BlockBootstrap(closes=closes, block=3, s0=50.0, strike=50.0, maturity=T, r=0.0)
    first = market.simulate(4, 6, seed=7)
    second = market.simulate(4, 6, seed=7)
    assert first.spot.shape == (4, 7)
    assert np.allclose(first.spot, second.spot)
    assert np.allclose(first.spot[:, 0], 50.0)
    rng = np.random.default_rng(7)
    starts = rng.integers(0, rets.size - 3 + 1, size=(4, 2))
    got = np.diff(np.log(first.spot), axis=1)
    for i in range(4):
        for block in range(2):
            window = rets[starts[i, block] : starts[i, block] + 3]
            assert got[i, block * 3 : (block + 1) * 3] == pytest.approx(window, abs=1e-10)


def test_parse_closes_skips_the_header_and_rejects_a_non_positive_price():
    closes = parse_closes("# comment\ndate,close\n2024-01-02,100\n2024-01-03,101\n2024-01-04,99\n")
    assert closes == (100.0, 101.0, 99.0)
    with pytest.raises(ValueError, match="positive"):
        parse_closes("date,close\n2024-01-02,100\n2024-01-03,0\n2024-01-04,99\n")


def test_example_csv_is_the_seeded_black_scholes_path():
    """80 closes, the first at 100, matching `bs_paths` with seed 1."""
    closes = np.asarray(parse_closes(EXAMPLE.read_text()))
    expected = bs_paths(1, 79, 79 / 252, 0.2, s0=100.0, mu=0.0, seed=1)[0]
    assert closes.shape == (80,)
    assert closes == pytest.approx(expected, abs=5e-9)
    assert closes[0] == pytest.approx(100.0, abs=1e-8)


def test_merton_template_passes_shapes_and_the_martingale_check():
    """m = 2000 paths, n = 10 steps, tolerance 4 standard errors."""
    source = TEMPLATE.read_text()
    market = load_uploaded_market(source, "merton_jump.py")
    message = validate_market(market, m=2000, n=10, seed=0)
    assert message.startswith("Market accepted")
    paths = market.simulate(32, 4, seed=1)
    assert paths.instruments.shape == (32, 5, 1)
    assert market.benchmark_positions(paths).shape == (32, 4, 1)
    assert market.payoff(paths).shape == (32,)


def test_upload_rejects_a_file_with_no_market():
    with pytest.raises(MarketUploadError, match="build_market"):
        load_uploaded_market("x = 1\n", "empty.py")


def test_upload_rejects_a_bad_payoff_shape():
    source = """
import numpy as np
from deephedge.env import Market, Paths

class Bad(Market):
    maturity = 1.0
    s0 = 100.0

    @property
    def n_instruments(self):
        return 1

    @property
    def n_features(self):
        return 1

    @property
    def instrument_names(self):
        return ("spot",)

    @property
    def feature_names(self):
        return ("spot",)

    def simulate(self, m, n, seed):
        spot = np.full((m, n + 1), self.s0)
        time = np.linspace(0.0, 1.0, n + 1)
        column = spot[:, :, None]
        return Paths(time, column, column, np.ones(n + 1))

    def payoff(self, paths):
        return np.zeros(1)

    def price_along(self, paths):
        return np.zeros(paths.spot.shape)

    def benchmark_positions(self, paths):
        return np.zeros((paths.spot.shape[0], paths.n_steps, 1))

    def premium(self):
        return 1.0

def build_market():
    return Bad()
"""
    market = load_uploaded_market(source, "bad.py")
    with pytest.raises(MarketUploadError, match="payoff"):
        validate_market(market, m=8, n=2, seed=0)


def test_black_scholes_drift_fails_the_martingale_check():
    """mu = 0.5 and r = 0, m = 2000. The gap is many standard errors."""
    market = BlackScholes(mu=0.5, r=0.0)
    with pytest.raises(MarketUploadError, match="Martingale check failed"):
        validate_market(market, m=2000, n=20, seed=0)


def test_black_scholes_with_mu_equal_to_r_passes():
    """m = 2000, n = 20, seed 0, tolerance 4 standard errors."""
    message = validate_market(BlackScholes(mu=0.0, r=0.0), m=2000, n=20, seed=0)
    assert message.startswith("Market accepted")


def test_rough_bergomi_stub_names_roughvol():
    market = RoughBergomiStub()
    with pytest.raises(NotImplementedError, match="roughvol"):
        market.simulate(2, 2, 0)
    spec = replace(PlaygroundSpec(), train_card="rough", test_card="bs", m=8, n_steps=2, epochs=1)
    with pytest.raises(PlaygroundError, match="roughvol"):
        run_experiment(spec)


def test_gru_rollout_resets_and_has_shape_m_by_n():
    """A hidden state planted before the call is discarded. Shape is (m, n)."""
    policy = GRUPolicy(100.0, T, hidden=8)
    paths = np.full((3, 5), 100.0)
    policy._hidden = torch.ones(3, policy.hidden)
    out = rollout_numpy(policy, paths, T)
    fresh = GRUPolicy(100.0, T, hidden=8)
    fresh.load_state_dict(policy.state_dict())
    assert out.shape == (3, 4)
    assert np.allclose(out, rollout_numpy(fresh, paths, T))


def test_position_grid_resets_the_gru():
    policy = GRUPolicy(100.0, T, hidden=8)
    spots = np.linspace(90.0, 110.0, 5)
    prevs = np.array([0.0, 0.5])
    policy._hidden = torch.ones(spots.size * prevs.size, policy.hidden)
    grid = position_grid(policy, T / 2, spots, prevs)
    fresh = GRUPolicy(100.0, T, hidden=8)
    fresh.load_state_dict(policy.state_dict())
    assert grid.shape == (2, 5)
    assert np.allclose(grid, position_grid(fresh, T / 2, spots, prevs))


def test_fit_policy_matches_train_at_zero_rate():
    """Same weights, r = 0, plain rate, 2 epochs, m = 256, n = 5, width 4."""
    market = BlackScholes()
    paths = market.simulate(256, 5, seed=0)
    premium = market.premium()
    torch.manual_seed(1)
    policy_a = MLPPolicy(100.0, T, width=4, depth=2)
    policy_b = MLPPolicy(100.0, T, width=4, depth=2)
    policy_b.load_state_dict({k: v.clone() for k, v in policy_a.state_dict().items()})
    spot = to_torch(paths.spot)
    trained = train(
        policy_a, spot, call_payoff(100.0), lambda pnl: entropic_torch(pnl, 1.0),
        0.005, 2, 0.005, 0, premium=premium, batch_size=128,
    )
    fitted = fit_policy(
        policy_b, paths.spot, 100.0, 0.0, 0.005, 1.0, 2, 0.005, 0, premium, 128,
    )
    assert fitted.history == pytest.approx(trained.history, abs=1e-5)
    pairs = zip(policy_a.state_dict().values(), policy_b.state_dict().values(), strict=True)
    for left, right in pairs:
        assert torch.allclose(left, right, atol=1e-5)


def test_discounted_training_pnl_matches_the_environment():
    """r = 0.03, m = 32, n = 5. Float32 P&L against the environment's float64 P&L."""
    rate = 0.03
    market = BlackScholes(r=rate, mu=rate)
    paths = market.simulate(32, 5, seed=3)
    positions = market.benchmark_positions(paths)[:, :, 0]
    env = Environment(market, 0.005).pnl(paths, positions)
    spot = to_torch(paths.spot)
    grid = torch.linspace(0.0, market.maturity, paths.spot.shape[1])
    price = spot * torch.exp(-rate * grid)
    strike = market.strike * math.exp(-rate * market.maturity)
    torch_pnl = pnl_from_positions_torch(
        price, torch.as_tensor(positions, dtype=torch.float32), call_payoff(strike),
        0.005, market.premium(), exact=True,
    )
    assert np.allclose(torch_pnl.double().numpy(), env, atol=1e-4)


def test_delta_terminal_pnl_matches_the_environment_on_black_scholes():
    market = BlackScholes()
    paths = market.simulate(64, 8, seed=2)
    positions = market.benchmark_positions(paths)[:, :, 0]
    direct = terminal_pnl(paths, positions, market.payoff(paths), 0.005, market.premium())
    env = Environment(market, 0.005).pnl(paths, positions)
    assert np.allclose(direct, env)


def test_spot_only_scores_a_heston_spot_hedge():
    """Heston has two instruments. The spot-only path accepts a position of shape (m, n)."""
    market = Heston()
    paths = market.simulate(16, 4, seed=1)
    reduced = spot_only(paths)
    assert reduced.instruments.shape == (16, 5, 1)
    assert np.allclose(reduced.spot, paths.spot)
    pnl = terminal_pnl(reduced, np.zeros((16, 4)), market.payoff(paths), 0.005, market.premium())
    assert pnl.shape == (16,)
    assert np.isfinite(pnl).all()


def test_default_picture_trains_black_scholes_and_tests_heston():
    spec = PlaygroundSpec()
    assert spec.train_card == "bs"
    assert spec.test_card == "heston"
    assert spec.hedger == "mlp"
    assert spec_id(spec) == spec_id(PlaygroundSpec())


def test_small_experiment_trains_and_scores_both_markets():
    """m = 64, n = 4, 1 epoch. Not the playground's default budget."""
    spec = replace(PlaygroundSpec(), m=64, n_steps=4, epochs=1, batch_size=32)
    result = run_experiment(spec)
    assert len(result.history) == 1
    assert result.same_market is False
    assert result.pnl_test_chosen.shape == (64,)
    assert result.test_chosen["m"] == 64
    assert math.isfinite(result.premium_train)
    assert math.isfinite(result.premium_test)


def test_run_experiment_reports_a_bad_python_file():
    spec = replace(
        PlaygroundSpec(),
        train_card="py",
        train_source="x = 1\n",
        train_filename="bad.py",
        m=16,
        n_steps=2,
        epochs=1,
    )
    with pytest.raises(PlaygroundError, match="build_market"):
        run_experiment(spec)
