"""A tiny training run: 2 000 paths, 10 steps, a few epochs, well under 20 seconds."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deephedge.hedging import bs_call_price, call_payoff, delta_hedge_pnl
from deephedge.markets import bs_paths, to_torch
from deephedge.policy import DeltaPolicy, MLPPolicy, rollout
from deephedge.risk import cvar, cvar_torch
from deephedge.train import Setup, load_policy, objective, save_policy, train

S0, K, T, SIGMA, COST, ALPHA = 100.0, 100.0, 30 / 365, 0.2, 0.005, 0.9


@pytest.fixture(scope="module")
def tiny_run():
    paths = to_torch(bs_paths(2_000, 10, T, SIGMA, s0=S0, seed=0))
    premium = float(bs_call_price(S0, K, T, SIGMA))
    payoff = call_payoff(K)

    def risk(pnl):
        return cvar_torch(pnl, ALPHA)

    torch.manual_seed(0)
    policy = MLPPolicy(K, T, width=16)
    before = objective(policy, paths, payoff, risk, COST, premium)
    result = train(policy, paths, payoff, risk, COST, epochs=40, lr=0.01, seed=0,
                   premium=premium, batch_size=500)
    return paths, premium, before, result


def test_training_reduces_objective(tiny_run):
    _, _, before, result = tiny_run
    assert result.history[-1] < before - 0.5
    assert result.wall_time < 20


def test_trained_close_to_or_better_than_delta(tiny_run):
    paths, premium, _, result = tiny_run
    delta = cvar(delta_hedge_pnl(paths.double().numpy(), K, T, SIGMA, 0.0, COST), ALPHA)
    assert result.history[-1] < delta + 0.15


def test_delta_policy_matches_numpy_delta(impl):
    p = bs_paths(300, 8, T, SIGMA, s0=S0, seed=4)
    pos = rollout(DeltaPolicy(K, T, SIGMA), torch.as_tensor(p), T).numpy()
    np.testing.assert_allclose(pos, impl.delta_positions(p, K, T, SIGMA), atol=1e-12)


def test_save_and_load_roundtrip(tmp_path):
    policy = MLPPolicy(K, T, width=8)
    setup = Setup(width=8)
    save_policy(policy, setup, tmp_path / "w.pt")
    loaded, loaded_setup = load_policy(tmp_path / "w.pt")
    s = torch.linspace(80, 120, 7)
    prev = torch.full_like(s, 0.4)
    assert torch.allclose(policy(0.01, s, prev), loaded(0.01, s, prev))
    assert loaded_setup == setup


def test_cached_default_network_beats_delta_out_of_sample():
    """The shipped weights, on 20 000 paths not used in training.

    Entropic risk SEs are about 0.01 at this sample size; the margin is far larger.
    """
    from dataclasses import replace

    from deephedge.hedging import pnl_from_positions
    from deephedge.policy import rollout_numpy
    from deephedge.risk import entropic
    from deephedge.train import DEFAULT, cached_policy

    s = DEFAULT
    paths = bs_paths(s.m_eval, s.n_steps, s.maturity, s.sigma, s0=s.s0, seed=999)
    for cost, margin in ((0.0, -0.01), (s.cost, 0.3)):
        policy, trained_for = cached_policy(replace(s, cost=cost))
        assert trained_for.cost == cost
        net = pnl_from_positions(
            paths, rollout_numpy(policy, paths, s.maturity), call_payoff(s.strike), cost,
            s.premium,
        )
        delta = delta_hedge_pnl(paths, s.strike, s.maturity, s.sigma, s.r, cost)
        assert entropic(net, s.lam) < entropic(delta, s.lam) - margin
