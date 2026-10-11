"""Whalley-Wilmott half-width: units, the cube root, and the edge-trading rule."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import norm

from deephedge.band import (
    band_positions,
    bs_call_gamma,
    describe_rule,
    fit_power_exponent,
    score_hedgers,
    whalley_wilmott_half_width,
)
from deephedge.env import BlackScholes, bs_call_delta
from deephedge.policy import GammaPolicy
from deephedge.risk import entropic, entropic_se, mean_se

S0, K, SIGMA, T = 100.0, 100.0, 0.2, 30 / 365


def test_half_width_matches_the_discounted_cube_root():
    """Section 4 of Whalley and Wilmott, with mu = r, at the session-00 point."""
    tau = T / 2
    cost = 0.005
    gamma_risk = 1.0
    r = 0.03
    width = whalley_wilmott_half_width(S0, tau, cost, gamma_risk, K, SIGMA, r)
    gamma = bs_call_gamma(S0, K, tau, SIGMA, r)
    expected = (1.5 * cost * S0 * math.exp(-r * tau) * gamma**2 / gamma_risk) ** (1.0 / 3.0)
    assert width == pytest.approx(expected, rel=0, abs=1e-12)
    assert width > 0.0


def test_zero_rate_matches_the_session_00_scale():
    """At r = 0 and 50 bp the half-width at T/2 is the 0.19 quoted in the log."""
    width = whalley_wilmott_half_width(S0, T / 2, 0.005, 1.0, K, SIGMA, 0.0)
    assert width == pytest.approx(0.19, abs=0.005)


def test_width_is_exactly_cube_root_in_cost():
    costs = np.array([0.0005, 0.001, 0.0025, 0.005, 0.02])
    widths = [
        float(whalley_wilmott_half_width(S0, T / 2, c, 1.0, K, SIGMA, 0.0)) for c in costs
    ]
    fit = fit_power_exponent(costs, widths)
    assert fit["exponent"] == pytest.approx(1.0 / 3.0, abs=1e-10)
    assert fit["se"] == pytest.approx(0.0, abs=1e-8)
    assert widths[-1] / widths[0] == pytest.approx((0.02 / 0.0005) ** (1.0 / 3.0), rel=1e-10)


def test_gamma_has_the_black_scholes_units():
    tau = T / 2
    sd = SIGMA * math.sqrt(tau)
    d1 = (math.log(S0 / K) + 0.5 * SIGMA**2 * tau) / sd
    expected = norm.pdf(d1) / (S0 * sd)
    assert bs_call_gamma(S0, K, tau, SIGMA) == pytest.approx(expected)
    assert bs_call_gamma(S0, K, 0.0, SIGMA) == 0.0


def test_band_trades_to_the_edge_and_holds_inside():
    """One path. Delta is 0.5, half-width is 0.1. Flat start buys to 0.4, then holds."""
    spot = np.full((1, 4), S0)
    time = np.linspace(0.0, T, 4)
    # Freeze delta at 0.5 by using a width function and a spot where we check clip logic
    # through a constant half-width and the true delta. Hand-check the first trade only.
    delta0 = float(bs_call_delta(S0, K, T, SIGMA, 0.0))
    width = 0.1
    positions = band_positions(spot, time, K, SIGMA, 0.0, width)
    assert positions[0, 0] == pytest.approx(delta0 - width)
    # Second date: spot unchanged, delta a bit higher, the holding is inside or on the edge.
    delta1 = float(bs_call_delta(S0, K, T - time[1], SIGMA, 0.0))
    expected = float(np.clip(positions[0, 0], delta1 - width, delta1 + width))
    assert positions[0, 1] == pytest.approx(expected)


def test_describe_rule_recovers_a_sharp_band():
    class Sharp(torch.nn.Module):
        def forward(self, t, S, prev):
            del t, S
            return torch.clamp(prev, 0.3, 0.7)

    rule = describe_rule(
        Sharp(),
        t=T / 2,
        spots=np.array([90.0, 100.0, 110.0]),
        strike=K,
        maturity=T,
        sigma=SIGMA,
        r=0.0,
        prevs=np.linspace(0.0, 1.0, 101),
    )
    assert rule["centre"] == pytest.approx(0.5, abs=0.02)
    assert rule["no_trade_half_width"] == pytest.approx(0.2, abs=0.02)
    assert rule["sharp_half_width"] == pytest.approx(0.2, abs=0.02)
    assert rule["slope"] == pytest.approx(1.0, abs=0.05)


def test_score_hedgers_names_the_three_bands():
    market = BlackScholes()
    paths = market.simulate(32, 4, seed=0)
    scores, books = score_hedgers(market, paths, 0.005, 1.0, {}, 0.1)
    assert set(scores) == {"delta", "ww", "fixed"}
    assert books["ww"].shape == (32, 4)
    assert scores["delta"]["n"] == 32


def test_gamma_policy_returns_a_position():
    policy = GammaPolicy(K, T, SIGMA, r=0.0)
    spot = torch.linspace(90, 110, 5)
    prev = torch.zeros(5)
    out = policy(T / 2, spot, prev)
    assert tuple(out.shape) == (5,)


def test_entropic_standard_error_matches_the_delta_method():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=-0.5, scale=0.4, size=4_000)
    lam = 1.0
    z = np.exp(-lam * x)
    expected = z.std(ddof=1) / (lam * z.mean() * math.sqrt(x.size))
    assert entropic_se(x, lam) == pytest.approx(expected, rel=1e-12)
    assert mean_se(x) == pytest.approx(x.std(ddof=1) / math.sqrt(x.size))
    assert entropic(x, lam) < 0.0 or entropic(x, lam) > 0.0
