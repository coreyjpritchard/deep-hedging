"""M11a: rough Bergomi adapter over `roughvol` at tag v0.1.0."""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

from deephedge.contracts import Vanilla
from deephedge.env import Environment
from deephedge.generators.rbergomi import RoughBergomi, RoughBergomiPricer, rbergomi_nu
from deephedge.pricing import bs_call_delta
from deephedge.registry import default_registry
from deephedge.spec import Node

pytest.importorskip("roughvol")

CALL = Vanilla("call", 100.0)
PRESET = dict(H=0.1, nu=1.5, rho=-0.7, xi0=0.04)


def test_adapter_fails_if_roughvol_signature_moves():
    """Loud break if `roughvol.rbergomi` drops what M11a calls."""
    from roughvol.estimate import rbergomi_nu as eng_nu
    from roughvol.rbergomi import RoughBergomi as Engine
    from roughvol.rbergomi import variance_swap

    sim = inspect.signature(Engine.simulate)
    assert "n_paths" in sim.parameters and "times" in sim.parameters
    assert "seed" in sim.parameters and "state0" in sim.parameters
    price = inspect.signature(Engine.price_with_error)
    for name in ("F", "K", "T", "df", "kind", "state", "n_paths", "seed"):
        assert name in price.parameters
    assert list(inspect.signature(variance_swap).parameters) == ["paths"]
    assert list(inspect.signature(eng_nu).parameters)[:2] == ["nu_per_day", "H"]


def test_shapes_seed_and_names():
    a = RoughBergomi(**PRESET).simulate(64, 30, seed=0)
    b = RoughBergomi(**PRESET).simulate(64, 30, seed=0)
    c = RoughBergomi(**PRESET).simulate(64, 30, seed=1)
    assert a.instruments.shape == (64, 31, 2)
    assert a.features.shape == (64, 31, 2)
    assert a.state["variance"].shape == (64, 31)
    assert a.state["xi"].shape == (64, 31, 31)
    assert a.instrument_names == ("spot", "variance_swap")
    assert a.feature_names == ("spot", "variance")
    np.testing.assert_array_equal(a.instruments, b.instruments)
    assert not np.array_equal(a.instruments, c.instruments)
    assert a.spot[0, 0] == pytest.approx(100.0)
    assert a.instruments[0, -1, 1] == pytest.approx(0.0)
    assert RoughBergomi().pricer() is not None


def test_spot_martingale_at_four_se():
    """E[discounted spot] = s0 at maturity, within 4 SE on 20,000 paths."""
    g = RoughBergomi(**PRESET, r=0.03, mu=0.03)
    paths = g.simulate(20_000, 30, seed=11)
    disc = paths.spot[:, -1] * paths.discount_terminal()
    se = disc.std(ddof=1) / np.sqrt(disc.size)
    assert abs(disc.mean() - g.s0) < 4 * se


def test_h_half_nu_zero_is_black_scholes_variance():
    """H = 0.5, nu = 0: variance stays at xi0 (flat forward variance)."""
    g = RoughBergomi(H=0.5, nu=0.0, rho=0.0, xi0=0.04)
    paths = g.simulate(512, 16, seed=2)
    np.testing.assert_allclose(paths.state["variance"], 0.04, rtol=0, atol=1e-12)


def test_premium_matches_roughvol_reference():
    """Conditional MC premium agrees with a direct roughvol call to 1e-12."""
    g = RoughBergomi(**PRESET, n_price_paths=2_000, price_seed=7)
    got = g.pricer().premium(CALL)
    from roughvol.rbergomi import RoughBergomi as Engine

    eng = Engine(H=0.1, nu=1.5, rho=-0.7, xi0=0.04, n_steps=30, seed=7)
    ref = eng.price_with_error(100.0, 100.0, 30 / 365, df=1.0, kind="call", n_paths=2_000, seed=7)
    assert float(got.value) == pytest.approx(float(ref.value), abs=1e-12)
    assert float(got.stderr) == pytest.approx(float(ref.stderr), abs=1e-12)
    assert got.stderr > 0


def test_benchmark_is_bs_delta_at_current_vol():
    g = RoughBergomi(**PRESET)
    paths = g.simulate(32, 10, seed=3)
    pos = g.benchmark_positions(paths)
    assert pos.shape == (32, 10, 2)
    np.testing.assert_array_equal(pos[:, :, 1], 0.0)
    for k in range(10):
        tau = g.maturity - paths.time[k]
        sig = np.sqrt(np.maximum(paths.features[:, k, 1], 0.0))
        expect = bs_call_delta(paths.spot[:, k], g.strike, tau, sig, g.r)
        np.testing.assert_allclose(pos[:, k, 0], expect, rtol=0, atol=1e-12)
    greeks = RoughBergomiPricer(g).greeks(paths, CALL)
    np.testing.assert_allclose(greeks["delta"][:, :-1, 0], pos[:, :, 0], rtol=0, atol=1e-12)


def test_rbergomi_nu_matches_roughvol():
    from roughvol.estimate import rbergomi_nu as eng

    assert rbergomi_nu(0.01, 0.1) == pytest.approx(eng(0.01, 0.1), abs=1e-14)


def test_registry_builds_real_adapter():
    built = default_registry().build(Node("generator.rough_bergomi"))
    assert isinstance(built, RoughBergomi)
    assert built.params()["H"] == 0.1
    assert built.pricer().supports(CALL)


def test_lab_preset_speed_20k_by_30():
    """Acceptance: 20k by 30 under 30 seconds on this box."""
    g = RoughBergomi(**PRESET)
    t0 = time.perf_counter()
    g.simulate(20_000, 30, seed=0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"simulate took {elapsed:.2f}s"


def test_environment_pnl_with_benchmark():
    g = RoughBergomi(**PRESET)
    env = Environment(g, 0.0)
    paths = env.simulate(2_000, 30, seed=5)
    pnl = env.pnl(paths, g.benchmark_positions(paths))
    assert pnl.shape == (2_000,)
    assert np.isfinite(pnl).all()
