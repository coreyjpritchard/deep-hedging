"""M02: Paths with state and names, the generator and pricer protocols, the MC pricer.

Monte Carlo checks use 4 standard errors on 20 000 inner paths, as in
`tests/test_env.py`. Heston inner paths are stepped at the pricer's default
`dt_max` (16 steps a day, 480 on 30 days): the 30-step hedging grid prices
the call about 0.1 too high under full truncation, see `tests/test_heston.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from deephedge import env
from deephedge.contracts import Vanilla
from deephedge.generators import (
    BlackScholes,
    BlackScholesPricer,
    Generator,
    Heston,
    HestonPricer,
    Market,
    Pricer,
)
from deephedge.paths import Paths
from deephedge.pricing import MonteCarloPricer
from deephedge.registry import default_registry

INNER = 20_000
CALL = Vanilla("call", 100.0)
PUT = Vanilla("put", 100.0)
CONTRACTS = (CALL, PUT)


def _generators():
    return (
        ("generator.black_scholes", BlackScholes(r=0.03)),
        ("generator.heston", Heston(r=0.03)),
    )


# ---- Paths --------------------------------------------------------------------


def _arrays(m=3, n=4, d=2, f=1):
    rng = np.random.default_rng(0)
    instruments = rng.random((m, n + 1, d)) + 1.0
    return dict(
        time=np.linspace(0.0, 1.0, n + 1),
        instruments=instruments,
        features=instruments[:, :, :f],
        discount=np.ones(n + 1),
    )


def test_paths_defaults_keep_the_pr1_constructor():
    paths = Paths(**_arrays())
    assert paths.instrument_names == ("spot", "instrument_1")
    assert paths.feature_names == ("spot",)
    assert dict(paths.state) == {}
    assert paths.source is None
    assert env.Paths is Paths


def test_paths_state_is_validated_and_read_only():
    a = _arrays()
    paths = Paths(**a, state={"variance": np.ones((3, 5)), "xi": np.ones((3, 5, 2))})
    assert paths.state_names == ("variance", "xi")
    with pytest.raises(TypeError):
        paths.state["other"] = np.ones((3, 5))
    with pytest.raises(ValueError, match="state 'variance'"):
        Paths(**a, state={"variance": np.ones((3, 4))})
    with pytest.raises(ValueError, match="state 'v'"):
        Paths(**a, state={"v": np.ones(3)})
    with pytest.raises(ValueError, match="non-empty"):
        Paths(**a, state={"": np.ones((3, 5))})


def test_paths_names_are_validated():
    a = _arrays()
    with pytest.raises(ValueError, match="2 columns"):
        Paths(**a, instrument_names=("spot",))
    with pytest.raises(ValueError, match="distinct"):
        Paths(**a, instrument_names=("spot", "spot"))
    with pytest.raises(ValueError, match="non-empty"):
        Paths(**a, feature_names=("",))


def test_paths_to_torch_keeps_shapes_and_names():
    torch = pytest.importorskip("torch")
    paths = Heston().simulate(8, 5, seed=1)
    tp = paths.to_torch()
    assert tp.instruments.dtype == torch.float32
    assert tuple(tp.instruments.shape) == (8, 6, 2)
    assert tuple(tp.state["variance"].shape) == (8, 6)
    assert tp.instrument_names == ("spot", "variance_swap")
    assert tuple(tp.spot.shape) == (8, 6)
    assert paths.to_torch(torch.float64).features.dtype == torch.float64


# ---- generators ---------------------------------------------------------------


@pytest.mark.parametrize("type_name, generator", _generators())
def test_generators_conform_and_name_what_they_put_on_paths(type_name, generator):
    assert isinstance(generator, Generator)
    assert isinstance(generator, Market)  # deprecated alias, one release
    assert isinstance(generator.pricer(), Pricer)
    assert isinstance(MonteCarloPricer(generator), Pricer)
    assert type_name == f"generator.{generator.name}"
    paths = generator.simulate(16, 6, seed=2)
    assert paths.instrument_names == generator.instrument_names
    assert paths.feature_names == generator.feature_names
    assert paths.state_names == generator.state_names
    again = generator.simulate(16, 6, seed=2)
    np.testing.assert_array_equal(paths.instruments, again.instruments)


@pytest.mark.parametrize("type_name, generator", _generators())
def test_generator_params_are_valid_spec_params_and_round_trip(type_name, generator):
    registry = default_registry()
    params = generator.params()
    assert "strike" not in params
    resolved, problems = registry.check_params(registry.param_schema(type_name), params)
    assert problems == []
    assert resolved == params
    assert type(generator)(**params).params() == params


def test_heston_state_is_the_truncated_variance_feature():
    paths = Heston().simulate(64, 10, seed=3)
    np.testing.assert_array_equal(paths.state["variance"], paths.features[:, :, 1])
    assert paths.state["variance"].min() >= 0.0


@pytest.mark.parametrize("generator", [BlackScholes(r=0.03, mu=0.2), Heston(r=0.03, mu=0.2)])
def test_simulate_from_restarts_under_the_pricing_measure(generator):
    """Each block of inner paths starts at its own state; the drift is r, not mu."""
    start = {"spot": np.array([90.0, 110.0]), "variance": np.array([0.02, 0.09])}
    tau, n_inner = 0.05, INNER
    n = 1 if isinstance(generator, BlackScholes) else 40
    inner = generator.simulate_from(start, tau, n, n_inner, seed=4)
    assert inner.n_paths == 2 * n_inner and inner.time[-1] == pytest.approx(tau)
    spot = inner.spot.reshape(2, n_inner, n + 1)
    np.testing.assert_array_equal(spot[:, :, 0], np.repeat(start["spot"], n_inner).reshape(2, -1))
    if generator.state_names:
        v0 = inner.state["variance"][:, 0].reshape(2, n_inner)
        np.testing.assert_array_equal(v0, np.repeat(start["variance"], n_inner).reshape(2, -1))
    terminal = spot[:, :, -1] * np.exp(-0.03 * tau)
    se = terminal.std(axis=1, ddof=1) / np.sqrt(n_inner)
    assert np.all(np.abs(terminal.mean(axis=1) - start["spot"]) < 4 * se)


def test_heston_moved_with_a_shim_and_identical_paths():
    from deephedge import heston as old
    from deephedge.generators import heston as new

    assert old.Heston is new.Heston
    paths = Heston(mu=0.05, kappa_sim=2.0).simulate(8, 5, seed=9)
    shocks = np.random.default_rng(9).standard_normal((8, 5, 2))
    np.testing.assert_array_equal(
        paths.instruments, Heston(mu=0.05, kappa_sim=2.0).paths_from_shocks(shocks).instruments
    )


# ---- closed-form pricers ------------------------------------------------------


def test_closed_form_pricers_agree_with_the_market_interface():
    for generator, pricer_type in (
        (BlackScholes(r=0.03), BlackScholesPricer),
        (Heston(r=0.03), HestonPricer),
    ):
        pricer = generator.pricer()
        assert isinstance(pricer, pricer_type)
        paths = generator.simulate(32, 6, seed=5)
        assert pricer.premium(CALL).value == pytest.approx(generator.premium(), abs=1e-12)
        assert pricer.premium(CALL).stderr == 0.0
        np.testing.assert_allclose(
            pricer.price_along(paths, CALL), generator.price_along(paths), atol=1e-12
        )
        delta = pricer.greeks(paths, CALL)["delta"]
        assert delta.shape == paths.instruments.shape
        np.testing.assert_allclose(delta[:, :-1], generator.benchmark_positions(paths), atol=1e-12)


@pytest.mark.parametrize("type_name, generator", _generators())
def test_put_call_parity(type_name, generator):
    pricer = generator.pricer()
    paths = generator.simulate(16, 4, seed=6)
    tau = generator.maturity - paths.time
    forward = paths.spot - 100.0 * np.exp(-generator.r * tau)[None, :]
    np.testing.assert_allclose(
        pricer.price_along(paths, CALL) - pricer.price_along(paths, PUT), forward, atol=1e-10
    )
    gap = (
        pricer.greeks(paths, CALL)["delta"][:, :-1, 0]
        - pricer.greeks(paths, PUT)["delta"][:, :-1, 0]
    )
    np.testing.assert_allclose(gap, 1.0, atol=1e-10)
    np.testing.assert_allclose(pricer.price_along(paths, PUT)[:, -1], PUT.payoff(paths), atol=1e-12)


def test_pricers_refuse_what_they_cannot_price():
    class Digital:
        kind = "digital"
        strike = 100.0

    for pricer in (BlackScholes().pricer(), Heston().pricer(), MonteCarloPricer(BlackScholes())):
        assert not pricer.supports(Digital())
        with pytest.raises(ValueError):
            pricer.premium(Digital())


# ---- Monte Carlo pricer -------------------------------------------------------


@pytest.mark.parametrize("type_name, generator", _generators())
@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.kind)
def test_mc_premium_matches_closed_form(type_name, generator, contract):
    """MC premium on 20 000 inner paths within 4 SE of BS closed form and Heston quadrature."""
    mc = MonteCarloPricer(generator, n_inner=INNER, seed=7).premium(contract)
    exact = generator.pricer().premium(contract).value
    assert mc.stderr > 0
    assert abs(mc.value - exact) < 4 * mc.stderr


def test_mc_price_along_and_greeks_match_black_scholes():
    generator = BlackScholes(r=0.03)
    paths = generator.simulate(4, 5, seed=8)
    mc = MonteCarloPricer(generator, n_inner=INNER, seed=9)
    priced = mc.price_along_with_error(paths, CALL)
    exact = generator.pricer().price_along(paths, CALL)
    # One path ends a day out far below the strike: every inner payoff is 0, so
    # its standard error is 0 and the closed form is 2e-6. Hence the 1e-4 floor.
    assert np.all(priced.stderr[:, -1] == 0.0) and np.all(priced.stderr[:, 0] > 0.0)
    assert np.all(np.abs(priced.value - exact) < 4 * priced.stderr + 1e-4)
    np.testing.assert_array_equal(priced.value[:, -1], CALL.payoff(paths))
    greeks = mc.greeks(paths, CALL, ("delta", "gamma"))
    closed = generator.pricer().greeks(paths, CALL, ("delta", "gamma"))
    for name in ("delta", "gamma"):
        est, se, ref = greeks[name][:, :-1], greeks[f"{name}_stderr"][:, :-1], closed[name][:, :-1]
        assert np.all(np.abs(est - ref) < 4 * se + 1e-4), name


def test_mc_delta_matches_heston_quadrature_delta():
    """Spot delta at fixed variance, two paths at two dates, against the quadrature delta.

    The reference is the quadrature, kept independent of the COS grid. (The
    COS floor bug that used to miss by 0.014 here is fixed; see LOG.md, 26
    October 2026.) The swap column of the MC delta stays 0.
    """
    generator = Heston(r=0.03)
    paths = generator.simulate(2, 2, seed=10)
    mc = MonteCarloPricer(generator, n_inner=INNER, seed=11)
    greeks = mc.greeks(paths, CALL)
    for i in range(2):
        for k in range(2):
            state = Heston(
                s0=paths.spot[i, k],
                v0=paths.state["variance"][i, k],
                maturity=generator.maturity - paths.time[k],
                r=0.03,
            )
            _, delta, _ = state.quad_greeks()
            se = greeks["delta_stderr"][i, k, 0]
            assert abs(greeks["delta"][i, k, 0] - delta) < 4 * se
    assert np.all(greeks["delta"][:, :, 1] == 0.0)


def test_mc_pricer_is_seeded_and_caches_on_an_explicit_key():
    generator = BlackScholes()
    paths = generator.simulate(3, 3, seed=12)
    first = MonteCarloPricer(generator, n_inner=1_000, seed=13)
    second = MonteCarloPricer(generator, n_inner=1_000, seed=13)
    a = first.price_along_with_error(paths, CALL, cache_key=("spec", "test"))
    np.testing.assert_array_equal(a.value, second.price_along(paths, CALL))
    used = first.n_inner_simulations
    assert first.price_along_with_error(paths, CALL, cache_key=("spec", "test")) is a
    assert first.n_inner_simulations == used
    first.price_along(paths, CALL)  # no key: recomputed
    assert first.n_inner_simulations == 2 * used
    other = MonteCarloPricer(generator, n_inner=1_000, seed=14).price_along(paths, CALL)
    assert not np.array_equal(a.value[:, :-1], other[:, :-1])


def test_mc_pricer_needs_the_state_the_generator_reads():
    generator = Heston()
    paths = generator.simulate(2, 2, seed=15)
    bare = Paths(
        time=paths.time,
        instruments=paths.instruments,
        features=paths.features,
        discount=paths.discount,
    )
    with pytest.raises(ValueError, match="no state 'variance'"):
        MonteCarloPricer(generator, n_inner=100).price_along(bare, CALL)
