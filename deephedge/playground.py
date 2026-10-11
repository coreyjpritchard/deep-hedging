"""Train a hedge in one market and score it in another.

The playground app in `playground/app.py` is a thin marimo shell over
these functions. Paths, costs, P&L and networks are the same objects the
sessions use. The hedge trades the spot only. A Heston path also carries
a variance swap; that swap is not in the book, and the delta is the
Black-Scholes delta with volatility sqrt(v0), not the characteristic
function hedge.

`fit_policy` is `deephedge.train.train` when the rate is zero and the
prices passed in are the spot: the network sees the spot, and the P&L
is charged on the spot. When the rate is positive the network still sees
the raw spot, and the P&L is charged on the discounted spot. The payoff
identity is `discount_T * max(S_T - K, 0) = max(discount_T * S_T - K *
discount_T, 0)`.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from deephedge.band import band_positions, whalley_wilmott_half_width
from deephedge.bootstrap import BlockBootstrap
from deephedge.costs import FixedPlusProportional, ProportionalCost, QuadraticCost, as_cost
from deephedge.env import BlackScholes, Market, Paths, bs_call_delta, terminal_pnl
from deephedge.hedging import call_payoff, delta_positions, pnl_from_positions_torch
from deephedge.heston import Heston
from deephedge.markets import to_torch
from deephedge.policy import GRUPolicy, MLPPolicy, position_grid, rollout, rollout_numpy
from deephedge.risk import cvar, entropic, entropic_se, entropic_torch, mean_se
from deephedge.train import TrainResult, objective
from deephedge.upload import MarketUploadError, load_uploaded_market, validate_market

__all__ = [
    "CARD_NAMES",
    "HEDGER_NAMES",
    "ExperimentResult",
    "PlaygroundError",
    "PlaygroundSpec",
    "RoughBergomiStub",
    "fit_policy",
    "hedging_sigma",
    "run_experiment",
    "spec_id",
    "spec_label",
    "spot_only",
]

CARD_NAMES = {
    "bs": "Black-Scholes",
    "heston": "Heston",
    "rough": "Rough Bergomi",
    "csv": "CSV bootstrap",
    "py": "Python market",
}

HEDGER_NAMES = {
    "delta": "delta",
    "ww": "Whalley-Wilmott band",
    "fixed": "fixed band",
    "mlp": "feed-forward network",
    "gru": "recurrent network",
}


class PlaygroundError(Exception):
    """A setup the app can show as a sentence."""


@dataclass(frozen=True)
class RoughBergomiStub(Market):
    """Parameters for rough Bergomi, with no simulator yet.

    The variance has memory of Hurst index `hurst`. The paths will come
    from the roughvol package. `simulate` raises `NotImplementedError`
    and names that package. The playground catches it.
    """

    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 30 / 365
    r: float = 0.0
    mu: float = 0.0
    hurst: float = 0.1
    eta: float = 1.5
    rho: float = -0.7
    xi: float = 0.3
    v0: float = 0.04

    def _missing(self) -> None:
        raise NotImplementedError(
            "Rough Bergomi is not simulated in this playground. "
            "The card is ready for the roughvol package "
            f"(H={self.hurst:g}, eta={self.eta:g}, rho={self.rho:g}, "
            f"xi={self.xi:g}, v0={self.v0:g}). "
            "roughvol draws the variance with memory of Hurst index H."
        )

    @property
    def n_instruments(self) -> int:
        return 1

    @property
    def n_features(self) -> int:
        return 1

    @property
    def instrument_names(self) -> tuple[str, ...]:
        return ("spot",)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("spot",)

    def simulate(self, m: int, n: int, seed: int) -> Paths:
        del m, n, seed
        self._missing()

    def payoff(self, paths: Paths) -> np.ndarray:
        del paths
        self._missing()

    def price_along(self, paths: Paths) -> np.ndarray:
        del paths
        self._missing()

    def benchmark_positions(self, paths: Paths) -> np.ndarray:
        del paths
        self._missing()

    def premium(self) -> float:
        self._missing()


@dataclass(frozen=True)
class PlaygroundSpec:
    """One run of the playground. Defaults are the picture the app opens on.

    Train in Black-Scholes, test in Heston, feed-forward network, 50 bp,
    2 000 paths, 30 steps, 20 epochs. Seeds are 0 for training and 12345
    for the test, the same split as the sessions.
    """

    train_card: str = "bs"
    test_card: str = "heston"
    s0: float = 100.0
    strike: float = 100.0
    maturity_days: float = 30.0
    r: float = 0.0
    sigma: float = 0.2
    mu: float = 0.0
    kappa: float = 1.0
    theta: float = 0.04
    xi: float = 2.0
    rho: float = -0.7
    v0: float = 0.04
    hurst: float = 0.1
    eta: float = 1.5
    bergomi_rho: float = -0.7
    bergomi_xi: float = 0.3
    block: int = 5
    demean: bool = False
    hedger: str = "mlp"
    cost_family: str = "proportional"
    cost_bp: float = 50.0
    fixed_cash: float = 0.01
    impact: float = 0.01
    half_width: float = 0.10
    smooth_width: float = 0.05
    m: int = 2000
    n_steps: int = 30
    epochs: int = 20
    width: int = 16
    depth: int = 2
    hidden: int = 16
    batch_size: int = 500
    lr: float = 0.005
    lam: float = 1.0
    alpha: float = 0.95
    seed_train: int = 0
    seed_test: int = 12345
    train_closes: tuple[float, ...] | None = None
    test_closes: tuple[float, ...] | None = None
    train_source: str | None = None
    test_source: str | None = None
    train_filename: str = "train.py"
    test_filename: str = "test.py"


@dataclass(frozen=True)
class ExperimentResult:
    """Numbers and arrays the app plots. The policy weights are not kept."""

    label: str
    spec_id: str
    hedger: str
    hedger_name: str
    train_name: str
    test_name: str
    history: tuple[float, ...]
    wall_time: float
    same_market: bool
    premium_train: float
    premium_test: float
    sigma_train: float
    sigma_test: float
    sigma_note_train: str
    sigma_note_test: str
    train_chosen: dict
    train_delta: dict
    test_chosen: dict
    test_delta: dict
    pnl_train_chosen: np.ndarray
    pnl_test_chosen: np.ndarray
    pnl_test_delta: np.ndarray
    sample_time: np.ndarray
    sample_spot: np.ndarray
    sample_chosen: np.ndarray
    sample_delta: np.ndarray
    band_spots: np.ndarray
    band_delta: np.ndarray
    band_half_width: np.ndarray
    band_fixed_half: float
    band_prevs: np.ndarray
    band_response: np.ndarray
    band_time: float
    ww_applies: bool
    notes: tuple[str, ...]


def spec_id(spec: PlaygroundSpec) -> str:
    """Short hash of the settings, so the app can tell a stale picture."""
    payload = json.dumps(asdict(spec), sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def spec_label(spec: PlaygroundSpec) -> str:
    """One line naming the cards, the hedge and the sample size."""
    cost = {
        "proportional": f"{spec.cost_bp:g} bp proportional",
        "fixed": f"{spec.cost_bp:g} bp plus fixed {spec.fixed_cash:g}",
        "quadratic": f"quadratic impact {spec.impact:g}",
    }.get(spec.cost_family, spec.cost_family)
    return (
        f"Train {CARD_NAMES.get(spec.train_card, spec.train_card)}, "
        f"test {CARD_NAMES.get(spec.test_card, spec.test_card)}, "
        f"{HEDGER_NAMES.get(spec.hedger, spec.hedger)}, {cost}, "
        f"{spec.m} paths, {spec.n_steps} steps"
    )


def _json_default(value):
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def spot_only(paths: Paths) -> Paths:
    """A one-instrument path whose instrument is the spot.

    Time and discount are copied. Heston keeps a variance swap in column
    1. Scoring a spot hedge needs a path with one instrument, so a
    position of shape `(m, n)` matches.
    """
    spot = np.asarray(paths.spot, dtype=float)
    return Paths(
        time=np.asarray(paths.time, dtype=float),
        instruments=spot[:, :, None],
        features=spot[:, :, None],
        discount=np.asarray(paths.discount, dtype=float),
    )


def hedging_sigma(market: Market) -> tuple[float, str]:
    """Black-Scholes volatility used for delta and for the band, and why."""
    if isinstance(market, BlackScholes):
        return float(market.sigma), "the Black-Scholes volatility of this card"
    if isinstance(market, Heston):
        return float(math.sqrt(market.v0)), (
            "the Black-Scholes volatility that matches the first instant, the square root of v0"
        )
    if isinstance(market, BlockBootstrap):
        return float(market.sigma), (
            "the annualised volatility of the log returns used to build the paths"
        )
    if isinstance(market, RoughBergomiStub):
        return float(math.sqrt(max(market.v0, 0.0))), (
            "the square root of v0; rough Bergomi is not simulated here"
        )
    if hasattr(market, "sigma"):
        return float(market.sigma), "the sigma attribute of this market"
    if hasattr(market, "v0"):
        return float(math.sqrt(max(float(market.v0), 0.0))), "the square root of v0"
    return 0.2, "20 percent, because this market does not name a volatility"


def make_cost(spec: PlaygroundSpec):
    """The cost model charged on the spot."""
    rate = float(spec.cost_bp) / 1e4
    if spec.cost_family == "proportional":
        return ProportionalCost(rate)
    if spec.cost_family == "fixed":
        return FixedPlusProportional(rate, float(spec.fixed_cash), width=float(spec.smooth_width))
    if spec.cost_family == "quadratic":
        return QuadraticCost(float(spec.impact))
    raise PlaygroundError(
        "Cost model must be proportional, fixed plus proportional, or quadratic."
    )


def band_epsilon(cost) -> float | None:
    """Proportional rate inside the Whalley-Wilmott formula, if there is one."""
    model = as_cost(cost)
    if isinstance(model, QuadraticCost):
        return None
    if isinstance(model, (ProportionalCost, FixedPlusProportional)):
        rate = model.rate
        if isinstance(rate, tuple):
            return float(rate[0])
        return float(rate)
    return None


def fit_policy(
    policy: nn.Module,
    spot: np.ndarray,
    strike: float,
    r: float,
    cost,
    lam: float,
    epochs: int,
    lr: float,
    seed: int,
    premium: float,
    batch_size: int,
    on_epoch: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """Minimise entropic risk. At r = 0 and a plain rate this matches `train`.

    The network is shown `spot`. The P&L prices are `spot` when r is 0
    and the discounted spot otherwise. `history` is the exact full-sample
    objective after each epoch. `on_epoch(epoch, value)` runs after that
    number is stored, with `epoch` starting at 1.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    maturity = float(policy.maturity)
    spot_t = to_torch(spot)
    rate = float(r)

    def risk_fn(pnl: torch.Tensor) -> torch.Tensor:
        return entropic_torch(pnl, lam)

    if rate == 0.0:
        price_t = spot_t
        payoff_fn = call_payoff(float(strike))
    else:
        grid = torch.linspace(0.0, maturity, spot_t.shape[1], dtype=spot_t.dtype)
        price_t = spot_t * torch.exp(-rate * grid)
        payoff_fn = call_payoff(float(strike) * math.exp(-rate * maturity))
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    m_paths = spot_t.shape[0]
    history: list[float] = []
    start = time.perf_counter()
    for epoch in range(epochs):
        order = torch.randperm(m_paths, generator=generator)
        for i in range(0, m_paths, batch_size):
            chosen = order[i : i + batch_size]
            positions = rollout(policy, spot_t[chosen], maturity)
            pnl = pnl_from_positions_torch(
                price_t[chosen], positions, payoff_fn, cost, premium, exact=False
            )
            loss = risk_fn(pnl)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        scheduler.step()
        if rate == 0.0:
            value = objective(policy, spot_t, payoff_fn, risk_fn, cost, premium)
        else:
            value = _objective_on_spot(
                policy, spot_t, price_t, payoff_fn, risk_fn, cost, premium
            )
        history.append(value)
        if on_epoch is not None:
            on_epoch(epoch + 1, value)
    return TrainResult(policy, history, time.perf_counter() - start)


def _objective_on_spot(policy, spot_t, price_t, payoff_fn, risk_fn, cost, premium) -> float:
    """Exact entropic risk. The network sees `spot_t` and the P&L uses `price_t`."""
    with torch.no_grad():
        positions = rollout(policy, spot_t, policy.maturity)
        pnl = pnl_from_positions_torch(price_t, positions, payoff_fn, cost, premium, exact=True)
        return float(risk_fn(pnl))


def _start_near_half(policy: MLPPolicy) -> None:
    """Output half a share before any training, same idea as the Heston net."""
    last = policy.net[-1]
    nn.init.zeros_(last.weight)
    nn.init.constant_(last.bias, 0.5)


def run_experiment(
    spec: PlaygroundSpec,
    on_epoch: Callable[[int, float], None] | None = None,
) -> ExperimentResult:
    """Simulate, fit when the hedge is a network, and score train and test."""
    started = time.perf_counter()
    _check_spec(spec)
    train_market, train_file_note = _build_market(spec, "train")
    test_market, test_file_note = _build_market(spec, "test")
    notes = _market_notes(spec, train_file_note, test_file_note)
    cost = make_cost(spec)
    premium_train = _premium(train_market)
    premium_test = _premium(test_market)
    sigma_train, sigma_note_train = hedging_sigma(train_market)
    sigma_test, sigma_note_test = hedging_sigma(test_market)
    train_paths = _simulate(train_market, spec.m, spec.n_steps, spec.seed_train)
    policy, history = _fit(spec, train_market, train_paths, cost, premium_train, on_epoch)
    test_paths = _simulate(test_market, spec.m, spec.n_steps, spec.seed_test)
    train_chosen_pos = _positions(spec, train_market, train_paths, policy, cost, sigma_train)
    train_delta_pos = _delta(train_market, train_paths, sigma_train)
    test_chosen_pos = _positions(spec, test_market, test_paths, policy, cost, sigma_test)
    test_delta_pos = _delta(test_market, test_paths, sigma_test)
    pnl_train_chosen = _pnl(train_market, train_paths, train_chosen_pos, cost, premium_train)
    pnl_train_delta = _pnl(train_market, train_paths, train_delta_pos, cost, premium_train)
    pnl_test_chosen = _pnl(test_market, test_paths, test_chosen_pos, cost, premium_train)
    pnl_test_delta = _pnl(test_market, test_paths, test_delta_pos, cost, premium_train)
    band = _band(spec, test_market, policy, cost, sigma_test)
    notes.extend(_result_notes(spec, cost, sigma_note_train, sigma_note_test))
    return ExperimentResult(
        label=spec_label(spec),
        spec_id=spec_id(spec),
        hedger=spec.hedger,
        hedger_name=HEDGER_NAMES.get(spec.hedger, spec.hedger),
        train_name=_card_name(spec, "train", train_market),
        test_name=_card_name(spec, "test", test_market),
        history=tuple(history),
        wall_time=time.perf_counter() - started,
        same_market=_same_market(spec),
        premium_train=premium_train,
        premium_test=premium_test,
        sigma_train=sigma_train,
        sigma_test=sigma_test,
        sigma_note_train=sigma_note_train,
        sigma_note_test=sigma_note_test,
        train_chosen=_score(pnl_train_chosen, spec.alpha, spec.lam),
        train_delta=_score(pnl_train_delta, spec.alpha, spec.lam),
        test_chosen=_score(pnl_test_chosen, spec.alpha, spec.lam),
        test_delta=_score(pnl_test_delta, spec.alpha, spec.lam),
        pnl_train_chosen=pnl_train_chosen,
        pnl_test_chosen=pnl_test_chosen,
        pnl_test_delta=pnl_test_delta,
        sample_time=np.asarray(test_paths.time, dtype=float),
        sample_spot=np.asarray(test_paths.spot[0], dtype=float),
        sample_chosen=np.asarray(test_chosen_pos[0], dtype=float),
        sample_delta=np.asarray(test_delta_pos[0], dtype=float),
        band_spots=band[0],
        band_delta=band[1],
        band_half_width=band[2],
        band_fixed_half=float(spec.half_width),
        band_prevs=band[3],
        band_response=band[4],
        band_time=band[5],
        ww_applies=band[6],
        notes=tuple(notes),
    )


def _check_spec(spec: PlaygroundSpec) -> None:
    if spec.train_card not in CARD_NAMES or spec.test_card not in CARD_NAMES:
        raise PlaygroundError("Each slot needs a known market card.")
    if spec.hedger not in HEDGER_NAMES:
        raise PlaygroundError("The hedge must be delta, a band, or a network.")
    if spec.m < 2 or spec.n_steps < 1:
        raise PlaygroundError("Use at least two paths and one rehedging step.")
    if spec.hedger in ("mlp", "gru") and spec.epochs < 1:
        raise PlaygroundError("A network needs at least one epoch.")
    if spec.half_width < 0.0 or spec.cost_bp < 0.0 or spec.fixed_cash < 0.0 or spec.impact < 0.0:
        raise PlaygroundError("Costs and the fixed half-width must be non-negative.")
    if spec.lam <= 0.0:
        raise PlaygroundError("Entropic risk aversion must be positive.")
    maturity = spec.maturity_days / 365.0
    if spec.s0 <= 0.0 or spec.strike <= 0.0 or maturity <= 0.0:
        raise PlaygroundError("Spot, strike and maturity must be positive.")


def _build_market(spec: PlaygroundSpec, slot: str) -> tuple[Market, str | None]:
    try:
        return _construct_market(spec, slot)
    except ValueError as exc:
        raise PlaygroundError(str(exc)) from exc


def _construct_market(spec: PlaygroundSpec, slot: str) -> tuple[Market, str | None]:
    card = spec.train_card if slot == "train" else spec.test_card
    maturity = float(spec.maturity_days) / 365.0
    if card == "bs":
        return BlackScholes(
            s0=spec.s0, strike=spec.strike, maturity=maturity,
            sigma=spec.sigma, r=spec.r, mu=spec.mu,
        ), None
    if card == "heston":
        return Heston(
            s0=spec.s0, strike=spec.strike, maturity=maturity, r=spec.r, mu=spec.mu,
            kappa=spec.kappa, theta=spec.theta, xi=spec.xi, rho=spec.rho, v0=spec.v0,
        ), None
    if card == "rough":
        return RoughBergomiStub(
            s0=spec.s0, strike=spec.strike, maturity=maturity, r=spec.r, mu=spec.mu,
            hurst=spec.hurst, eta=spec.eta, rho=spec.bergomi_rho, xi=spec.bergomi_xi,
            v0=spec.v0,
        ), None
    if card == "csv":
        closes = spec.train_closes if slot == "train" else spec.test_closes
        if not closes:
            raise PlaygroundError(
                "The CSV card needs a file of closes, or the bundled example."
            )
        try:
            return BlockBootstrap(
                closes=tuple(closes), s0=spec.s0, strike=spec.strike, maturity=maturity,
                r=spec.r, block=int(spec.block), demean=bool(spec.demean),
            ), None
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc
    if card == "py":
        source = spec.train_source if slot == "train" else spec.test_source
        filename = spec.train_filename if slot == "train" else spec.test_filename
        if not source:
            raise PlaygroundError(
                "The Python card needs a .py file, or the Merton template."
            )
        try:
            market = load_uploaded_market(source, filename)
            note = validate_market(market)
        except MarketUploadError as exc:
            raise PlaygroundError(str(exc)) from exc
        _require_contract(market, filename)
        return market, note
    raise PlaygroundError(f"Unknown card {card}.")


def _require_contract(market: Market, filename: str) -> None:
    missing = [name for name in ("s0", "strike", "maturity", "r") if not hasattr(market, name)]
    if missing:
        joined = ", ".join(missing)
        raise PlaygroundError(
            f"{filename} has no {joined}. Set them on the market, as the Merton template does."
        )


def _premium(market: Market) -> float:
    try:
        value = float(np.asarray(market.premium()))
    except NotImplementedError as exc:
        raise PlaygroundError(str(exc)) from exc
    except Exception as exc:
        raise PlaygroundError(f"premium failed: {type(exc).__name__}: {exc}") from exc
    if not math.isfinite(value):
        raise PlaygroundError(f"premium is {value}.")
    return value


def _simulate(market: Market, m: int, n: int, seed: int) -> Paths:
    try:
        return market.simulate(m, n, seed)
    except NotImplementedError as exc:
        raise PlaygroundError(str(exc)) from exc
    except PlaygroundError:
        raise
    except Exception as exc:
        raise PlaygroundError(f"simulate failed: {type(exc).__name__}: {exc}") from exc


def _fit(spec, market, paths, cost, premium, on_epoch):
    if spec.hedger not in ("mlp", "gru"):
        return None, []
    if spec.hedger == "mlp":
        policy: nn.Module = MLPPolicy(
            float(market.strike), float(market.maturity), spec.width, spec.depth
        )
        _start_near_half(policy)
    else:
        policy = GRUPolicy(float(market.strike), float(market.maturity), spec.hidden)
    trained = fit_policy(
        policy, paths.spot, float(market.strike), float(market.r), cost, spec.lam,
        spec.epochs, spec.lr, spec.seed_train, premium, spec.batch_size, on_epoch,
    )
    return trained.policy, trained.history


def _delta(market, paths, sigma) -> np.ndarray:
    return delta_positions(
        paths.spot, float(market.strike), float(market.maturity), float(sigma), float(market.r)
    )


def _positions(spec, market, paths, policy, cost, sigma) -> np.ndarray:
    if spec.hedger == "delta":
        return _delta(market, paths, sigma)
    strike = float(market.strike)
    maturity = float(market.maturity)
    rate = float(market.r)
    if spec.hedger == "ww":
        epsilon = band_epsilon(cost)
        if epsilon is None:
            epsilon = 0.0

        def width(column, tau, epsilon=epsilon):
            return whalley_wilmott_half_width(
                column, tau, epsilon, spec.lam, strike, sigma, rate
            )

        return band_positions(paths.spot, paths.time, strike, sigma, rate, width)
    if spec.hedger == "fixed":
        return band_positions(paths.spot, paths.time, strike, sigma, rate, float(spec.half_width))
    if policy is None:
        raise PlaygroundError("The network hedge has no trained policy.")
    return rollout_numpy(policy, paths.spot, maturity)


def _pnl(market, paths, positions, cost, premium) -> np.ndarray:
    try:
        payoff = market.payoff(paths)
    except Exception as exc:
        raise PlaygroundError(f"payoff failed: {type(exc).__name__}: {exc}") from exc
    return terminal_pnl(spot_only(paths), positions, payoff, cost, premium)


def _score(pnl: np.ndarray, alpha: float, lam: float) -> dict:
    return {
        "mean": float(np.mean(pnl)),
        "mean_se": mean_se(pnl),
        "entropic": entropic(pnl, lam),
        "entropic_se": entropic_se(pnl, lam),
        "cvar": cvar(pnl, alpha),
        "m": int(np.asarray(pnl).size),
    }


def _band(spec, market, policy, cost, sigma):
    step = spec.n_steps // 2
    t = step * float(market.maturity) / spec.n_steps
    tau = float(market.maturity) - t
    spots = np.linspace(float(market.s0) * 0.85, float(market.s0) * 1.15, 81)
    delta = np.asarray(
        bs_call_delta(spots, float(market.strike), tau, sigma, float(market.r)), dtype=float
    )
    epsilon = band_epsilon(cost)
    if epsilon is None:
        half = np.zeros_like(delta)
        applies = False
    else:
        half = np.asarray(
            whalley_wilmott_half_width(
                spots, tau, epsilon, spec.lam, float(market.strike), sigma, float(market.r)
            ),
            dtype=float,
        )
        applies = True
    prevs = np.array([0.0, 0.5])
    if spec.hedger in ("mlp", "gru") and policy is not None:
        response = position_grid(policy, t, spots, prevs)
    elif spec.hedger == "ww":
        response = np.stack([np.clip(prev, delta - half, delta + half) for prev in prevs])
    elif spec.hedger == "fixed":
        width = float(spec.half_width)
        response = np.stack([np.clip(prev, delta - width, delta + width) for prev in prevs])
    else:
        response = np.stack([delta, delta])
    return spots, delta, half, prevs, response, t, applies


def _same_market(spec: PlaygroundSpec) -> bool:
    if spec.train_card != spec.test_card:
        return False
    if spec.train_card == "py":
        return spec.train_source == spec.test_source
    if spec.train_card == "csv":
        return spec.train_closes == spec.test_closes
    return True


def _card_name(spec: PlaygroundSpec, slot: str, market: Market) -> str:
    card = spec.train_card if slot == "train" else spec.test_card
    name = CARD_NAMES.get(card, card)
    if card == "py":
        return f"{name} ({type(market).__name__})"
    return name


def _market_notes(spec, train_file_note: str | None, test_file_note: str | None) -> list[str]:
    notes = [
        "The hedge trades the spot only. A variance swap, when the market has one, "
        "is not in the book."
    ]
    if train_file_note:
        notes.append(f"Training file: {train_file_note}")
    if test_file_note:
        notes.append(f"Test file: {test_file_note}")
    if spec.train_card == "csv" or spec.test_card == "csv":
        if spec.demean:
            notes.append(
                "CSV paths are a block bootstrap of demeaned log returns. "
                "Removing the mean takes out the historical drift."
            )
        else:
            notes.append(
                "CSV paths are a block bootstrap of log returns. "
                "The historical drift is left in, so the discounted spot need not be a martingale."
            )
    return notes


def _result_notes(spec, cost, sigma_note_train, sigma_note_test) -> list[str]:
    notes = [
        f"Training delta uses {sigma_note_train}.",
        f"Test delta uses {sigma_note_test}.",
    ]
    epsilon = band_epsilon(cost)
    if epsilon is None:
        notes.append(
            "The Whalley-Wilmott formula is for a proportional cost. "
            "With a quadratic cost the shaded band has width zero, and that hedge is delta."
        )
    elif isinstance(as_cost(cost), FixedPlusProportional):
        notes.append(
            "The shaded band uses the proportional rate. The fixed cash is in the P&L "
            "and is outside the 1997 formula."
        )
    else:
        notes.append(
            "The shaded band is the Black-Scholes Whalley-Wilmott formula at the "
            "volatility used for delta on the test market. It is the 1997 proportional-cost band."
        )
    if _same_market(spec):
        notes.append(
            "Train and test use the same market. "
            f"The test paths are a fresh draw with seed {spec.seed_test}."
        )
    else:
        notes.append(
            "Every P&L uses the training premium. The test paths are drawn from the other market."
        )
    if spec.hedger == "gru":
        notes.append(
            "The response curve is the recurrent network from an empty memory. "
            "Along the sample path the memory carries from one date to the next."
        )
    return notes
