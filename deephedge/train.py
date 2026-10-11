"""Train a hedging policy by minimising a risk measure of its P&L, end to end.

Run `python -m deephedge.train` to regenerate the cached weights used by session 00.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from deephedge.env import BlackScholes, Environment, ProportionalCost
from deephedge.hedging import call_payoff, pnl_from_positions_torch
from deephedge.markets import to_torch
from deephedge.policy import HestonPolicy, MLPPolicy, rollout, rollout_positions
from deephedge.risk import cvar_torch, entropic_torch

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "data" / "weights"


@dataclass(frozen=True)
class Setup:
    """Market, contract and training settings for session 00.

    The network minimises the entropic risk with risk aversion `lam`; `alpha` is only the
    level at which CVaR is reported.
    """

    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 30 / 365
    sigma: float = 0.2
    r: float = 0.0
    mu: float = 0.0
    cost: float = 0.005
    n_steps: int = 30
    lam: float = 1.0
    alpha: float = 0.95
    m_train: int = 20_000
    m_eval: int = 20_000
    width: int = 32
    depth: int = 2
    epochs: int = 100
    batch_size: int = 2_000
    lr: float = 0.005
    seed: int = 0

    def market(self) -> BlackScholes:
        """The Black-Scholes market this setup trains against."""
        return BlackScholes(
            s0=self.s0,
            strike=self.strike,
            maturity=self.maturity,
            sigma=self.sigma,
            r=self.r,
            mu=self.mu,
        )

    def environment(self) -> Environment:
        """Market plus the proportional cost of this setup."""
        return Environment(self.market(), ProportionalCost(self.cost))

    @property
    def premium(self) -> float:
        return self.market().premium()

    def key(self) -> str:
        return f"c{round(self.cost * 1e4):03d}bp_n{self.n_steps:03d}_lam{self.lam:g}"


DEFAULT = Setup()


@dataclass
class TrainResult:
    policy: nn.Module
    history: list[float]
    wall_time: float


def train(
    policy: nn.Module,
    paths_tensor: torch.Tensor,
    payoff_fn: Callable,
    risk_fn: Callable[[torch.Tensor], torch.Tensor],
    cost: float,
    epochs: int,
    lr: float,
    seed: int,
    premium: float = 0.0,
    batch_size: int = 2_000,
) -> TrainResult:
    """Minimise risk_fn(P&L) over the policy weights with Adam on minibatches of paths.

    The P&L is simulated in torch exactly as `pnl_from_positions` does in numpy, so the
    gradient flows through every trade and every cost. `history` holds the full-sample
    objective after each epoch.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    T = policy.maturity
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    m = paths_tensor.shape[0]
    history = []
    start = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(m, generator=generator)
        for i in range(0, m, batch_size):
            batch = paths_tensor[order[i : i + batch_size]]
            positions = rollout(policy, batch, T)
            pnl = pnl_from_positions_torch(batch, positions, payoff_fn, cost, premium)
            loss = risk_fn(pnl)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        scheduler.step()
        history.append(objective(policy, paths_tensor, payoff_fn, risk_fn, cost, premium))
    return TrainResult(policy, history, time.perf_counter() - start)


def objective(
    policy: nn.Module,
    paths_tensor: torch.Tensor,
    payoff_fn: Callable,
    risk_fn: Callable,
    cost: float,
    premium: float = 0.0,
) -> float:
    """risk_fn of the policy's P&L on the given paths, without gradients."""
    with torch.no_grad():
        positions = rollout(policy, paths_tensor, policy.maturity)
        pnl = pnl_from_positions_torch(paths_tensor, positions, payoff_fn, cost, premium)
        return float(risk_fn(pnl))


def train_setup(setup: Setup = DEFAULT) -> TrainResult:
    """Train an `MLPPolicy` for a `Setup` on freshly simulated Black-Scholes paths.

    Paths, premium and payoff come from the setup's market. The network still
    sees the spot, which is feature column 0, through the same rollout as
    before.
    """
    market = setup.market()
    paths = market.simulate(setup.m_train, setup.n_steps, setup.seed)
    torch.manual_seed(setup.seed)
    policy = MLPPolicy(setup.strike, setup.maturity, setup.width, setup.depth)
    return train(
        policy,
        to_torch(paths.spot),
        call_payoff(setup.strike),
        lambda pnl: entropic_torch(pnl, setup.lam),
        setup.cost,
        setup.epochs,
        setup.lr,
        setup.seed,
        premium=market.premium(),
        batch_size=setup.batch_size,
    )


def save_policy(policy: MLPPolicy, setup: Setup, path: Path, wall_time: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    torch.save(
        {"state_dict": state, "config": policy.config(), "setup": asdict(setup),
         "wall_time": wall_time},
        path,
    )


def load_policy(path: Path) -> tuple[MLPPolicy, Setup]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    policy = MLPPolicy(**blob["config"])
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    return policy, Setup(**blob["setup"])


def cached_weights_path(setup: Setup) -> Path:
    if setup == DEFAULT:
        return WEIGHTS_DIR / "s00_default.pt"
    return WEIGHTS_DIR / f"s00_{setup.key()}.pt"


@dataclass(frozen=True)
class HestonSetup:
    """Market and training settings for the 2019 Heston experiment.

    The network minimises CVaR at `alpha` (the paper's average value at
    risk: the mean loss on the worst `1 - alpha` of paths). Defaults are
    section 5.2 of Buehler, Gonon, Teichmann and Wood (2019): zero rates,
    variance-swap hedge, thirty daily steps, no transaction cost.
    """

    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 30 / 365
    r: float = 0.0
    mu: float = 0.0
    kappa: float = 1.0
    theta: float = 0.04
    xi: float = 2.0
    rho: float = -0.7
    v0: float = 0.04
    cost: float = 0.0
    n_steps: int = 30
    alpha: float = 0.5
    m_train: int = 20_000
    m_eval: int = 20_000
    width: int = 64
    depth: int = 3
    varswap_scale: float = 100.0
    epochs: int = 100
    batch_size: int = 2_000
    lr: float = 0.005
    seed: int = 0

    def market(self):
        """The Heston market this setup trains against."""
        from deephedge.heston import Heston

        return Heston(
            s0=self.s0,
            strike=self.strike,
            maturity=self.maturity,
            r=self.r,
            mu=self.mu,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
            v0=self.v0,
        )

    def environment(self) -> Environment:
        return Environment(self.market(), ProportionalCost(self.cost))

    def key(self) -> str:
        return f"c{round(self.cost * 1e4):03d}bp_a{round(self.alpha * 100):03d}"


DEFAULT_HESTON = HestonSetup()


def train_book(
    policy: nn.Module,
    prices: torch.Tensor,
    features: torch.Tensor,
    times,
    payoff: torch.Tensor,
    risk_fn: Callable[[torch.Tensor], torch.Tensor],
    cost: float | tuple[float, ...],
    epochs: int,
    lr: float,
    seed: int,
    premium: float = 0.0,
    batch_size: int = 2_000,
) -> TrainResult:
    """Minimise a risk of the multi-instrument P&L. The Black-Scholes `train` is unchanged.

    `prices` are already discounted, shape `(m, n + 1, d)`. `payoff` is
    already multiplied by the maturity discount, shape `(m,)`. `features`
    has shape `(m, n + 1, f)`.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    m, _, n_instruments = prices.shape
    history = []
    start = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(m, generator=generator)
        for i in range(0, m, batch_size):
            idx = order[i : i + batch_size]
            positions = rollout_positions(policy, features[idx], times, n_instruments)
            pnl = pnl_from_positions_torch(
                prices[idx], positions, payoff[idx], cost, premium
            )
            loss = risk_fn(pnl)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        scheduler.step()
        history.append(
            _book_objective(policy, prices, features, times, payoff, risk_fn, cost, premium)
        )
    return TrainResult(policy, history, time.perf_counter() - start)


def _book_objective(policy, prices, features, times, payoff, risk_fn, cost, premium) -> float:
    with torch.no_grad():
        positions = rollout_positions(policy, features, times, prices.shape[-1])
        pnl = pnl_from_positions_torch(prices, positions, payoff, cost, premium)
        return float(risk_fn(pnl))


def train_heston(setup: HestonSetup = DEFAULT_HESTON) -> TrainResult:
    """Train a `HestonPolicy` to minimise CVaR at `setup.alpha`."""
    torch.manual_seed(setup.seed)
    market = setup.market()
    paths = market.simulate(setup.m_train, setup.n_steps, setup.seed)
    prices = torch.as_tensor(paths.discounted_instruments(), dtype=torch.float32)
    features = torch.as_tensor(paths.features, dtype=torch.float32)
    terminal = np.asarray(paths.discount_terminal(), dtype=np.float32)
    payoff = torch.as_tensor(market.payoff(paths), dtype=torch.float32) * torch.as_tensor(terminal)
    policy = HestonPolicy(
        setup.strike,
        setup.maturity,
        setup.theta,
        width=setup.width,
        depth=setup.depth,
        varswap_scale=setup.varswap_scale,
    )
    return train_book(
        policy,
        prices,
        features,
        paths.time,
        payoff,
        lambda pnl: cvar_torch(pnl, setup.alpha),
        setup.cost,
        setup.epochs,
        setup.lr,
        setup.seed,
        premium=market.premium(),
        batch_size=setup.batch_size,
    )


def save_heston_policy(
    policy: HestonPolicy, setup: HestonSetup, path: Path, wall_time: float = 0.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "config": policy.config(),
            "setup": asdict(setup),
            "wall_time": wall_time,
        },
        path,
    )


def load_heston_policy(path: Path) -> tuple[HestonPolicy, HestonSetup]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    policy = HestonPolicy(**blob["config"])
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    return policy, HestonSetup(**blob["setup"])


def cached_heston_weights_path(setup: HestonSetup) -> Path:
    if setup == DEFAULT_HESTON:
        return WEIGHTS_DIR / "s01_default.pt"
    return WEIGHTS_DIR / f"s01_{setup.key()}.pt"


def cached_heston_policy(setup: HestonSetup) -> tuple[HestonPolicy, HestonSetup]:
    """Weights trained for `setup` if cached, else the zero-cost defaults."""
    path = cached_heston_weights_path(setup)
    if not path.exists():
        path = cached_heston_weights_path(DEFAULT_HESTON)
    return load_heston_policy(path)


CACHED_COSTS = (0.0, 0.0025, 0.005, 0.01, 0.02)


def cached_policy(setup: Setup) -> tuple[MLPPolicy, Setup]:
    """Weights trained for `setup` if cached, else the defaults. Check the returned Setup."""
    path = cached_weights_path(setup)
    if not path.exists():
        path = cached_weights_path(DEFAULT)
    return load_policy(path)


def main() -> None:
    torch.set_num_threads(4)
    for cost in CACHED_COSTS:
        setup = replace(DEFAULT, cost=cost)
        result = train_setup(setup)
        path = cached_weights_path(setup)
        save_policy(result.policy, setup, path, result.wall_time)
        print(
            f"{setup.key()}: {result.wall_time:.1f} s, objective "
            f"{result.history[0]:.4f} -> {result.history[-1]:.4f}, saved {path.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
