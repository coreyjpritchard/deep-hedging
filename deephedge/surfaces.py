"""Train and evaluate snapshot-calibrated deep hedgers.

A calibration JSON is a user-supplied training document from the
volsurface calibrator. Fitted parameters, premiums, weights and evaluation
tables derived from a historical option snapshot are written under local
export folders and are not committed. Session 00-03 weights under
``data/weights/`` are not touched.

The network minimises entropic risk at ``lam = 1``, as in session 00.
Evaluation reports mean P&L, standard deviation, CVaR at 50 percent and
95 percent, and entropic risk at ``lam = 1``, for the network, the model
hedge, a spot-only delta, and no hedge, at the same proportional cost.
P&L uses the simulator's own premium. The surface premium is stored
alongside so a constant shift can be applied later.

Snapshot Heston is simulated on a fine Euler grid (480 steps, as in
session 01) and then subsampled so the hedge still trades only on the
30 daily dates. The variance-swap mark is taken from that fine grid,
not recomputed on the coarse dates. A 30-step Euler scheme is too
coarse for a high vol-of-vol fit: the unhedged mean P&L must be
consistent with zero within two standard errors before the table is
trusted.

How to read the evaluation table: at zero cost, a sound hedge has mean
P&L near zero and a much smaller standard deviation than doing nothing.
CVaR at 95 percent and entropic risk at lambda = 1 are the cash that
must be added to make the book acceptable. A network that beats the
model hedge by a wide margin at zero cost is a warning, not a result;
check the unhedged bias first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deephedge.env import Environment, ProportionalCost
from deephedge.paths import Paths
from deephedge.policy import HestonPolicy, MLPPolicy, rollout_numpy, rollout_positions
from deephedge.risk import cvar, entropic, entropic_torch, mean_se
from deephedge.train import (
    HestonSetup,
    Setup,
    TrainResult,
    load_heston_policy,
    load_policy,
    save_heston_policy,
    save_policy,
    train_book,
    train_setup,
)

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "data" / "weights" / "surfaces"
RESULTS_DIR = ROOT / "data" / "exports" / "local"
EVAL_SEED = 12345
LAM = 1.0
COSTS = (0.0, 0.005)
MARKETS = ("snapshot_bs", "snapshot_heston")
# Session 01: simulate 480 Euler steps and keep every 16th date, so the
# hedge still trades on the same 30 daily dates.
HEDGE_STEPS = 30
SIM_STEPS = 480


def load_calibration(path: Path | str) -> dict[str, Any]:
    """Load a training document. The path is an input; nothing is committed."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "contract" not in payload or "black_scholes" not in payload:
        raise ValueError(f"{path} is not a training calibration document")
    return payload


def heston_params(calibration: dict[str, Any], fit: str = "contract") -> dict[str, float]:
    """Return the named Heston fit, defaulting to the contract-weighted entry."""
    fits = calibration.get("heston_fits") or {}
    if fit in fits:
        row = fits[fit]
    elif fit == "contract" and "heston" in calibration:
        row = calibration["heston"]
    else:
        raise KeyError(f"calibration has no Heston fit {fit!r}")
    return {key: float(row[key]) for key in ("v0", "kappa", "theta", "xi", "rho")}


def black_scholes_setup(calibration: dict[str, Any], cost: float, **overrides: Any) -> Setup:
    """Session-00-style setup with the calibration's at-the-money vol and contract."""
    contract = calibration["contract"]
    setup = Setup(
        s0=float(contract["s0"]),
        strike=float(contract["strike"]),
        maturity=float(contract["maturity"]),
        sigma=float(calibration["black_scholes"]["sigma"]),
        r=float(contract.get("r", 0.0)),
        mu=float(contract.get("r", 0.0)),
        cost=float(cost),
        n_steps=int(contract.get("n_steps", 30)),
        lam=LAM,
    )
    return replace(setup, **overrides) if overrides else setup


def heston_setup(
    calibration: dict[str, Any],
    cost: float,
    *,
    fit: str = "contract",
    **overrides: Any,
) -> HestonSetup:
    """Session-01-style setup with one of the calibration's Heston fits."""
    contract = calibration["contract"]
    params = heston_params(calibration, fit)
    setup = HestonSetup(
        s0=float(contract["s0"]),
        strike=float(contract["strike"]),
        maturity=float(contract["maturity"]),
        r=float(contract.get("r", 0.0)),
        mu=float(contract.get("r", 0.0)),
        cost=float(cost),
        n_steps=int(contract.get("n_steps", 30)),
        **params,
    )
    return replace(setup, **overrides) if overrides else setup


def weights_path(market: str, cost: float, *, sim_steps: int | None = None) -> Path:
    """Local, gitignored path for a snapshot-market policy. Not session 00-03."""
    bp = int(round(float(cost) * 1e4))
    if market == "snapshot_heston":
        n_sim = int(sim_steps if sim_steps is not None else SIM_STEPS)
        return WEIGHTS_DIR / f"{market}_c{bp:03d}bp_sim{n_sim}.pt"
    return WEIGHTS_DIR / f"{market}_c{bp:03d}bp.pt"


def subsample_hedge_dates(paths: Paths, n_hedge: int) -> Paths:
    """Keep ``n_hedge`` equally spaced dates, including inception and maturity.

    The instrument marks, including the variance-swap price, are taken from
    the fine grid rather than recomputed on the coarse dates.
    """
    n_sim = paths.n_steps
    if n_hedge < 1:
        raise ValueError("n_hedge must be at least 1")
    if n_sim % n_hedge != 0:
        raise ValueError(f"n_sim={n_sim} must be a multiple of n_hedge={n_hedge}")
    idx = np.arange(0, n_sim + 1, n_sim // n_hedge)
    if int(idx[-1]) != n_sim:
        raise ValueError("subsample must include the terminal date")

    def take(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim == 1:
            return array[idx]
        return np.take(array, idx, axis=1)

    return Paths(
        time=take(paths.time),
        instruments=take(paths.instruments),
        features=take(paths.features),
        discount=take(paths.discount),
        state={name: take(value) for name, value in paths.state.items()},
        instrument_names=paths.instrument_names,
        feature_names=paths.feature_names,
        source=paths.source,
    )


def simulate_hedge_dates(market, m: int, n_hedge: int, n_sim: int, seed: int) -> Paths:
    """Simulate on ``n_sim`` steps and, if needed, subsample to ``n_hedge`` dates."""
    if n_sim < n_hedge:
        raise ValueError(f"n_sim={n_sim} must be at least n_hedge={n_hedge}")
    paths = market.simulate(m, n_sim, seed)
    if n_sim == n_hedge:
        return paths
    return subsample_hedge_dates(paths, n_hedge)


def unhedged_premium_gap(
    market,
    *,
    n_paths: int,
    n_sim: int,
    seed: int,
) -> dict[str, float]:
    """Unhedged mean P&L (simulator premium minus payoff) and its standard error."""
    paths = market.simulate(n_paths, n_sim, seed)
    premium = float(market.premium())
    payoff = np.asarray(market.payoff(paths) * paths.discount_terminal(), dtype=float)
    pnl = premium - payoff
    mean = float(pnl.mean())
    se = mean_se(pnl)
    swap = paths.instruments[:, :, 1] if paths.n_instruments > 1 else None
    rec = {
        "mean": mean,
        "mean_se": se,
        "abs_mean_over_se": (abs(mean) / se) if se > 0.0 else float("inf"),
        "within_2se": bool(abs(mean) <= 2.0 * se),
        "premium": premium,
        "n_paths": float(n_paths),
        "n_sim": float(n_sim),
        "eval_seed": float(seed),
    }
    if swap is not None:
        disc = paths.discount
        if disc.ndim == 1:
            rec["swap_gap"] = float(np.mean(swap[:, -1] * disc[-1] - swap[:, 0] * disc[0]))
        else:
            rec["swap_gap"] = float(np.mean(swap[:, -1] * disc[:, -1] - swap[:, 0] * disc[:, 0]))
    return rec


def train_black_scholes(setup: Setup, path: Path | None = None) -> TrainResult:
    """Train an ``MLPPolicy`` by minimising entropic risk at ``setup.lam``."""
    result = train_setup(setup)
    destination = path or weights_path("snapshot_bs", setup.cost)
    save_policy(result.policy, setup, destination, result.wall_time)
    return result


def train_heston_entropic(
    setup: HestonSetup,
    path: Path | None = None,
    *,
    lam: float = LAM,
    market: str = "snapshot_heston",
    n_sim: int = SIM_STEPS,
) -> TrainResult:
    """Train a ``HestonPolicy`` by minimising entropic risk, not session-01 CVaR.

    Paths are simulated on ``n_sim`` Euler steps and subsampled to
    ``setup.n_steps`` hedge dates, as in the session 01 fine-grid check.
    """
    torch.manual_seed(setup.seed)
    market_model = setup.market()
    paths = simulate_hedge_dates(market_model, setup.m_train, setup.n_steps, n_sim, setup.seed)
    prices = torch.as_tensor(paths.discounted_instruments(), dtype=torch.float32)
    features = torch.as_tensor(paths.features, dtype=torch.float32)
    terminal = np.asarray(paths.discount_terminal(), dtype=np.float32)
    payoff = torch.as_tensor(market_model.payoff(paths), dtype=torch.float32) * torch.as_tensor(
        terminal
    )
    policy = HestonPolicy(
        setup.strike,
        setup.maturity,
        setup.theta,
        width=setup.width,
        depth=setup.depth,
        varswap_scale=setup.varswap_scale,
    )
    result = train_book(
        policy,
        prices,
        features,
        paths.time,
        payoff,
        lambda pnl, _lam=lam: entropic_torch(pnl, _lam),
        setup.cost,
        setup.epochs,
        setup.lr,
        setup.seed,
        premium=market_model.premium(),
        batch_size=setup.batch_size,
    )
    destination = path or weights_path(market, setup.cost, sim_steps=n_sim)
    save_heston_policy(result.policy, setup, destination, result.wall_time)
    return result


def summarise_pnl(pnl: np.ndarray, *, premium: float) -> dict[str, float]:
    """Mean, standard error, two CVaR levels, and entropic risk of a P&L sample."""
    x = np.asarray(pnl, dtype=float)
    cvar_50 = cvar(x, 0.5)
    cvar_95 = cvar(x, 0.95)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "mean_se": mean_se(x),
        "cvar_50": cvar_50,
        "cvar_95": cvar_95,
        "entropic_1": entropic(x, LAM),
        "premium": float(premium),
        "premium_plus_cvar_50": float(premium) + cvar_50,
        "premium_plus_cvar_95": float(premium) + cvar_95,
        "n_paths": int(x.size),
    }


def _zero_positions(paths) -> np.ndarray:
    instruments = paths.discounted_instruments()
    m, n_plus, d = instruments.shape
    return np.zeros((m, n_plus - 1, d))


def _spot_only(positions: np.ndarray) -> np.ndarray:
    out = np.array(positions, dtype=float, copy=True)
    if out.ndim == 2:
        return out
    out[..., 1:] = 0.0
    return out


def network_positions(policy, paths) -> np.ndarray:
    """Roll out a trained policy on ``paths``, as a numpy array."""
    if isinstance(policy, MLPPolicy):
        pos = rollout_numpy(policy, paths.spot, policy.maturity)
        return pos[:, :, None] if pos.ndim == 2 else pos
    d = paths.discounted_instruments().shape[-1]
    with torch.no_grad():
        pos = rollout_positions(policy, paths.features, paths.time, d)
    return np.asarray(pos.detach().cpu().numpy(), dtype=float)


def evaluate_market(
    market_name: str,
    market,
    policy,
    cost: float,
    *,
    n_paths: int,
    n_steps: int,
    seed: int,
    surface_premium: float | None = None,
    n_sim: int | None = None,
) -> list[dict[str, Any]]:
    """Score the network, the model hedge, a spot-only delta, and no hedge."""
    if n_sim is None or n_sim == n_steps:
        paths = market.simulate(n_paths, n_steps, seed)
    else:
        paths = simulate_hedge_dates(market, n_paths, n_steps, n_sim, seed)
    env = Environment(market, ProportionalCost(cost))
    premium = float(market.premium())
    hedges = {
        "network": network_positions(policy, paths),
        "model": np.asarray(market.benchmark_positions(paths), dtype=float),
        "none": _zero_positions(paths),
    }
    hedges["delta"] = _spot_only(hedges["model"])
    rows = []
    for hedge, positions in hedges.items():
        if market_name == "snapshot_bs" and hedge == "model":
            # One instrument: the model hedge is the Black-Scholes delta.
            continue
        stats = summarise_pnl(env.pnl(paths, positions, premium), premium=premium)
        row = {
            "market": market_name,
            "hedge": hedge,
            "cost": float(cost),
            "eval_seed": int(seed),
            "n_steps": int(n_steps),
            "n_sim": int(n_sim if n_sim is not None else n_steps),
            **stats,
        }
        if surface_premium is not None:
            row["surface_premium"] = float(surface_premium)
            row["premium_gap"] = premium - float(surface_premium)
        rows.append(row)
    return rows


def train_markets(
    calibration: dict[str, Any],
    *,
    markets: tuple[str, ...] = MARKETS,
    costs: tuple[float, ...] = COSTS,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Train each requested market and cost. Weights stay under ``WEIGHTS_DIR``."""
    extra = overrides or {}
    summary = []
    for market_name in markets:
        for cost in costs:
            if market_name == "snapshot_bs":
                path = weights_path(market_name, cost)
                setup_kwargs = {k: v for k, v in extra.items() if k != "n_sim"}
                setup = black_scholes_setup(calibration, cost, **setup_kwargs)
                result = train_black_scholes(setup, path)
            elif market_name == "snapshot_heston":
                n_sim = int(extra.get("n_sim", SIM_STEPS))
                path = weights_path(market_name, cost, sim_steps=n_sim)
                setup_kwargs = {k: v for k, v in extra.items() if k != "n_sim"}
                setup = heston_setup(calibration, cost, fit="contract", **setup_kwargs)
                result = train_heston_entropic(setup, path, market=market_name, n_sim=n_sim)
            else:
                raise ValueError(f"unknown market {market_name!r}")
            summary.append(
                {
                    "market": market_name,
                    "cost": float(cost),
                    "path": str(path),
                    "wall_time": result.wall_time,
                    "final_objective": result.history[-1] if result.history else None,
                    "n_sim": int(
                        extra.get(
                            "n_sim",
                            SIM_STEPS if market_name == "snapshot_heston" else 30,
                        )
                    ),
                }
            )
    return summary


def evaluate_markets(
    calibration: dict[str, Any],
    *,
    markets: tuple[str, ...] = MARKETS,
    costs: tuple[float, ...] = COSTS,
    n_paths: int = 20_000,
    seed: int = EVAL_SEED,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate cached snapshot-market policies. Results are not committed."""
    extra = overrides or {}
    surface_premium = float(calibration["surface"]["premium"])
    rows: list[dict[str, Any]] = []
    for market_name in markets:
        for cost in costs:
            setup_kwargs = {k: v for k, v in extra.items() if k != "n_sim"}
            contract = calibration["contract"]
            n_steps = int(setup_kwargs.get("n_steps", contract.get("n_steps", 30)))
            if market_name == "snapshot_bs":
                path = weights_path(market_name, cost)
                if not path.exists():
                    raise FileNotFoundError(f"missing weights {path}; train first")
                policy, _setup = load_policy(path)
                market = black_scholes_setup(calibration, cost, **setup_kwargs).market()
                n_sim = n_steps
            elif market_name == "snapshot_heston":
                n_sim = int(extra.get("n_sim", SIM_STEPS))
                path = weights_path(market_name, cost, sim_steps=n_sim)
                if not path.exists():
                    raise FileNotFoundError(f"missing weights {path}; train first")
                policy, _setup = load_heston_policy(path)
                market = heston_setup(calibration, cost, fit="contract", **setup_kwargs).market()
            else:
                raise ValueError(f"unknown market {market_name!r}")
            rows.extend(
                evaluate_market(
                    market_name,
                    market,
                    policy,
                    cost,
                    n_paths=n_paths,
                    n_steps=n_steps,
                    seed=seed,
                    surface_premium=surface_premium,
                    n_sim=n_sim,
                )
            )
    return rows


def write_results(path: Path | str, rows: list[dict[str, Any]], **meta: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "deephedge.surfaces_eval/v1", "rows": rows, **meta}
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_costs(text: str) -> tuple[float, ...]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part) if "." in part or "e" in part.lower() else int(part) / 1e4)
    if not values:
        raise ValueError("no costs given")
    return tuple(values)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval", "bias"))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--markets", default=",".join(MARKETS))
    parser.add_argument("--costs", default="0,0.005")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "surfaces_eval.json")
    parser.add_argument("--paths", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--sim-steps", type=int, default=SIM_STEPS)
    args = parser.parse_args(argv)
    calibration = load_calibration(args.calibration)
    markets = tuple(name.strip() for name in args.markets.split(",") if name.strip())
    costs = _parse_costs(args.costs)
    overrides = {"n_sim": int(args.sim_steps)}
    if args.command == "bias":
        market = heston_setup(calibration, 0.0, fit="contract").market()
        rec = unhedged_premium_gap(
            market, n_paths=args.paths, n_sim=int(args.sim_steps), seed=args.seed
        )
        print(
            f"n_sim={int(rec['n_sim'])} n={int(rec['n_paths'])} "
            f"unhedged mean={rec['mean']:+.6f} se={rec['mean_se']:.6f} "
            f"|mean|/se={rec['abs_mean_over_se']:.2f} within_2se={rec['within_2se']}"
        )
        return
    if args.command == "train":
        summary = train_markets(calibration, markets=markets, costs=costs, overrides=overrides)
        for row in summary:
            print(
                f"{row['market']} cost={row['cost']} "
                f"wall={row['wall_time']:.1f}s objective={row['final_objective']}"
            )
        return
    rows = evaluate_markets(
        calibration,
        markets=markets,
        costs=costs,
        n_paths=args.paths,
        seed=args.seed,
        overrides=overrides,
    )
    write_results(
        args.out,
        rows,
        calibration=str(args.calibration),
        n_sim=int(args.sim_steps),
        note=(
            "P&L uses the simulator premium. Snapshot Heston is simulated on "
            "a fine Euler grid and hedged on the daily dates. Snapshot-derived "
            "numbers in this file stay local and are not committed."
        ),
    )
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
