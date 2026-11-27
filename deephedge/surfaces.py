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


def weights_path(market: str, cost: float) -> Path:
    """Local, gitignored path for a snapshot-market policy. Not session 00-03."""
    bp = int(round(float(cost) * 1e4))
    return WEIGHTS_DIR / f"{market}_c{bp:03d}bp.pt"


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
) -> TrainResult:
    """Train a ``HestonPolicy`` by minimising entropic risk, not session-01 CVaR."""
    torch.manual_seed(setup.seed)
    market_model = setup.market()
    paths = market_model.simulate(setup.m_train, setup.n_steps, setup.seed)
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
    destination = path or weights_path(market, setup.cost)
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
) -> list[dict[str, Any]]:
    """Score the network, the model hedge, a spot-only delta, and no hedge."""
    paths = market.simulate(n_paths, n_steps, seed)
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
            path = weights_path(market_name, cost)
            if market_name == "snapshot_bs":
                setup = black_scholes_setup(calibration, cost, **extra)
                result = train_black_scholes(setup, path)
            elif market_name == "snapshot_heston":
                setup = heston_setup(calibration, cost, fit="contract", **extra)
                result = train_heston_entropic(setup, path, market=market_name)
            else:
                raise ValueError(f"unknown market {market_name!r}")
            summary.append(
                {
                    "market": market_name,
                    "cost": float(cost),
                    "path": str(path),
                    "wall_time": result.wall_time,
                    "final_objective": result.history[-1] if result.history else None,
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
            path = weights_path(market_name, cost)
            if not path.exists():
                raise FileNotFoundError(f"missing weights {path}; train first")
            contract = calibration["contract"]
            n_steps = int(extra.get("n_steps", contract.get("n_steps", 30)))
            if market_name == "snapshot_bs":
                policy, _setup = load_policy(path)
                market = black_scholes_setup(calibration, cost, **extra).market()
            elif market_name == "snapshot_heston":
                policy, _setup = load_heston_policy(path)
                market = heston_setup(calibration, cost, fit="contract", **extra).market()
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
    parser.add_argument("command", choices=("train", "eval"))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--markets", default=",".join(MARKETS))
    parser.add_argument("--costs", default="0,0.005")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "surfaces_eval.json")
    parser.add_argument("--paths", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    args = parser.parse_args(argv)
    calibration = load_calibration(args.calibration)
    markets = tuple(name.strip() for name in args.markets.split(",") if name.strip())
    costs = _parse_costs(args.costs)
    if args.command == "train":
        summary = train_markets(calibration, markets=markets, costs=costs)
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
    )
    write_results(
        args.out,
        rows,
        calibration=str(args.calibration),
        note=(
            "P&L uses the simulator premium. Snapshot-derived numbers in this "
            "file stay local and are not committed."
        ),
    )
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
