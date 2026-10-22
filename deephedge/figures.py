"""Static figures and the numbers quoted in the README.

Run `python -m deephedge.figures`. It loads the cached weights in `data/weights/`,
scores every hedger on fresh paths (seed 12345), redraws the four PNGs in `docs/`
and writes the numbers to `docs/results.csv`. About two minutes on four CPU cores.
"""

from __future__ import annotations

import csv
import textwrap
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deephedge.band import score_hedgers  # noqa: E402
from deephedge.env import Environment  # noqa: E402
from deephedge.policy import (  # noqa: E402
    held_positions,
    position_grid,
    rollout_paths,
    rollout_positions,
)
from deephedge.risk import summarise  # noqa: E402
from deephedge.train import (  # noqa: E402
    COST_SESSION,
    DEFAULT,
    DEFAULT_HESTON,
    WW_COST_GRID,
    WW_FIXED_HALF_WIDTH,
    cached_heston_policy,
    cached_policy,
    cost_model_from_spec,
    cost_session_market,
    load_cost_session_policy,
    load_ww_comparison_policy,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"
SEED = 12345
CAPTION = dict(fontsize=8, color="#52514e", va="bottom")
BS_MARKET = (
    "Simulated Black-Scholes market: S0 = K = 100, sigma = 20%, r = 0, "
    "short 30-day at-the-money call"
)


def _caption(fig, text: str) -> float:
    """Wrapped caption at the foot of the figure. Returns the bottom of the plot area."""
    width = int(fig.get_figwidth() * 16)
    lines = [line for part in text.split("\n") for line in textwrap.wrap(part, width)]
    fig.text(0.01, 0.01, "\n".join(lines), **CAPTION)
    return 0.02 + len(lines) * 0.17 / fig.get_figheight()


def band_figure(path: Path = DOCS / "00_no_trade_band.png") -> Path:
    """Session 00: positions held at day 15, at zero cost and at 50 bp."""
    s = DEFAULT
    k = s.n_steps // 2
    t = k * s.maturity / s.n_steps
    market = s.market()
    paths = market.simulate(s.m_eval, s.n_steps, seed=SEED).spot
    show = np.random.default_rng(0).choice(s.m_eval, 3_000, replace=False)
    spots = np.linspace(88, 112, 241)
    shown = np.array([0.2, 0.5, 0.8])
    delta = market.delta(spots, s.maturity - t)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    for ax, cost in zip(axes, (0.0, s.cost), strict=True):
        policy, trained_for = cached_policy(replace(s, cost=cost))
        spot, held = held_positions(policy, paths, s.maturity, k)
        ax.scatter(spot[show], held[show], s=2, color="tab:red", alpha=0.25, lw=0,
                   label="network's holding after the day-15 rehedge, one dot per path")
        ax.plot(spots, delta, color="tab:blue", lw=2.5, label="Black-Scholes delta")
        for p, y, ls in zip(shown, position_grid(policy, t, spots, shown), [":", "--", "-."],
                            strict=True):
            ax.plot(spots, np.clip(y, -0.05, 1.05), color="0.15", ls=ls, lw=1.4,
                    label=f"network's new position if it currently holds {p:.1f}")
        bp = trained_for.cost * 1e4
        ax.set_title("no trading cost" if cost == 0 else f"{bp:.0f} bp proportional cost")
        ax.set_xlabel("spot S on day 15")
        ax.set_xlim(88, 112)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("position in the underlying (shares)")
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False, markerscale=4)
    fig.suptitle("With trading costs the network stops tracking delta and holds a band around it")
    bottom = _caption(fig, f"{BS_MARKET}, 30 daily rehedges. Network trained on seed 0 to minimise "
             f"entropic risk (lambda = 1); dots are 3,000 of {s.m_eval:,} fresh paths, "
             f"seed {SEED}.")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def session00_numbers() -> list[dict]:
    """Session 00 at 50 bp, and the 100 bp network across rehedging frequencies."""
    rows = []
    s = DEFAULT
    env = s.environment()
    sim = env.market.simulate(s.m_eval, s.n_steps, seed=SEED)
    policy, _ = cached_policy(s)
    for name, pos in (
        ("delta", env.market.benchmark_positions(sim)),
        ("network", rollout_paths(policy, sim)),
    ):
        st = summarise(env.pnl(sim, pos), s.alpha, s.lam)
        rows.append({"session": "00", "case": "50 bp, 30 steps", "hedger": name,
                     "mean": st["mean"], "std": st["std"], "entropic": st["entropic"]})
    rows.append({"session": "00", "case": "premium", "hedger": "", "mean": s.premium})
    s100 = replace(s, cost=0.01)
    policy100, _ = cached_policy(s100)
    for n in (10, 30, 60, 120):
        sn = replace(s100, n_steps=n)
        env = sn.environment()
        sim = env.market.simulate(sn.m_eval, n, seed=SEED)
        for name, pos in (
            ("delta", env.market.benchmark_positions(sim)),
            ("network trained at 30 steps", rollout_paths(policy100, sim)),
        ):
            st = summarise(env.pnl(sim, pos), sn.alpha, sn.lam)
            rows.append({"session": "00", "case": f"100 bp, {n} steps", "hedger": name,
                         "mean": st["mean"], "std": st["std"], "entropic": st["entropic"]})
    return rows


def heston_figure(path: Path = DOCS / "01_heston_pnl.png") -> tuple[Path, list[dict]]:
    """Session 01: P&L of the model hedge, the network and spot-only delta, zero cost."""
    setup = DEFAULT_HESTON
    market = setup.market()
    sim = market.simulate(setup.m_eval, setup.n_steps, seed=SEED)
    bench = market.benchmark_positions(sim)
    policy, _ = cached_heston_policy(setup)
    with torch.no_grad():
        net = rollout_positions(policy, sim.features, sim.time, 2).double().numpy()
    spot_only = bench.copy()
    spot_only[:, :, 1] = 0.0
    env = Environment(market, setup.cost)
    pnl = {
        "model hedge (delta + variance swap)": env.pnl(sim, bench),
        "network (spot + variance swap)": env.pnl(sim, net),
        "delta in the spot only": env.pnl(sim, spot_only),
    }
    stats = {k: summarise(v, setup.alpha, 1.0) for k, v in pnl.items()}
    premium = market.premium()
    rows = [{"session": "01", "case": "Heston, 0 bp", "hedger": k, "mean": st["mean"],
             "std": st["std"], "cvar": st["cvar"], "premium_plus_cvar": premium + st["cvar"]}
            for k, st in stats.items()]
    rows.append({"session": "01", "case": "premium", "hedger": "", "mean": premium})
    colours = ["tab:blue", "tab:red", "0.5"]
    edges = np.linspace(-4, 2, 121)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for (name, x), colour in zip(pnl.items(), colours, strict=True):
        st = stats[name]
        ax.hist(np.clip(x, edges[0], edges[-1]), bins=edges, density=True, histtype="stepfilled",
                alpha=0.4, color=colour, edgecolor=colour,
                label=f"{name}: std {st['std']:.2f}, CVaR {st['cvar']:.2f}")
    ax.set_xlabel("terminal P&L of the short call (premium - payoff + trading gains)")
    ax.set_ylabel("density")
    ax.set_xlim(edges[0], edges[-1])
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)
    model, network = (stats[k]["cvar"] for k in list(pnl)[:2])
    ax.set_title(
        f"In the Buehler et al. Heston market the network matches the model hedge\n"
        f"premium + CVaR {premium + network:.2f} against {premium + model:.2f}; "
        f"without the variance swap the spread is about four times wider",
        loc="left", fontsize=10.5,
    )
    bottom = _caption(fig, f"Simulated Heston: kappa = {setup.kappa:g}, theta = {setup.theta:g}, "
             f"xi = {setup.xi:g}, rho = {setup.rho:g}, v0 = {setup.v0:g}; 30-day ATM call, "
             f"premium {premium:.3f}, 30 rehedges, no cost. CVaR at {setup.alpha:.0%} "
             f"(mean of the worst half of losses). {setup.m_eval:,} fresh paths, seed {SEED}; "
             f"tails beyond [-4, 2] are piled into the end bins.")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, rows


def cost_models_figure(path: Path = DOCS / "02_cost_models.png") -> tuple[Path, list[dict]]:
    """Session 02: delta against the network under three cost models."""
    market = cost_session_market("bs")
    sim = market.simulate(20_000, 30, seed=SEED)
    bench = market.benchmark_positions(sim)
    keys = ["bs-prop", "bs-quad", "bs-fixed"]
    rows = []
    for key in keys:
        spec = COST_SESSION[key]
        policy, _ = load_cost_session_policy(key)
        env = Environment(market, cost_model_from_spec(spec))
        for name, pos in (("delta", bench), ("network", rollout_paths(policy, sim))):
            st = summarise(env.pnl(sim, pos), spec["alpha"], spec["lam"])
            rows.append({"session": "02", "case": spec["label"], "hedger": name,
                         "mean": st["mean"], "std": st["std"], "entropic": st["entropic"]})
    labels = [COST_SESSION[k]["label"] for k in keys]
    d = [r["entropic"] for r in rows if r["hedger"] == "delta"]
    n = [r["entropic"] for r in rows if r["hedger"] == "network"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bd = ax.bar(x - 0.2, d, 0.4, color="tab:blue", label="Black-Scholes delta")
    bn = ax.bar(x + 0.2, n, 0.4, color="tab:red", label="network trained for that cost")
    ax.bar_label(bd, fmt="%.2f", fontsize=8.5)
    ax.bar_label(bn, fmt="%.2f", fontsize=8.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("entropic risk at lambda = 1 (lower is better)")
    ax.set_ylim(0, max(d + n) * 1.18)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "The network beats delta under proportional and quadratic costs, not under a fixed "
        "charge",
        loc="left", fontsize=10.5,
    )
    bottom = _caption(fig, f"{BS_MARKET}, 30 daily rehedges, 20,000 fresh paths, seed {SEED}. The "
             "fixed-charge network was trained on a smooth surrogate of the charge and is "
             "scored on the exact charge, which it pays on almost every date.")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, rows


def risk_vs_cost_figure(path: Path = DOCS / "03_risk_vs_cost.png") -> tuple[Path, list[dict]]:
    """Session 03: entropic risk against cost for delta, the two bands and the network."""
    market = DEFAULT.market()
    names = {
        "delta": ("Black-Scholes delta", "tab:blue", "o"),
        "ww": ("Whalley-Wilmott asymptotic band", "tab:green", "s"),
        "fixed": ("constant-width band, width tuned on seed 0", "tab:purple", "D"),
        "mlp": ("network", "tab:red", "^"),
    }
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, n_steps in zip(axes, (120, 30), strict=True):
        costs = WW_COST_GRID[n_steps]
        paths = market.simulate(DEFAULT.m_eval, n_steps, seed=SEED)
        table = {k: [] for k in names}
        for cost in costs:
            policy, _ = load_ww_comparison_policy(n_steps, cost, "mlp")
            scores, _ = score_hedgers(market, paths, cost, DEFAULT.lam, {"mlp": policy},
                                      WW_FIXED_HALF_WIDTH[(n_steps, cost)])
            for k in names:
                table[k].append(scores[k]["entropic"])
                rows.append({"session": "03", "case": f"{n_steps} steps, {cost * 1e4:g} bp",
                             "hedger": k, "mean": scores[k]["mean"],
                             "entropic": scores[k]["entropic"],
                             "entropic_se": scores[k]["entropic_se"]})
        bp = np.array(costs) * 1e4
        for k, (label, colour, marker) in names.items():
            ax.plot(bp, table[k], marker=marker, color=colour, label=label)
        ax.set_xscale("log")
        ax.set_xticks(bp, [f"{b:g}" for b in bp])
        ax.minorticks_off()
        ax.set_xlabel("proportional cost (bp of the spot traded)")
        ax.set_title(f"{n_steps} rehedges over 30 days", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("entropic risk at lambda = 1 (lower is better)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Every band beats delta once trading costs money; at 200 bp the network beats "
                 "both bands", fontsize=11.5)
    bottom = _caption(fig, f"{BS_MARKET}. Each point is one batch of {DEFAULT.m_eval:,} fresh "
             f"paths, seed {SEED}, shared by every hedger; entropic standard errors are in "
             "docs/results.csv. Networks trained on seed 0 at each cost (30 steps: the "
             "session 00 networks).")
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, rows


def write_results(rows: list[dict], path: Path = DOCS / "results.csv") -> Path:
    fields = ["session", "case", "hedger", "mean", "std", "cvar", "premium_plus_cvar",
              "entropic", "entropic_se"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})
    return path


def main() -> None:
    torch.set_num_threads(4)
    print(band_figure())
    rows = session00_numbers()
    for make in (heston_figure, cost_models_figure, risk_vs_cost_figure):
        out, more = make()
        print(out)
        rows += more
    print(write_results(rows))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
