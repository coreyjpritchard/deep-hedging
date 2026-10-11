"""Static figures for the README. Run `python -m deephedge.figures`."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deephedge.policy import held_positions, position_grid  # noqa: E402
from deephedge.train import DEFAULT, cached_policy  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"


def band_figure(path: Path = DOCS / "00_no_trade_band.png") -> Path:
    """Panel (b) of session 00 at zero cost and at the default cost, side by side."""
    s = DEFAULT
    k = s.n_steps // 2
    t = k * s.maturity / s.n_steps
    market = s.market()
    paths = market.simulate(s.m_eval, s.n_steps, seed=12345).spot
    show = np.random.default_rng(0).choice(s.m_eval, 3_000, replace=False)
    spots = np.linspace(88, 112, 241)
    shown = np.array([0.2, 0.5, 0.8])
    delta = market.delta(spots, s.maturity - t)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, cost in zip(axes, (0.0, s.cost), strict=True):
        policy, trained_for = cached_policy(replace(s, cost=cost))
        spot, held = held_positions(policy, paths, s.maturity, k)
        ax.scatter(spot[show], held[show], s=2, color="tab:red", alpha=0.25, lw=0,
                   label="position held at T/2, one dot per path")
        ax.plot(spots, delta, color="tab:blue", lw=2.5, label="Black-Scholes delta")
        for p, y, ls in zip(shown, position_grid(policy, t, spots, shown), [":", "--", "-."],
                            strict=True):
            ax.plot(spots, y, color="0.15", ls=ls, lw=1.4,
                    label=f"new position if it holds {p:.1f}")
        ax.set_title(f"cost {trained_for.cost * 1e4:.0f} bp")
        ax.set_xlabel("spot S")
        ax.set_xlim(88, 112)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("position in the underlying")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, markerscale=4)
    fig.suptitle("A neural hedger learns to hold a band around delta when trading costs money")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(band_figure())
