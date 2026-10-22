"""Compact JSON for a future hedge widget. No web code lives here.

Schema `deephedge.hedge/v1`. One document is one market, one cost, and
one or more books scored on the same paths. Arrays are nested lists,
rounded to `decimals` places (default 6). A widget can draw the paths
and the positions without importing this package.

The schema is provisional, not frozen. A later run-store layout may
replace it, so a reader should check the `schema` key.

Required keys:

* `schema`: `deephedge.hedge/v1`
* `market`: short name, for example `BlackScholes`
* `cost`: the cost model's label
* `time`: length `n + 1`, including `t = 0` and maturity
* `instrument_names`: length `d`
* `instruments`: shape `(m, n + 1, d)`, the prices that were traded
* `positions`: shape `(m, n, d)`, the primary book. Flat at maturity,
  so there is no position column there
* `pnl`: length `m`, that book's terminal P&L on these paths
* `stats`: at least `mean`, `std` and `n` for the primary book. `n` may
  be larger than `m` when the exported paths are a subset and the
  statistics were computed on the full sample

Optional keys: `seed`, `note`, `books`. Each entry of `books` has
`positions`, `pnl` and `stats` for another book on the same paths.
`stats` on a book uses the same fields as the top-level `stats`.

`max_paths` keeps the file small by slicing every path array on the
first axis. Statistics are the caller's: pass numbers already computed
on the sample you care about. They are not recomputed from the slice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "deephedge.hedge/v1"

__all__ = ["SCHEMA", "export_hedge", "load_hedge", "write_bs_example"]


def _round_array(values: np.ndarray, decimals: int) -> list:
    array = np.asarray(values, dtype=float)
    return np.round(array, decimals=decimals).tolist()


def _stats(stats: dict, decimals: int) -> dict:
    out = {}
    for key, value in stats.items():
        if key == "n":
            out[key] = int(value)
        else:
            out[key] = round(float(value), decimals)
    if "mean" not in out or "std" not in out or "n" not in out:
        raise ValueError("stats must include mean, std and n")
    return out


def _check_shapes(time, names, instruments, positions, pnl) -> None:
    time = np.asarray(time)
    instruments = np.asarray(instruments, dtype=float)
    positions = np.asarray(positions, dtype=float)
    pnl = np.asarray(pnl, dtype=float)
    if instruments.ndim != 3:
        raise ValueError(f"instruments must have shape (m, n + 1, d), got {instruments.shape}")
    m, n_plus, d = instruments.shape
    if time.shape != (n_plus,):
        raise ValueError(f"time has shape {time.shape}, expected {(n_plus,)}")
    if len(names) != d:
        raise ValueError(f"{len(names)} instrument names for d = {d}")
    if positions.shape != (m, n_plus - 1, d):
        raise ValueError(f"positions have shape {positions.shape}, expected {(m, n_plus - 1, d)}")
    if pnl.shape != (m,):
        raise ValueError(f"pnl has shape {pnl.shape}, expected {(m,)}")


def export_hedge(
    path: Path | str,
    *,
    market: str,
    cost: str,
    time: np.ndarray,
    instrument_names: tuple[str, ...] | list[str],
    instruments: np.ndarray,
    positions: np.ndarray,
    pnl: np.ndarray,
    stats: dict,
    books: dict | None = None,
    seed: int | None = None,
    note: str = "",
    max_paths: int | None = None,
    decimals: int = 6,
) -> dict:
    """Write a `deephedge.hedge/v1` document and return it.

    `books` maps a name to a dict with `positions`, `pnl` and `stats`.
    The primary book is the top-level `positions`, `pnl` and `stats`.
    """
    names = list(instrument_names)
    instruments = np.asarray(instruments, dtype=float)
    positions = np.asarray(positions, dtype=float)
    pnl = np.asarray(pnl, dtype=float)
    if max_paths is not None:
        instruments = instruments[:max_paths]
        positions = positions[:max_paths]
        pnl = pnl[:max_paths]
    _check_shapes(time, names, instruments, positions, pnl)
    document = {
        "schema": SCHEMA,
        "market": market,
        "cost": cost,
        "time": _round_array(time, decimals),
        "instrument_names": names,
        "instruments": _round_array(instruments, decimals),
        "positions": _round_array(positions, decimals),
        "pnl": _round_array(pnl, decimals),
        "stats": _stats(stats, decimals),
    }
    if seed is not None:
        document["seed"] = int(seed)
    if note:
        document["note"] = note
    if books:
        encoded = {}
        for name, book in books.items():
            book_pos = np.asarray(book["positions"], dtype=float)
            book_pnl = np.asarray(book["pnl"], dtype=float)
            if max_paths is not None:
                book_pos = book_pos[:max_paths]
                book_pnl = book_pnl[:max_paths]
            _check_shapes(time, names, instruments, book_pos, book_pnl)
            encoded[name] = {
                "positions": _round_array(book_pos, decimals),
                "pnl": _round_array(book_pnl, decimals),
                "stats": _stats(book["stats"], decimals),
            }
        document["books"] = encoded
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    return document


def load_hedge(path: Path | str) -> dict:
    """Read a document written by `export_hedge`."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ValueError(f"schema is {document.get('schema')!r}, expected {SCHEMA!r}")
    return document


def write_bs_example(
    path: Path | str,
    *,
    m_paths: int = 80,
    m_stats: int = 20_000,
    seed: int = 12345,
    cost: float = 0.005,
) -> dict:
    """A 30-day Black-Scholes call at `cost`, delta hedge as the primary book.

    The file stores `m_paths` paths. `stats.n` is `m_stats`, the sample
    the mean and standard deviation were computed on. The network book
    uses the cached session-00 weights trained at this proportional cost
    when that file is present, and is omitted otherwise.
    """
    from deephedge.env import BlackScholes, Environment, ProportionalCost
    from deephedge.policy import rollout_paths
    from deephedge.risk import summarise
    from deephedge.train import DEFAULT, Setup, cached_policy

    market = BlackScholes(
        s0=DEFAULT.s0,
        strike=DEFAULT.strike,
        maturity=DEFAULT.maturity,
        sigma=DEFAULT.sigma,
        r=DEFAULT.r,
        mu=DEFAULT.mu,
    )
    model = ProportionalCost(cost)
    env = Environment(market, model)
    full = market.simulate(m_stats, DEFAULT.n_steps, seed)
    primary = market.benchmark_positions(full)[:, :, 0]
    delta_pnl = env.pnl(full, primary)
    alpha = 0.95
    delta_stats = summarise(delta_pnl, alpha, DEFAULT.lam)
    delta_stats["n"] = m_stats
    books = {}
    flat = np.zeros((m_stats, DEFAULT.n_steps))
    flat_pnl = env.pnl(full, flat)
    flat_stats = summarise(flat_pnl, alpha, DEFAULT.lam)
    flat_stats["n"] = m_stats
    books["no hedge"] = {"positions": flat[:, :, None], "pnl": flat_pnl, "stats": flat_stats}
    setup = Setup(
        s0=market.s0,
        strike=market.strike,
        maturity=market.maturity,
        sigma=market.sigma,
        r=market.r,
        mu=market.mu,
        cost=cost,
        n_steps=DEFAULT.n_steps,
        lam=DEFAULT.lam,
    )
    try:
        policy, trained = cached_policy(setup)
    except FileNotFoundError:
        policy = None
        trained = None
    if policy is not None and trained.cost == cost and trained.n_steps == DEFAULT.n_steps:
        net_pos = rollout_paths(policy, full)
        net_pnl = env.pnl(full, net_pos)
        net_stats = summarise(net_pnl, alpha, DEFAULT.lam)
        net_stats["n"] = m_stats
        books["network"] = {
            "positions": net_pos[:, :, None],
            "pnl": net_pnl,
            "stats": net_stats,
        }
    return export_hedge(
        path,
        market="BlackScholes",
        cost=model.label(),
        time=full.time,
        instrument_names=market.instrument_names,
        instruments=full.instruments,
        positions=primary[:, :, None],
        pnl=delta_pnl,
        stats=delta_stats,
        books=books,
        seed=seed,
        note=(
            "Thirty-day at-the-money Black-Scholes call, S0 = K = 100, sigma = 0.2, "
            "r = mu = 0. Primary book is the analytic delta. Statistics use the full "
            "sample; instruments, positions and pnl are the first paths only."
        ),
        max_paths=m_paths,
    )
