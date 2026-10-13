"""One path, stepped date by date, for the live hedging view.

The book trades the spot only. A variance swap, when the market has one,
is not held. On Black-Scholes the delta is the analytic call delta. On
Heston it is the cosine-expansion delta at today's variance, the same
greek as `Heston.greeks`. The band is a fixed half-width around that
delta: the holding is left unchanged while it sits inside, and moved to
the nearer edge when delta leaves.

Cash is time-0 money. It starts as the premium, pays for shares and for
proportional costs, and at maturity pays the discounted call. The hedge
P&L at a trading date is cash plus the shares minus the model price of
the call. At maturity the shares are sold and the call is paid, so the
P&L equals `terminal_pnl` for the same positions, premium and cost.

`ledger` prices any holding of the spot. A later lesson can pass a
network's positions through it and draw them on the same path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from deephedge.costs import ProportionalCost
from deephedge.env import (
    BlackScholes,
    Market,
    Paths,
    bs_call_delta,
    bs_call_price,
    terminal_pnl,
)
from deephedge.heston import Heston
from deephedge.risk import entropic, entropic_se, mean_se

__all__ = [
    "BatchScore",
    "Book",
    "LivePath",
    "clip_positions",
    "describe",
    "describe_batch",
    "draw_batch",
    "draw_live",
    "ledger",
    "market_by_name",
    "score_books",
    "simulate_live",
    "terminal_of",
]

_TRADE = 1e-10
_EDGE = 1e-8

_BAND = "#c2410c"
_DELTA = "#4b5563"
_MARK = "#2563eb"
_CASH = "#0f766e"
_SPOT = "#111827"


@dataclass(frozen=True)
class Book:
    """One hedger's account along a single path. Each array has length `n + 1`.

    Index `k < n` is the state after the trade at date `k`, before the
    next move. Index `n` is the state after liquidation and payment of
    the call. `position[n]` is 0. `cost` is cumulative.
    """

    position: np.ndarray
    trade: np.ndarray
    cost_step: np.ndarray
    cost: np.ndarray
    cash: np.ndarray
    gains: np.ndarray
    pnl: np.ndarray
    rebalanced: np.ndarray


@dataclass(frozen=True)
class LivePath:
    """A simulated path and the two books walked along it.

    `delta` and `price` include maturity. `price` is the undiscounted
    model price, and `price[-1]` is the call payoff. `variance` is None
    on Black-Scholes.
    """

    market_name: str
    seed: int
    n: int
    half_width: float
    cost: float
    strike: float
    premium: float
    payoff: float
    payoff_disc: float
    delta_note: str
    time: np.ndarray
    spot: np.ndarray
    discount: np.ndarray
    variance: np.ndarray | None
    delta: np.ndarray
    price: np.ndarray
    band: Book
    delta_hedge: Book

    @property
    def token(self) -> tuple:
        """Identity of the draw, so a batch histogram can refuse a stale path."""
        return (self.market_name, self.n, self.seed, self.half_width, self.cost)

    @property
    def spot_disc(self) -> np.ndarray:
        """Spot in time-0 money, shape `(n + 1,)`."""
        return self.spot * self.discount


@dataclass(frozen=True)
class BatchScore:
    """Terminal P&L of the same two hedgers on a fresh sample."""

    token: tuple
    seed: int
    m: int
    lam: float
    pnl_band: np.ndarray
    pnl_delta: np.ndarray
    entropic_band: float
    entropic_delta: float
    entropic_band_se: float
    entropic_delta_se: float
    mean_band: float
    mean_delta: float
    mean_band_se: float
    mean_delta_se: float


def market_by_name(name: str) -> Market:
    """The two markets the live view can draw.

    Both are the short at-the-money call used in the sessions: spot 100,
    maturity 30/365, rates zero. Heston is the 2019 parameter set.
    """
    if name == "Black-Scholes":
        return BlackScholes(s0=100.0, strike=100.0, maturity=30 / 365, sigma=0.2, r=0.0, mu=0.0)
    if name == "Heston":
        return Heston()
    raise ValueError(f"market must be Black-Scholes or Heston, got {name!r}")


def clip_positions(delta: np.ndarray, half_width: float) -> np.ndarray:
    """Holdings from clipping to a band around delta.

    `delta` has shape `(n,)` or `(m, n)` and is the delta at each trading
    date, not at maturity. The book starts flat. The holding is left
    unchanged when it already lies between `delta - half_width` and
    `delta + half_width`. Returns the same shape as `delta`.
    """
    if half_width < 0.0:
        raise ValueError("half_width must be non-negative")
    raw = np.asarray(delta, dtype=float)
    single = raw.ndim == 1
    if single:
        raw = raw[None, :]
    if raw.ndim != 2:
        raise ValueError(f"delta must have shape (n,) or (m, n), got {delta.shape}")
    m, n = raw.shape
    held = np.zeros(m)
    out = np.empty((m, n))
    width = float(half_width)
    for k in range(n):
        column = raw[:, k]
        held = np.clip(held, column - width, column + width)
        out[:, k] = held
    return out[0] if single else out


def ledger(
    spot_disc: np.ndarray,
    discount: np.ndarray,
    price: np.ndarray,
    position: np.ndarray,
    premium: float,
    payoff: float,
    cost: float,
) -> Book:
    """Walk one spot holding to maturity.

    `position` has shape `(n,)` and is held over each interval. `spot_disc`,
    `discount` and `price` have shape `(n + 1,)`. `price` is undiscounted.
    `cost` is a proportional rate on the discounted spot. The last frame
    sells the shares and pays `payoff`.
    """
    held = np.asarray(position, dtype=float)
    spot_disc = np.asarray(spot_disc, dtype=float)
    discount = np.asarray(discount, dtype=float)
    price = np.asarray(price, dtype=float)
    n = held.shape[0]
    if spot_disc.shape != (n + 1,) or discount.shape != (n + 1,) or price.shape != (n + 1,):
        raise ValueError("spot, discount and price must have length n + 1")
    if cost < 0.0:
        raise ValueError("cost must be non-negative")

    out_position = np.zeros(n + 1)
    trade = np.zeros(n + 1)
    cost_step = np.zeros(n + 1)
    cost_cum = np.zeros(n + 1)
    cash = np.zeros(n + 1)
    gains = np.zeros(n + 1)
    pnl = np.zeros(n + 1)
    rebalanced = np.zeros(n + 1, dtype=bool)

    cash_running = float(premium)
    gain_running = 0.0
    cost_running = 0.0
    prev = 0.0
    rate = float(cost)
    for k in range(n):
        if k > 0:
            gain_running += prev * (spot_disc[k] - spot_disc[k - 1])
        change = float(held[k] - prev)
        charge = rate * abs(change) * float(spot_disc[k])
        cash_running = cash_running - change * float(spot_disc[k]) - charge
        cost_running += charge
        out_position[k] = held[k]
        trade[k] = change
        cost_step[k] = charge
        cost_cum[k] = cost_running
        cash[k] = cash_running
        gains[k] = gain_running
        pnl[k] = cash_running + held[k] * float(spot_disc[k]) - float(discount[k]) * float(price[k])
        rebalanced[k] = abs(change) > _TRADE
        prev = float(held[k])

    gain_running += prev * (float(spot_disc[n]) - float(spot_disc[n - 1]))
    change = -prev
    charge = rate * abs(change) * float(spot_disc[n])
    cash_running = cash_running - change * float(spot_disc[n]) - charge
    payoff_disc = float(discount[n]) * float(payoff)
    cash_running -= payoff_disc
    cost_running += charge
    out_position[n] = 0.0
    trade[n] = change
    cost_step[n] = charge
    cost_cum[n] = cost_running
    cash[n] = cash_running
    gains[n] = gain_running
    pnl[n] = cash_running
    rebalanced[n] = abs(change) > _TRADE
    return Book(
        position=out_position,
        trade=trade,
        cost_step=cost_step,
        cost=cost_cum,
        cash=cash,
        gains=gains,
        pnl=pnl,
        rebalanced=rebalanced,
    )


def _greeks(market: Market, paths: Paths) -> tuple[np.ndarray, np.ndarray]:
    """Undiscounted price and delta, both shape `(m, n + 1)`, including maturity."""
    spot = np.asarray(paths.spot, dtype=float)
    tau = float(market.maturity) - np.asarray(paths.time, dtype=float)
    if isinstance(market, BlackScholes):
        price = bs_call_price(spot, market.strike, tau, market.sigma, market.r)
        delta = bs_call_delta(spot, market.strike, tau, market.sigma, market.r)
        return np.asarray(price, dtype=float), np.asarray(delta, dtype=float)
    if isinstance(market, Heston):
        variance = np.asarray(paths.features[:, :, 1], dtype=float)
        price, delta, _vega = market.greeks(spot, variance, tau)
        return np.asarray(price, dtype=float), np.asarray(delta, dtype=float)
    raise TypeError(f"live view supports BlackScholes and Heston, got {type(market).__name__}")


def _note(market: Market) -> str:
    if isinstance(market, BlackScholes):
        return (
            "Delta is the Black-Scholes call delta at volatility "
            f"{market.sigma:.0%}."
        )
    if isinstance(market, Heston):
        return (
            "Delta is the Heston call delta from the cosine expansion, "
            "using today's variance. The hedge does not trade the variance swap."
        )
    raise TypeError(f"live view supports BlackScholes and Heston, got {type(market).__name__}")


def _spot_paths(paths: Paths) -> Paths:
    spot = np.asarray(paths.spot, dtype=float)
    return Paths(
        time=np.asarray(paths.time, dtype=float),
        instruments=spot[:, :, None],
        features=spot[:, :, None],
        discount=np.asarray(paths.discount, dtype=float),
    )


def simulate_live(market: Market, n: int, seed: int, half_width: float, cost: float) -> LivePath:
    """Draw one path and walk the band and delta hedging along it."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if half_width < 0.0 or cost < 0.0:
        raise ValueError("half_width and cost must be non-negative")
    paths = market.simulate(1, n, int(seed))
    price, delta = _greeks(market, paths)
    held = clip_positions(delta[0, :-1], half_width)
    delta_held = clip_positions(delta[0, :-1], 0.0)
    spot = paths.spot[0]
    discount = np.asarray(paths.discount, dtype=float)
    spot_disc = spot * discount
    premium = float(market.premium())
    payoff = float(market.payoff(paths)[0])
    band = ledger(spot_disc, discount, price[0], held, premium, payoff, cost)
    plain = ledger(spot_disc, discount, price[0], delta_held, premium, payoff, cost)
    variance = None
    if isinstance(market, Heston):
        variance = np.asarray(paths.features[0, :, 1], dtype=float)
    name = "Heston" if isinstance(market, Heston) else "Black-Scholes"
    return LivePath(
        market_name=name,
        seed=int(seed),
        n=n,
        half_width=float(half_width),
        cost=float(cost),
        strike=float(market.strike),
        premium=premium,
        payoff=payoff,
        payoff_disc=float(discount[-1]) * payoff,
        delta_note=_note(market),
        time=np.asarray(paths.time, dtype=float),
        spot=np.asarray(spot, dtype=float),
        discount=discount,
        variance=variance,
        delta=delta[0],
        price=price[0],
        band=band,
        delta_hedge=plain,
    )


def terminal_of(market: Market, paths: Paths, position: np.ndarray, cost: float) -> np.ndarray:
    """`terminal_pnl` of a spot holding. `position` has shape `(m, n)`."""
    return terminal_pnl(
        _spot_paths(paths),
        position,
        market.payoff(paths),
        ProportionalCost(cost),
        market.premium(),
    )


def score_books(
    market: Market,
    n: int,
    m: int,
    seed: int,
    half_width: float,
    cost: float,
    token: tuple,
    lam: float = 1.0,
) -> BatchScore:
    """Score the band and delta hedging on `m` fresh paths. `seed` is the draw."""
    if m < 2 or n < 1:
        raise ValueError("m must be at least 2 and n at least 1")
    paths = market.simulate(m, n, int(seed))
    _price, delta = _greeks(market, paths)
    band = clip_positions(delta[:, :-1], half_width)
    plain = clip_positions(delta[:, :-1], 0.0)
    pnl_band = terminal_of(market, paths, band, cost)
    pnl_delta = terminal_of(market, paths, plain, cost)
    return BatchScore(
        token=token,
        seed=int(seed),
        m=int(m),
        lam=float(lam),
        pnl_band=pnl_band,
        pnl_delta=pnl_delta,
        entropic_band=entropic(pnl_band, lam),
        entropic_delta=entropic(pnl_delta, lam),
        entropic_band_se=entropic_se(pnl_band, lam),
        entropic_delta_se=entropic_se(pnl_delta, lam),
        mean_band=float(pnl_band.mean()),
        mean_delta=float(pnl_delta.mean()),
        mean_band_se=mean_se(pnl_band),
        mean_delta_se=mean_se(pnl_delta),
    )


def _check_frame(path: LivePath, frame: int) -> int:
    frame = int(frame)
    if frame < 0 or frame > path.n:
        raise ValueError(f"frame must be between 0 and {path.n}")
    return frame


def _shares(value: float) -> str:
    return f"{value:.3f}"


def _money(value: float) -> str:
    return f"{value:.3f}"


def _band_sentence(path: LivePath, frame: int) -> str:
    delta = float(path.delta[frame])
    width = path.half_width
    lower = delta - width
    upper = delta + width
    position = float(path.band.position[frame])
    previous = 0.0 if frame == 0 else float(path.band.position[frame - 1])
    change = float(path.band.trade[frame])
    where = f"Delta is {_shares(delta)}. The band runs from {_shares(lower)} to {_shares(upper)}."
    if width <= _EDGE:
        action = (
            f"The half-width is zero, so the holding is set to delta, {_shares(position)} shares."
        )
    elif abs(change) <= _TRADE:
        action = (
            f"The holding {_shares(previous)} sits inside the band, so the hedger does not trade."
        )
    else:
        if abs(position - lower) <= _EDGE:
            edge = "lower"
        elif abs(position - upper) <= _EDGE:
            edge = "upper"
        else:
            edge = "near"
        verb = "buys" if change > 0.0 else "sells"
        action = (
            f"The holding is pushed to the {edge} edge, {_shares(position)} shares. "
            f"The hedger {verb} {_shares(abs(change))} shares."
        )
    charge = float(path.band.cost_step[frame])
    if charge > _TRADE:
        action += f" The trade costs {_money(charge)}."
    return f"{where} {action}"


def describe(path: LivePath, frame: int) -> str:
    """Plain-language account of one date. British spelling, no symbol for the prose."""
    frame = _check_frame(path, frame)
    band = path.band
    plain = path.delta_hedge
    spot = float(path.spot[frame])
    if frame == path.n:
        opening = (
            f"Maturity. The spot is {spot:.2f}. "
            f"The call pays {_money(path.payoff)} and both books are closed."
        )
        gain = float(band.gains[frame] - band.gains[frame - 1])
        opening += f" The shares held into the last step gained {_money(gain)}."
        if band.rebalanced[frame]:
            opening += " Closing the band hedge is charged."
    else:
        opening = f"Date {frame} of {path.n}. The spot is {spot:.2f}."
        if frame == 0:
            opening += f" The call is sold for {_money(path.premium)}."
        if path.variance is not None:
            opening += f" Variance is {float(path.variance[frame]):.4f}."
        opening += " " + _band_sentence(path, frame)
        if frame > 0:
            gain = float(band.gains[frame] - band.gains[frame - 1])
            opening += f" The shares held into this date gained {_money(gain)}."
    seen = frame + 1 if frame < path.n else path.n
    band_trades = int(band.rebalanced[:seen].sum())
    delta_trades = int(plain.rebalanced[:seen].sum())
    accounts = (
        f"Cash is {_money(float(band.cash[frame]))}. "
        f"The hedge P/L is {_money(float(band.pnl[frame]))}, "
        f"against {_money(float(plain.pnl[frame]))} for delta hedging. "
        f"Cumulative costs are {_money(float(band.cost[frame]))}, "
        f"against {_money(float(plain.cost[frame]))} for delta hedging, "
        f"which has traded on {delta_trades} of the {seen} dates so far. "
        f"The band has traded on {band_trades}."
    )
    return f"{opening} {accounts}"


def describe_batch(score: BatchScore) -> str:
    """The histogram in one paragraph, with the sample size and the standard errors."""
    return (
        f"On {score.m} fresh paths (seed {score.seed}), entropic risk at "
        f"lambda {score.lam:g} is {_money(score.entropic_band)} "
        f"(standard error {_money(score.entropic_band_se)}) for the band and "
        f"{_money(score.entropic_delta)} "
        f"(standard error {_money(score.entropic_delta_se)}) for delta hedging. "
        f"The mean P/L is {_money(score.mean_band)} "
        f"(standard error {_money(score.mean_band_se)}) against "
        f"{_money(score.mean_delta)} "
        f"(standard error {_money(score.mean_delta_se)}). "
        "The vertical lines are the path on screen."
    )


def _limits(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-9:
        span = max(abs(lo), 1.0) * 0.05
    else:
        span = (hi - lo) * pad
    return lo - span, hi + span


def _style(ax) -> None:
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_live(path: LivePath, frame: int) -> Figure:
    """Four panels up to `frame`: spot, delta and the band, cash, P&L and costs."""
    frame = _check_frame(path, frame)
    fig, axes = plt.subplots(4, 1, figsize=(8.6, 9.2), sharex=True)
    shown = slice(0, frame + 1)
    time = path.time[shown]
    last_trade = min(frame, path.n - 1)
    trade_time = path.time[: last_trade + 1]

    spot_ax, band_ax, cash_ax, pnl_ax = axes
    spot_ax.plot(time, path.spot[shown], color=_SPOT, lw=1.6)
    spot_ax.scatter([path.time[frame]], [path.spot[frame]], color=_SPOT, s=18, zorder=3)
    spot_ax.axhline(path.strike, color="#9ca3af", lw=0.7, ls=":")
    spot_ax.set_ylabel("spot")
    spot_ax.set_ylim(*_limits(path.spot))
    _style(spot_ax)
    if path.variance is not None:
        var_ax = spot_ax.twinx()
        var_ax.plot(time, path.variance[shown], color="#059669", lw=1.0, alpha=0.85)
        var_ax.set_ylabel("variance", color="#059669", fontsize=8)
        var_ax.set_ylim(*_limits(path.variance))
        var_ax.tick_params(labelsize=8, colors="#059669")
        var_ax.grid(False)

    lower = path.delta[: last_trade + 1] - path.half_width
    upper = path.delta[: last_trade + 1] + path.half_width
    band_ax.fill_between(trade_time, lower, upper, color=_MARK, alpha=0.18, lw=0, label="band")
    band_ax.plot(trade_time, path.delta[: last_trade + 1], color=_MARK, lw=1.3, label="delta")
    band_ax.plot(
        time, path.band.position[shown], color=_BAND, lw=1.6, label="band holding"
    )
    band_ax.plot(
        time,
        path.delta_hedge.position[shown],
        color=_DELTA,
        lw=1.0,
        ls="--",
        label="delta holding",
    )
    marks = path.band.rebalanced.copy()
    marks[frame + 1 :] = False
    if np.any(marks):
        band_ax.scatter(
            path.time[marks],
            path.band.position[marks],
            color=_BAND,
            s=22,
            zorder=4,
            label="rebalance",
        )
    band_ax.set_ylabel("shares")
    band_lo = float(min(np.min(path.delta) - path.half_width, 0.0))
    band_hi = float(max(np.max(path.delta) + path.half_width, 1.0))
    band_ax.set_ylim(*_limits(np.array([band_lo, band_hi]), pad=0.04))
    band_ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
    _style(band_ax)

    cash_ax.plot(time, path.band.cash[shown], color=_CASH, lw=1.5, label="band cash")
    cash_ax.plot(
        time, path.delta_hedge.cash[shown], color=_DELTA, lw=1.0, ls="--", label="delta cash"
    )
    cash_ax.set_ylabel("cash")
    cash_ax.set_ylim(*_limits(np.concatenate([path.band.cash, path.delta_hedge.cash])))
    cash_ax.legend(loc="best", fontsize=7, frameon=False)
    _style(cash_ax)

    pnl_ax.plot(time, path.band.pnl[shown], color=_BAND, lw=1.5, label="band P&L")
    pnl_ax.plot(
        time, path.delta_hedge.pnl[shown], color=_DELTA, lw=1.1, ls="--", label="delta P&L"
    )
    pnl_ax.plot(time, path.band.cost[shown], color=_BAND, lw=1.0, ls=":", label="band cost")
    pnl_ax.plot(
        time, path.delta_hedge.cost[shown], color=_DELTA, lw=1.0, ls=":", label="delta cost"
    )
    pnl_ax.axhline(0.0, color="#9ca3af", lw=0.6)
    pnl_ax.set_ylabel("P&L and cost")
    pnl_ax.set_xlabel("time, in years")
    stacked = np.concatenate(
        [path.band.pnl, path.delta_hedge.pnl, path.band.cost, path.delta_hedge.cost]
    )
    pnl_ax.set_ylim(*_limits(stacked))
    pnl_ax.legend(loc="best", fontsize=7, frameon=False, ncol=2)
    _style(pnl_ax)

    for ax in axes:
        ax.axvline(path.time[frame], color="#d1d5db", lw=0.8)
        ax.set_xlim(float(path.time[0]), float(path.time[-1]))
    bp = path.cost * 1e4
    fig.suptitle(
        f"{path.market_name}, seed {path.seed}, {path.n} steps, "
        f"{bp:.0f} bp, half-width {path.half_width:.2f}. Date {frame} of {path.n}.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def draw_batch(path: LivePath, score: BatchScore) -> Figure:
    """Histogram of the two books, with this path marked."""
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    pooled = np.concatenate([score.pnl_band, score.pnl_delta])
    lo, hi = np.quantile(pooled, [0.005, 0.995])
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    bins = np.linspace(lo, hi, 61)
    ax.hist(
        score.pnl_delta, bins=bins, density=True, color=_DELTA, alpha=0.45, label="delta hedging"
    )
    ax.hist(score.pnl_band, bins=bins, density=True, color=_BAND, alpha=0.55, label="band")
    ax.axvline(path.delta_hedge.pnl[-1], color=_DELTA, lw=1.2, ls="--")
    ax.axvline(path.band.pnl[-1], color=_BAND, lw=1.4)
    ax.set_xlabel("terminal P&L, time-0 money")
    ax.set_ylabel("density")
    ax.set_title(f"Same hedgers on {score.m} paths. Lines are this path.")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig
