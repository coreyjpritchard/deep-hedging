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
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

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
    "LIVE_HESTON_XI",
    "clip_call_delta",
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
# The 2019 Heston uses xi = 2. On a 30-day daily grid that vol of vol
# drives the full-truncation variance through zero and leaves it there,
# so the spot freezes. The live demo keeps every other 2019 parameter
# and uses a smaller xi, which still moves variance and rarely hits zero.
LIVE_HESTON_XI = 0.5

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
    delta_raw: np.ndarray
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
    maturity 30/365, rates zero. Heston keeps the 2019 kappa, theta, v0
    and rho, with `xi` equal to `LIVE_HESTON_XI` rather than 2.
    """
    if name == "Black-Scholes":
        return BlackScholes(s0=100.0, strike=100.0, maturity=30 / 365, sigma=0.2, r=0.0, mu=0.0)
    if name == "Heston":
        return Heston(xi=LIVE_HESTON_XI)
    raise ValueError(f"market must be Black-Scholes or Heston, got {name!r}")


def clip_call_delta(delta: np.ndarray) -> np.ndarray:
    """Clip a call delta to `[0, 1]`.

    With no dividends, a vanilla call delta lies in that interval. The
    cosine expansion can print a value just outside it when variance sits
    on the truncation floor. The excess is a numerical error.
    """
    return np.clip(np.asarray(delta, dtype=float), 0.0, 1.0)


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
        line = (
            "Delta is the Heston call delta from the cosine expansion, "
            "using today's variance. The hedge does not trade the variance swap."
        )
        if market.xi < 1.0:
            line += (
                f" This demo uses xi = {market.xi:g}. The 2019 market uses xi = 2, "
                "which drives daily variance onto the truncation floor and leaves "
                "the spot frozen."
            )
        return line
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
    price, raw = _greeks(market, paths)
    delta = clip_call_delta(raw)
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
        delta_raw=raw[0],
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
    _price, raw = _greeks(market, paths)
    delta = clip_call_delta(raw)
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


def _variance_sentence(path: LivePath, frame: int) -> str:
    assert path.variance is not None
    variance = float(path.variance[frame])
    if variance <= 1e-8:
        return " Variance is on the truncation floor, so the spot's shock is off."
    return f" Variance is {variance:.4f}."


def _cosine_note(path: LivePath, frame: int) -> str:
    raw = float(path.delta_raw[frame])
    if -1e-6 <= raw <= 1.0 + 1e-6:
        return ""
    return (
        f" The cosine delta was {raw:.3f}. A call with no dividends has delta "
        "between 0 and 1, so the figure uses the clipped value."
    )


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
        opening += _cosine_note(path, frame)
    else:
        opening = f"Date {frame} of {path.n}. The spot is {spot:.2f}."
        if frame == 0:
            opening += f" The call is sold for {_money(path.premium)}."
        if path.variance is not None:
            opening += _variance_sentence(path, frame)
        opening += " " + _band_sentence(path, frame)
        opening += _cosine_note(path, frame)
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


def _range(values: np.ndarray, pad: float = 0.08) -> list[float]:
    lo, hi = _limits(values, pad)
    return [lo, hi]


def _grids(fig: Figure) -> None:
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False, tickfont=dict(size=10))


def draw_live(path: LivePath, frame: int) -> Figure:
    """Four panels up to `frame`: spot, delta and the band, cash, P/L and costs.

    The figure is plotly, so a marimo cell can send it in the output itself.
    Axis ranges cover the whole path and the lines stop at `frame`.
    """
    frame = _check_frame(path, frame)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )
    shown = slice(0, frame + 1)
    time = path.time[shown]
    last_trade = min(frame, path.n - 1)
    trade_time = path.time[: last_trade + 1]
    x_range = [float(path.time[0]), float(path.time[-1])]

    fig.add_trace(
        go.Scatter(
            x=time,
            y=path.spot[shown],
            mode="lines",
            name="spot",
            line=dict(color=_SPOT, width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[float(path.time[frame])],
            y=[float(path.spot[frame])],
            mode="markers",
            marker=dict(color=_SPOT, size=8),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=path.strike,
        line=dict(color="#9ca3af", width=1, dash="dot"),
        row=1,
        col=1,
    )
    if path.variance is not None:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=path.variance[shown],
                mode="lines",
                name="variance",
                line=dict(color="#059669", width=1.5),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    lower = path.delta[: last_trade + 1] - path.half_width
    upper = path.delta[: last_trade + 1] + path.half_width
    fig.add_trace(
        go.Scatter(
            x=trade_time,
            y=lower,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trade_time,
            y=upper,
            mode="lines",
            name="band",
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",
            line=dict(width=0),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trade_time,
            y=path.delta[: last_trade + 1],
            mode="lines",
            name="delta",
            line=dict(color=_MARK, width=1.6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=path.band.position[shown],
            mode="lines",
            name="band holding",
            line=dict(color=_BAND, width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=path.delta_hedge.position[shown],
            mode="lines",
            name="delta holding",
            line=dict(color=_DELTA, width=1.4, dash="dash"),
        ),
        row=2,
        col=1,
    )
    marks = path.band.rebalanced.copy()
    marks[frame + 1 :] = False
    if np.any(marks):
        fig.add_trace(
            go.Scatter(
                x=path.time[marks],
                y=path.band.position[marks],
                mode="markers",
                name="rebalance",
                marker=dict(color=_BAND, size=8),
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=path.band.cash[shown],
            mode="lines",
            name="band cash",
            line=dict(color=_CASH, width=2),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=path.delta_hedge.cash[shown],
            mode="lines",
            name="delta cash",
            line=dict(color=_DELTA, width=1.4, dash="dash"),
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=path.band.pnl[shown],
            mode="lines",
            name="band P/L",
            line=dict(color=_BAND, width=2),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=path.delta_hedge.pnl[shown],
            mode="lines",
            name="delta P/L",
            line=dict(color=_DELTA, width=1.5, dash="dash"),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=path.band.cost[shown],
            mode="lines",
            name="band cost",
            line=dict(color=_BAND, width=1.3, dash="dot"),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=path.delta_hedge.cost[shown],
            mode="lines",
            name="delta cost",
            line=dict(color=_DELTA, width=1.3, dash="dot"),
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=0.0, line=dict(color="#9ca3af", width=1), row=4, col=1)
    for row in range(1, 5):
        fig.add_vline(
            x=float(path.time[frame]),
            line=dict(color="#d1d5db", width=1),
            row=row,
            col=1,
        )

    fig.update_yaxes(title_text="spot", range=_range(path.spot), row=1, col=1)
    if path.variance is not None:
        fig.update_yaxes(
            title_text="variance",
            range=_range(path.variance),
            secondary_y=True,
            row=1,
            col=1,
            showgrid=False,
            title_font=dict(color="#059669", size=11),
            tickfont=dict(color="#059669", size=10),
        )
    share_lo = float(min(np.min(path.delta) - path.half_width, 0.0))
    share_hi = float(max(np.max(path.delta) + path.half_width, 1.0))
    fig.update_yaxes(
        title_text="shares",
        range=_range(np.array([share_lo, share_hi]), pad=0.04),
        row=2,
        col=1,
    )
    cash = np.concatenate([path.band.cash, path.delta_hedge.cash])
    fig.update_yaxes(title_text="cash", range=_range(cash), row=3, col=1)
    stacked = np.concatenate(
        [path.band.pnl, path.delta_hedge.pnl, path.band.cost, path.delta_hedge.cost]
    )
    fig.update_yaxes(title_text="P/L and cost", range=_range(stacked), row=4, col=1)
    fig.update_xaxes(range=x_range)
    fig.update_xaxes(title_text="time, in years", row=4, col=1)

    bp = path.cost * 1e4
    right = 58 if path.variance is not None else 18
    fig.update_layout(
        title=dict(
            text=(
                f"{path.market_name}, seed {path.seed}, {path.n} steps<br>"
                f"{bp:.0f} bp, half-width {path.half_width:.2f}, "
                f"date {frame} of {path.n}"
            ),
            font=dict(size=14),
            x=0,
            xanchor="left",
        ),
        height=700,
        autosize=True,
        showlegend=False,
        margin=dict(l=58, r=right, t=64, b=42),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11, color="#111827"),
        hovermode="x unified",
    )
    _grids(fig)
    return fig


def draw_batch(path: LivePath, score: BatchScore) -> Figure:
    """Histogram of the two books, with this path marked."""
    fig = go.Figure()
    pooled = np.concatenate([score.pnl_band, score.pnl_delta])
    lo = float(np.quantile(pooled, 0.005))
    hi = float(np.quantile(pooled, 0.995))
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    bins = dict(start=lo, end=hi, size=(hi - lo) / 60.0)
    fig.add_trace(
        go.Histogram(
            x=score.pnl_delta,
            name="delta hedging",
            histnorm="probability density",
            xbins=bins,
            marker_color="rgba(75, 85, 99, 0.45)",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=score.pnl_band,
            name="band",
            histnorm="probability density",
            xbins=dict(bins),
            marker_color="rgba(194, 65, 12, 0.55)",
        )
    )
    fig.add_vline(
        x=float(path.delta_hedge.pnl[-1]),
        line=dict(color=_DELTA, width=1.5, dash="dash"),
    )
    fig.add_vline(x=float(path.band.pnl[-1]), line=dict(color=_BAND, width=1.6))
    fig.update_layout(
        barmode="overlay",
        title=dict(
            text=f"Same hedgers on {score.m} paths. Lines are this path.",
            font=dict(size=14),
            x=0,
            xanchor="left",
        ),
        height=340,
        autosize=True,
        margin=dict(l=52, r=16, t=56, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11, color="#111827"),
        xaxis_title="terminal P/L, time-0 money",
        yaxis_title="density",
    )
    _grids(fig)
    return fig
