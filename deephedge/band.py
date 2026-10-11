"""The Whalley-Wilmott no-trade band for a Black-Scholes call.

Whalley and Wilmott (1997, Mathematical Finance 7(3), equation (17) and
section 4) analyse the Davis, Panas and Zariphopoulou utility-hedging
problem for small proportional costs. A trade of N shares loses
`epsilon * N * S`. That is the same charge as `ProportionalCost(epsilon)`
in this package: one way, on the mid price, including the open and,
in our engine, the liquidation.

Their utility is `U(x) = 1 - exp(-gamma x)`. The constant does not
change the maximiser, so this is exponential utility with absolute risk
aversion `gamma` (units: one over a unit of terminal wealth). Maximising
expected utility is the same problem as minimising the entropic risk
`(1 / gamma) log E[exp(-gamma W)]`. Our `entropic` is that functional of
the time-0 P&L. At `r = 0` the time-0 P&L is the change in wealth, so
`gamma = lambda`. At `r > 0`, utility of terminal wealth `W_T` and
entropic risk of the discounted P&L `X_0 = exp(-r T) W_T` match when
`lambda = gamma * exp(r T)`, because
`exp(-gamma W_T) = exp(-(gamma exp(r T)) X_0)`.

When the real-world drift equals the risk-free rate, the centre of the
band is the Black-Scholes delta. The half-width, in shares, is

    (3/2 * epsilon * S * exp(-r (T - t)) * Gamma**2 / gamma) ** (1/3).

`Gamma` is the Black-Scholes gamma, in one over a unit of spot.
`epsilon * S * Gamma**2 / gamma` is dimensionless, and so is the cube
root: the half-width is a number of shares. The discount factor is in
Whalley and Wilmott's section 4 display,
`epsilon**(1/3) * (3 * S * delta * Gamma**2 / (2 * gamma))**(1/3)`
with `delta = exp(-r (T - t))`. It equals 1 when `r = 0`, which is the
formula used by PFHedge. Zakamouline (2006) writes the same expression.
The formula in session 00 is this one. It is not a free hypothesis once
`mu = r`.

A drift `mu != r` adds the Merton term
`exp(-r (T - t)) * (mu - r) / (gamma * S * sigma**2)` to the centre.
This session has `mu = r = 0`, so that term is zero and is not applied.

Inside the band the hedger does not trade. Outside it, the hedger trades
to the nearest edge, not to delta. A fixed-width band is the same rule
with a half-width that does not depend on spot or time.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm

from deephedge.env import bs_call_delta
from deephedge.policy import position_grid
from deephedge.risk import entropic, entropic_se, mean_se

__all__ = [
    "band_positions",
    "bs_call_gamma",
    "cloud_half_width",
    "describe_rule",
    "fit_power_exponent",
    "score_pnl",
    "whalley_wilmott_half_width",
]


def bs_call_gamma(S, K: float, tau, sigma: float, r: float = 0.0):
    """Black-Scholes call gamma. Vectorised in S and tau. Zero at expiry."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.where(tau > 0, tau, 1.0)
    sd = sigma * np.sqrt(safe_tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * safe_tau) / sd
    gamma = norm.pdf(d1) / (S * sd)
    return np.where(tau > 0, gamma, 0.0)


def whalley_wilmott_half_width(
    spot,
    tau,
    cost: float,
    risk_aversion: float,
    strike: float,
    sigma: float,
    r: float = 0.0,
):
    """Half-width of the Whalley-Wilmott band, in shares.

    `cost` is the one-way proportional rate (their epsilon).
    `risk_aversion` is their gamma. See the module docstring for units
    and for the entropic lambda.
    """
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    if cost < 0:
        raise ValueError("cost must be non-negative")
    spot = np.asarray(spot, dtype=float)
    tau = np.asarray(tau, dtype=float)
    gamma = bs_call_gamma(spot, strike, tau, sigma, r)
    discount = np.exp(-r * tau)
    raw = 1.5 * float(cost) * spot * discount * gamma**2 / float(risk_aversion)
    return np.where(raw > 0.0, np.cbrt(raw), 0.0)


def band_positions(
    spot: np.ndarray,
    time: np.ndarray,
    strike: float,
    sigma: float,
    r: float,
    half_width: float | Callable,
) -> np.ndarray:
    """Positions from trading to the edge of a band around delta.

    `spot` has shape `(m, n + 1)` and includes t = 0. The book starts
    flat. At each date the holding is left unchanged when it lies between
    `delta - w` and `delta + w`, and otherwise it is moved to the nearer
    edge. `half_width` is a number of shares, or a function `(spot, tau)`
    returning an array of that shape. Returns shape `(m, n)`.
    """
    spot = np.asarray(spot, dtype=float)
    time = np.asarray(time, dtype=float)
    m, n_plus = spot.shape
    n = n_plus - 1
    if time.shape != (n_plus,):
        raise ValueError(f"time has shape {time.shape}, expected {(n_plus,)}")
    held = np.zeros(m)
    out = np.empty((m, n))
    maturity = float(time[-1])
    for k in range(n):
        tau = maturity - float(time[k])
        column = spot[:, k]
        delta = bs_call_delta(column, strike, tau, sigma, r)
        if callable(half_width):
            width = np.asarray(half_width(column, tau), dtype=float)
        else:
            width = float(half_width)
        held = np.clip(held, delta - width, delta + width)
        out[:, k] = held
    return out


def score_pnl(pnl: np.ndarray, lam: float) -> dict[str, float]:
    """Mean P&L and entropic risk, each with a standard error. Sample size `n`."""
    x = np.asarray(pnl, dtype=float)
    return {
        "mean": float(x.mean()),
        "mean_se": mean_se(x),
        "entropic": entropic(x, lam),
        "entropic_se": entropic_se(x, lam),
        "n": int(x.size),
    }


def fit_power_exponent(costs, widths) -> dict[str, float]:
    """OLS slope of log(width) on log(cost), with the regression standard error.

    The standard error is the usual one for an i.i.d. Gaussian residual.
    On a designed grid of costs it measures lack of fit, not sampling
    error of a Monte Carlo width. At least three positive pairs are
    required. A pure cube-root law has exponent 1/3 and a residual of zero.
    """
    cost = np.asarray(costs, dtype=float)
    width = np.asarray(widths, dtype=float)
    mask = (cost > 0.0) & (width > 0.0) & np.isfinite(width)
    n = int(mask.sum())
    if n < 3:
        raise ValueError(f"need at least three positive widths, got {n}")
    x = np.log(cost[mask])
    y = np.log(width[mask])
    design = np.column_stack([np.ones(n), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = n - 2
    sigma2 = float(np.sum(resid**2) / dof)
    xtx_inv = np.linalg.inv(design.T @ design)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    return {
        "exponent": float(coef[1]),
        "se": se,
        "n": float(n),
        "intercept": float(coef[0]),
    }


def cloud_half_width(
    held: np.ndarray,
    spot: np.ndarray,
    delta: np.ndarray,
    spot_lo: float,
    spot_hi: float,
) -> float:
    """Half the 10-to-90 percentile range of held minus delta, near the money.

    Paths with spot outside `[spot_lo, spot_hi]` are dropped. This is the
    cloud session 00 measured, not the decision-rule width.
    """
    mask = (spot >= spot_lo) & (spot <= spot_hi)
    gap = np.asarray(held, dtype=float)[mask] - np.asarray(delta, dtype=float)[mask]
    if gap.size < 20:
        raise ValueError(f"only {gap.size} paths in the spot window")
    lo, hi = np.quantile(gap, [0.1, 0.9])
    return float(0.5 * (hi - lo))


def describe_rule(
    policy,
    t: float,
    spots: np.ndarray,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    prevs: np.ndarray | None = None,
    trade_tol: float = 0.005,
) -> dict[str, np.ndarray]:
    """Read a hedging rule as a centre, a slope, and two widths.

    For each spot, the centre is the midpoint of the set of incoming
    positions the rule leaves unchanged, or the unique fixed point if that
    set is a single root. The slope is the derivative of the chosen
    position with respect to the incoming position at that centre. A sharp
    band has slope 1 inside and 0 outside.

    `no_trade_half_width` is the largest deviation from the centre at which
    the absolute trade stays within `trade_tol` shares (default half a
    hundredth of a share). A sharp band of half-width w returns w, up to
    the spacing of the incoming-position grid. A rule that pulls toward
    its centre on every deviation returns a width near 0.

    `sharp_half_width` is the half-width of the sharp band, centred at
    that centre, closest in square error to the rule. `sharp_rmse` is
    that error. A soft rule can have a large `sharp_rmse`.
    """
    spots = np.asarray(spots, dtype=float)
    if prevs is None:
        prevs = np.linspace(-0.2, 1.2, 141)
    prevs = np.asarray(prevs, dtype=float)
    grid = position_grid(policy, t, spots, prevs)
    tau = maturity - t
    delta = bs_call_delta(spots, strike, tau, sigma, r)
    n_spot = spots.size
    centre = np.empty(n_spot)
    slope = np.empty(n_spot)
    no_trade = np.empty(n_spot)
    sharp = np.empty(n_spot)
    rmse = np.empty(n_spot)
    width_grid = np.linspace(0.0, 0.8, 161)
    for j in range(n_spot):
        chosen = grid[:, j]
        gap = chosen - prevs
        centre[j] = _fixed_point(prevs, gap)
        slope[j] = _slope_at(prevs, chosen, centre[j])
        no_trade[j] = _no_trade_half_width(prevs, chosen, centre[j], trade_tol)
        sharp[j], rmse[j] = _best_sharp_width(prevs, chosen, centre[j], width_grid)
    return {
        "spot": spots,
        "delta": delta,
        "centre": centre,
        "centre_offset": centre - delta,
        "slope": slope,
        "no_trade_half_width": no_trade,
        "sharp_half_width": sharp,
        "sharp_rmse": rmse,
    }


def _fixed_point(prevs: np.ndarray, gap: np.ndarray) -> float:
    """Midpoint of the flat fixed set, or a root of chosen minus prev."""
    quiet = np.flatnonzero(np.abs(gap) <= 1e-8)
    if quiet.size:
        return float(0.5 * (prevs[quiet[0]] + prevs[quiet[-1]]))
    sign = np.sign(gap)
    crosses = np.where(sign[:-1] * sign[1:] < 0.0)[0]
    if crosses.size:
        i = int(crosses[len(crosses) // 2])
        g0, g1 = float(gap[i]), float(gap[i + 1])
        weight = abs(g0) / (abs(g0) + abs(g1))
        return float(prevs[i] + weight * (prevs[i + 1] - prevs[i]))
    return float(prevs[int(np.argmin(np.abs(gap)))])


def _slope_at(prevs: np.ndarray, chosen: np.ndarray, centre: float) -> float:
    i = int(np.argmin(np.abs(prevs - centre)))
    i0 = max(i - 1, 0)
    i1 = min(i + 1, prevs.size - 1)
    return float((chosen[i1] - chosen[i0]) / (prevs[i1] - prevs[i0]))


def _no_trade_half_width(
    prevs: np.ndarray, chosen: np.ndarray, centre: float, trade_tol: float
) -> float:
    """Largest deviation that is still inside a contiguous almost-flat region."""
    deviation = np.abs(prevs - centre)
    closed = np.abs(chosen - prevs)
    order = np.argsort(deviation)
    width = 0.0
    for i in order:
        if closed[i] <= trade_tol:
            width = float(deviation[i])
        else:
            break
    return width


def _best_sharp_width(
    prevs: np.ndarray, chosen: np.ndarray, centre: float, width_grid: np.ndarray
) -> tuple[float, float]:
    best_w = 0.0
    best_err = np.inf
    for width in width_grid:
        sharp = np.clip(prevs, centre - width, centre + width)
        err = float(np.mean((sharp - chosen) ** 2))
        if err < best_err:
            best_err = err
            best_w = float(width)
    return best_w, float(np.sqrt(best_err))
