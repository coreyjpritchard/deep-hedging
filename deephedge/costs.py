"""Transaction costs. Pluggable, and independent of the volatility model.

The P&L engine calls `amount` with discounted instrument prices, so a
charge on the price basis is already in time-0 money. The fixed charge
and a numeric `unit` are cash on the day of the trade and are not
discounted, so the whole amount is in time-0 money only for the price
basis or when `r = 0`. Every session uses `r = 0`. Positions have shape
`(m, n, d)`. The book is flat before the first date and flat again at
maturity, and both of those trades are charged. Liquidation is not
optional: a position held to expiry pays the closing trade.

Three models:

* `ProportionalCost`: `rate * |change in position| * basis`. This is the
  cost in Buehler, Gonon, Teichmann and Wood (2019), section 3, when the
  basis is the instrument price.
* `FixedPlusProportional`: that charge, plus a cash amount each time an
  instrument's position changes, including the open and the liquidation.
* `QuadraticCost`: `impact * (change in position)^2 * basis`. A simple
  market-impact charge. It is smooth, so training and evaluation use the
  same formula.

The basis is the discounted price by default. Pass `unit` to charge
against a cash notional per unit of position instead. The variance swap's
mark is an integrated variance of order 10^-3, so a rate times that mark
barely charges the swap. `unit=(None, 1.0)` keeps the spot on its price
and charges one currency unit per variance swap traded. The swap's price,
the gains and the hedge ratio are unchanged. Only the cost uses the unit.

`rate`, `fixed`, `impact` and `unit` are a scalar, broadcast to every
instrument, or a tuple of length `d`. A `None` entry in `unit` means
"this instrument's price".

The fixed cost is an indicator, which has no useful derivative. Numpy
`amount`, and torch `amount_torch(..., exact=True)`, use that indicator.
Training calls `amount_torch(..., exact=False)`, which replaces the
indicator by `tanh(|change| / width)`. That is 0 when the position does
not change, near 1 once the change is several widths, and differentiable
at 0. Evaluation goes through numpy and is exact. `width` is in position
units and does not enter the exact charge. The default `1e-2` treats
a trade of a few hundredths of a share as essentially a full fixed
cost, and a numerical dust trade as free.
A daily Black-Scholes delta move is about that size, so a width of
`1e-2` is already saturated there and the training step gets no
gradient from the fixed cost. A run that has to learn whether to
trade at all should pick a larger width. Session 02 uses `0.05`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FixedPlusProportional",
    "ProportionalCost",
    "QuadraticCost",
    "as_cost",
    "scaled",
]


def _as_positions(positions: np.ndarray, n_instruments: int) -> np.ndarray:
    """Positions as an array of shape (m, n, d). A (m, n) array means d = 1."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 2:
        if n_instruments != 1:
            raise ValueError(f"positions have shape {pos.shape}; expected (m, n, {n_instruments})")
        return pos[:, :, None]
    if pos.ndim != 3 or pos.shape[-1] != n_instruments:
        raise ValueError(f"positions have shape {pos.shape}; expected (m, n, {n_instruments})")
    return pos


def _check_alignment(prices: np.ndarray, pos: np.ndarray) -> None:
    if pos.shape[0] != prices.shape[0] or pos.shape[1] != prices.shape[1] - 1:
        raise ValueError(f"positions shape {pos.shape} does not match prices shape {prices.shape}")


def traded_changes(positions: np.ndarray) -> np.ndarray:
    """Absolute position changes, shape `(m, n + 1, d)`, open through liquidation.

    `positions` has shape `(m, n, d)`. The book starts flat and ends flat.
    """
    pos = np.asarray(positions, dtype=float)
    m, _n, d = pos.shape
    flat = np.zeros((m, 1, d))
    full = np.concatenate([flat, pos, flat], axis=1)
    return np.abs(np.diff(full, axis=1))


def _rates(value, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or a length-n tuple to shape (n,)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} has shape {arr.shape}, expected {(n,)} or a scalar")
    return arr


def charge_basis(prices: np.ndarray, unit) -> np.ndarray:
    """What `|change in position|` is multiplied by. Same shape as `prices`.

    `unit is None`: the discounted price, one column per instrument.
    A number: that cash notional for every instrument, constant along the path.
    A tuple of length `d`: a cash notional per instrument, or `None` in an
    entry to keep that instrument on its price.
    """
    prices = np.asarray(prices, dtype=float)
    if unit is None:
        return prices
    d = prices.shape[-1]
    if isinstance(unit, (int, float, np.floating)):
        return np.full_like(prices, float(unit))
    if isinstance(unit, tuple):
        if len(unit) != d:
            raise ValueError(f"unit has length {len(unit)}, expected {d}")
        columns = []
        for j, entry in enumerate(unit):
            if entry is None:
                columns.append(prices[:, :, j])
            else:
                columns.append(np.full(prices.shape[:2], float(entry)))
        return np.stack(columns, axis=-1)
    raise TypeError(f"unit must be a number, a tuple, or None, got {type(unit).__name__}")


def _prepare(prices, positions):
    prices = np.asarray(prices, dtype=float)
    pos = _as_positions(positions, prices.shape[-1])
    _check_alignment(prices, pos)
    return prices, pos, traded_changes(pos)


def _torch_traded(positions):
    import torch

    m, _n, d = positions.shape
    flat = torch.zeros(m, 1, d, dtype=positions.dtype, device=positions.device)
    full = torch.cat([flat, positions, flat], dim=1)
    return (full[:, 1:] - full[:, :-1]).abs()


def _torch_basis(prices, unit):
    import torch

    if unit is None:
        return prices
    d = prices.shape[-1]
    if isinstance(unit, (int, float, np.floating)) and not isinstance(unit, bool):
        return torch.full_like(prices, float(unit))
    if isinstance(unit, tuple):
        if len(unit) != d:
            raise ValueError(f"unit has length {len(unit)}, expected {d}")
        columns = []
        for j, entry in enumerate(unit):
            if entry is None:
                columns.append(prices[:, :, j])
            else:
                columns.append(
                    torch.full(
                        prices.shape[:2],
                        float(entry),
                        dtype=prices.dtype,
                        device=prices.device,
                    )
                )
        return torch.stack(columns, dim=-1)
    raise TypeError(f"unit must be a number, a tuple, or None, got {type(unit).__name__}")


def _torch_rates(value, n: int, dtype, device, name: str):
    import torch

    if isinstance(value, tuple):
        if len(value) != n:
            raise ValueError(f"{name} has length {len(value)}, expected {n}")
        return torch.tensor(value, dtype=dtype, device=device)
    return torch.as_tensor(float(value), dtype=dtype, device=device)


@dataclass(frozen=True)
class ProportionalCost:
    """Proportional cost `rate * |change in position| * basis`.

    `rate` is a scalar, or one rate per instrument. `unit` selects the
    basis: the price when `unit` is `None`, or a cash notional. The
    opening trade, every rebalance, and the liquidating trade are included.

    With `unit=None` this is the cost in Buehler et al. (2019), section 3.
    """

    rate: float | tuple[float, ...] = 0.0
    unit: float | tuple[float | None, ...] | None = None

    def amount(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        prices, _pos, traded = _prepare(prices, positions)
        rate = _rates(self.rate, prices.shape[-1], "rate")
        basis = charge_basis(prices, self.unit)
        return (rate * traded * basis).sum(axis=(1, 2))

    def per_date(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Cost paid at each date, shape `(m, n + 1)`. Sums to `amount` per path."""
        prices, _pos, traded = _prepare(prices, positions)
        rate = _rates(self.rate, prices.shape[-1], "rate")
        return (rate * traded * charge_basis(prices, self.unit)).sum(axis=2)

    def amount_torch(self, prices, positions, *, exact: bool = True):
        """Torch version of `amount`. `exact` is unused: `|x|` is already usable in training.

        `|x|` is differentiable away from zero. At zero the subgradient is 0,
        which is the same convention the one-instrument training loss uses.
        """
        del exact
        traded = _torch_traded(positions)
        rate = _torch_rates(self.rate, prices.shape[-1], prices.dtype, prices.device, "rate")
        basis = _torch_basis(prices, self.unit)
        return (rate * traded * basis).sum(dim=(1, 2))

    def label(self) -> str:
        return f"ProportionalCost(rate={self.rate}, unit={self.unit})"


@dataclass(frozen=True)
class FixedPlusProportional:
    """Cash when an instrument trades, plus a proportional charge.

    `fixed` is time-0 cash per instrument per date on which that
    instrument's position changes. It is not multiplied by the price.
    `amount` does not see the discount curve, so a later trade is not
    automatically smaller when `r > 0`. Sessions use `r = 0`, where the
    charge is the cash paid on the day. Opening and liquidation count,
    so a constant holding pays `fixed` twice and nothing in between. A
    scalar `fixed` is charged on every instrument that trades.

    Numpy, and `amount_torch(..., exact=True)`, use the indicator of a
    nonzero change. `amount_torch(..., exact=False)` uses
    `tanh(|change| / width)` so a training step has a gradient. Evaluation
    should use `amount`, which is exact.
    """

    rate: float | tuple[float, ...] = 0.0
    fixed: float | tuple[float, ...] = 0.0
    unit: float | tuple[float | None, ...] | None = None
    width: float = 1e-2

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")

    def amount(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        prices, _pos, traded = _prepare(prices, positions)
        d = prices.shape[-1]
        rate = _rates(self.rate, d, "rate")
        fixed = _rates(self.fixed, d, "fixed")
        basis = charge_basis(prices, self.unit)
        proportional = (rate * traded * basis).sum(axis=(1, 2))
        hits = (traded > 0).astype(float)
        return proportional + (fixed * hits).sum(axis=(1, 2))

    def per_date(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Cost paid at each date, shape `(m, n + 1)`, with the exact indicator."""
        prices, _pos, traded = _prepare(prices, positions)
        d = prices.shape[-1]
        rate = _rates(self.rate, d, "rate")
        fixed = _rates(self.fixed, d, "fixed")
        proportional = (rate * traded * charge_basis(prices, self.unit)).sum(axis=2)
        return proportional + (fixed * (traded > 0).astype(float)).sum(axis=2)

    def amount_torch(self, prices, positions, *, exact: bool = True):
        """Torch version of `amount`. `exact=False` is the smooth indicator used in training."""
        traded = _torch_traded(positions)
        rate = _torch_rates(self.rate, prices.shape[-1], prices.dtype, prices.device, "rate")
        fixed = _torch_rates(self.fixed, prices.shape[-1], prices.dtype, prices.device, "fixed")
        basis = _torch_basis(prices, self.unit)
        proportional = (rate * traded * basis).sum(dim=(1, 2))
        if exact:
            hits = (traded > 0).to(prices.dtype)
        else:
            hits = torch_tanh(traded / self.width)
        return proportional + (fixed * hits).sum(dim=(1, 2))

    def label(self) -> str:
        return f"FixedPlusProportional(rate={self.rate}, fixed={self.fixed}, unit={self.unit})"


def torch_tanh(values):
    """`tanh`, imported lazily so importing this module does not require a torch op at load."""
    import torch

    return torch.tanh(values)


@dataclass(frozen=True)
class QuadraticCost:
    """Market-impact cost `impact * (change in position)^2 * basis`.

    Smooth, so `amount` and `amount_torch` are the same formula and `exact`
    is unused. Opening and liquidation are included. `impact` is a scalar
    or one coefficient per instrument.
    """

    impact: float | tuple[float, ...] = 0.0
    unit: float | tuple[float | None, ...] | None = None

    def amount(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        prices, _pos, traded = _prepare(prices, positions)
        impact = _rates(self.impact, prices.shape[-1], "impact")
        basis = charge_basis(prices, self.unit)
        return (impact * traded**2 * basis).sum(axis=(1, 2))

    def per_date(self, prices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Cost paid at each date, shape `(m, n + 1)`. Sums to `amount` per path."""
        prices, _pos, traded = _prepare(prices, positions)
        impact = _rates(self.impact, prices.shape[-1], "impact")
        return (impact * traded**2 * charge_basis(prices, self.unit)).sum(axis=2)

    def amount_torch(self, prices, positions, *, exact: bool = True):
        """Torch version of `amount`. `exact` is unused."""
        del exact
        traded = _torch_traded(positions)
        impact = _torch_rates(self.impact, prices.shape[-1], prices.dtype, prices.device, "impact")
        basis = _torch_basis(prices, self.unit)
        return (impact * traded**2 * basis).sum(dim=(1, 2))

    def label(self) -> str:
        return f"QuadraticCost(impact={self.impact}, unit={self.unit})"


def scaled(cost, scale: float):
    """The same model with `rate`, `fixed` and `impact` multiplied by `scale`.

    `unit` and `width` are unchanged. `scale = 0` is a zero-cost model of
    the same family. A number or a tuple is treated as `ProportionalCost`.
    """
    scale = float(scale)
    model = as_cost(cost)
    if isinstance(model, ProportionalCost):
        return ProportionalCost(_scale_value(model.rate, scale), model.unit)
    if isinstance(model, FixedPlusProportional):
        return FixedPlusProportional(
            _scale_value(model.rate, scale),
            _scale_value(model.fixed, scale),
            model.unit,
            model.width,
        )
    if isinstance(model, QuadraticCost):
        return QuadraticCost(_scale_value(model.impact, scale), model.unit)
    raise TypeError(f"cannot scale {type(model).__name__}")


def _scale_value(value, scale: float):
    if isinstance(value, tuple):
        return tuple(float(v) * scale for v in value)
    return float(value) * scale


def as_cost(cost):
    """A cost model, or a proportional rate standing in for one.

    A number or a tuple of numbers is `ProportionalCost` on the price,
    which is the call the rest of the package already used.
    """
    if isinstance(cost, (ProportionalCost, FixedPlusProportional, QuadraticCost)):
        return cost
    if isinstance(cost, tuple):
        return ProportionalCost(cost)
    if isinstance(cost, (int, float, np.floating)):
        return ProportionalCost(float(cost))
    if hasattr(cost, "amount"):
        return cost
    raise TypeError(f"cost must be a rate or a CostModel, got {type(cost).__name__}")
