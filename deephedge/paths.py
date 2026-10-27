"""One Monte Carlo batch: the time grid, tradable prices, observable features, named state.

`Paths` is what every generator returns and what the P&L, the hedgers and the
pricers read. Shapes follow `docs/plan/01-interfaces.md`:

* `time (n + 1,)` from 0 to maturity, column 0 is `t = 0`;
* `instruments (m, n + 1, d)`, the tradable prices that enter the P&L, with
  instrument 0 the spot;
* `features (m, n + 1, f)`, what a hedger may observe, feature 0 the spot;
* `discount (n + 1,)` or `(m, n + 1)`, factors to time 0;
* `state`, named extras of shape `(m, n + 1)` or `(m, n + 1, k)`, such as the
  variance; a feature block or a pricer reads them by name;
* `instrument_names` and `feature_names`, one per column;
* `source`, the generator's native object when an adapter wraps another
  package (decision D4), passed through untouched.

`state`, the names and `source` have defaults, so code written against the
four-field `Paths` of PR 1 still builds one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = ["Paths", "TorchPaths"]


def _default_names(prefix: str, count: int) -> tuple[str, ...]:
    if count == 1:
        return ("spot",)
    return ("spot",) + tuple(f"{prefix}_{i}" for i in range(1, count))


def _check_names(names, count: int, what: str) -> tuple[str, ...]:
    names = tuple(names)
    if len(names) != count:
        raise ValueError(f"{what} has {len(names)} names for {count} columns: {names}")
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError(f"{what} must be non-empty strings, got {names}")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} must be distinct, got {names}")
    return names


@dataclass(frozen=True)
class Paths:
    """One batch of `m` paths on `n` steps. See the module docstring for shapes."""

    time: np.ndarray
    instruments: np.ndarray
    features: np.ndarray
    discount: np.ndarray
    state: Mapping[str, np.ndarray] = field(default_factory=dict)
    instrument_names: tuple[str, ...] | None = None
    feature_names: tuple[str, ...] | None = None
    source: Any = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        instruments = np.asarray(self.instruments, dtype=float)
        features = np.asarray(self.features, dtype=float)
        discount = np.asarray(self.discount, dtype=float)
        if instruments.ndim != 3:
            raise ValueError(f"instruments must have shape (m, n + 1, d), got {instruments.shape}")
        m, n_plus, d = instruments.shape
        if features.ndim != 3 or features.shape[:2] != (m, n_plus):
            raise ValueError(
                "features must share the path and time axes of instruments, "
                f"got {features.shape} against {instruments.shape}"
            )
        if time.shape != (n_plus,):
            raise ValueError(f"time must have shape ({n_plus},), got {time.shape}")
        if discount.shape != (n_plus,) and discount.shape != (m, n_plus):
            raise ValueError(
                f"discount must have shape ({n_plus},) or ({m}, {n_plus}), got {discount.shape}"
            )
        state = {}
        for name, value in dict(self.state).items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"state names must be non-empty strings, got {name!r}")
            array = np.asarray(value, dtype=float)
            if array.ndim not in (2, 3) or array.shape[:2] != (m, n_plus):
                raise ValueError(
                    f"state {name!r} must have shape ({m}, {n_plus}) or ({m}, {n_plus}, k), "
                    f"got {array.shape}"
                )
            state[name] = array
        instrument_names = (
            _default_names("instrument", d)
            if self.instrument_names is None
            else _check_names(self.instrument_names, d, "instrument_names")
        )
        feature_names = (
            _default_names("feature", features.shape[-1])
            if self.feature_names is None
            else _check_names(self.feature_names, features.shape[-1], "feature_names")
        )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "instruments", instruments)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "state", MappingProxyType(state))
        object.__setattr__(self, "instrument_names", instrument_names)
        object.__setattr__(self, "feature_names", feature_names)

    @property
    def spot(self) -> np.ndarray:
        """First instrument, shape (m, n + 1). The underlying of a call."""
        return self.instruments[:, :, 0]

    @property
    def n_paths(self) -> int:
        return int(self.instruments.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.time.size - 1)

    @property
    def n_instruments(self) -> int:
        return int(self.instruments.shape[-1])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[-1])

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(self.state)

    def discounted_instruments(self) -> np.ndarray:
        """Instrument prices in time-0 money, shape (m, n + 1, d)."""
        if self.discount.ndim == 1:
            return self.instruments * self.discount[None, :, None]
        return self.instruments * self.discount[:, :, None]

    def discount_terminal(self) -> np.ndarray | float:
        """Discount factor at maturity. A scalar, or shape (m,) if rates vary."""
        if self.discount.ndim == 1:
            return self.discount[-1]
        return self.discount[:, -1]

    def to_torch(self, dtype=None) -> TorchPaths:
        """The same arrays as torch tensors. float32 unless `dtype` says otherwise."""
        import torch

        dtype = torch.float32 if dtype is None else dtype
        return TorchPaths(
            time=torch.as_tensor(self.time, dtype=dtype),
            instruments=torch.as_tensor(self.instruments, dtype=dtype),
            features=torch.as_tensor(self.features, dtype=dtype),
            discount=torch.as_tensor(self.discount, dtype=dtype),
            state={k: torch.as_tensor(v, dtype=dtype) for k, v in self.state.items()},
            instrument_names=self.instrument_names,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True)
class TorchPaths:
    """`Paths` as torch tensors, for training. Built by `Paths.to_torch`."""

    time: Any
    instruments: Any
    features: Any
    discount: Any
    state: Mapping[str, Any]
    instrument_names: tuple[str, ...]
    feature_names: tuple[str, ...]

    @property
    def spot(self):
        return self.instruments[:, :, 0]
