"""Contracts: what is owed at maturity. M02 ships the vanilla call and put only.

M03 adds the rest of the contract library (digitals, barriers, Asians,
lookbacks, variance swaps) and a `Monitoring` object. Until then a contract
here has `payoff(paths)`, `payoff_torch(paths)`, `terminal_payoff(spot_T)`
for the Monte Carlo pricer, and `params()` for the graph spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deephedge.paths import Paths

__all__ = ["Vanilla"]


@dataclass(frozen=True)
class Vanilla:
    """European call or put on the spot, instrument 0. Undiscounted payoff at maturity."""

    kind: str = "call"
    strike: float = 100.0

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        if not self.strike > 0:
            raise ValueError("strike must be positive")
        object.__setattr__(self, "strike", float(self.strike))

    name = "vanilla"

    def params(self) -> dict:
        return {"kind": self.kind, "strike": self.strike}

    def terminal_payoff(self, spot_t) -> np.ndarray:
        spot_t = np.asarray(spot_t, dtype=float)
        if self.kind == "call":
            return np.maximum(spot_t - self.strike, 0.0)
        return np.maximum(self.strike - spot_t, 0.0)

    def payoff(self, paths: Paths) -> np.ndarray:
        return self.terminal_payoff(paths.spot[:, -1])

    def payoff_torch(self, paths):
        """Payoff from a `TorchPaths`, shape (m,)."""
        import torch

        spot_t = paths.instruments[:, -1, 0]
        if self.kind == "call":
            return torch.clamp(spot_t - self.strike, min=0.0)
        return torch.clamp(self.strike - spot_t, min=0.0)
