"""Convex risk measures of a P&L X. Larger P&L is better; larger risk is worse.

Both measures are monetary: rho(X + c) = rho(X) - c. So rho(X) is the cash that must be
added to the position to make it acceptable.

The argument is a sample of terminal P&L. These functions do not take a market,
a payoff or a cost model. Whatever environment produced the sample is upstream,
in `deephedge.env`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import logsumexp


def entropic(pnl: np.ndarray, lam: float) -> float:
    """Entropic risk (1 / lam) log E[exp(-lam X)], computed stably with logsumexp."""
    x = np.asarray(pnl, dtype=float)
    return float((logsumexp(-lam * x) - math.log(x.size)) / lam)


def cvar(pnl: np.ndarray, alpha: float) -> float:
    """Conditional value at risk of the loss -X at confidence level alpha.

    The mean loss over the worst (1 - alpha) fraction of outcomes. alpha = 0.95 averages
    the worst 5 per cent. Uses the empirical tail of k = ceil((1 - alpha) m) samples.
    """
    loss = -np.asarray(pnl, dtype=float)
    k = max(1, math.ceil((1.0 - alpha) * loss.size))
    return float(np.partition(loss, loss.size - k)[loss.size - k :].mean())


def entropic_torch(pnl: torch.Tensor, lam: float) -> torch.Tensor:
    """Torch version of `entropic`, differentiable in pnl."""
    return (torch.logsumexp(-lam * pnl, dim=0) - math.log(pnl.numel())) / lam


def cvar_torch(pnl: torch.Tensor, alpha: float) -> torch.Tensor:
    """Torch version of `cvar`, differentiable in pnl through the tail samples."""
    k = max(1, math.ceil((1.0 - alpha) * pnl.numel()))
    return torch.topk(-pnl, k).values.mean()


def summarise(pnl: np.ndarray, alpha: float, lam: float) -> dict[str, float]:
    """Mean, standard deviation, CVaR_alpha and entropic risk at lam of a P&L sample."""
    x = np.asarray(pnl, dtype=float)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "cvar": cvar(x, alpha),
        "entropic": entropic(x, lam),
    }
