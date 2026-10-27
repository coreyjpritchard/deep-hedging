"""Generators: simulators that draw `Paths`, each with an optional closed-form pricer.

See `deephedge.generators.base` for the `Generator` and `Pricer` protocols.
"""

from deephedge.generators.base import Generator, Market, Pricer
from deephedge.generators.bs import BlackScholes, BlackScholesPricer
from deephedge.generators.heston import Heston, HestonPricer

__all__ = [
    "BlackScholes",
    "BlackScholesPricer",
    "Generator",
    "Heston",
    "HestonPricer",
    "Market",
    "Pricer",
]

try:
    from deephedge.generators.rbergomi import RoughBergomi, RoughBergomiPricer, rbergomi_nu

    __all__ += ["RoughBergomi", "RoughBergomiPricer", "rbergomi_nu"]
except ImportError:
    pass
