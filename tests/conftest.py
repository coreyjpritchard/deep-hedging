"""Shared fixtures. `--impl exercises` runs the mathematical tests against Corey's code."""

from __future__ import annotations

import importlib

import pytest

MODULES = ("env", "markets", "hedging", "risk", "policy", "train")


def pytest_addoption(parser):
    parser.addoption(
        "--impl",
        default="deephedge",
        help="package whose functions the tests check: deephedge (default) or exercises",
    )


class Impl:
    """Looks a function up in <package>.env, .markets, .hedging, .risk, .policy, .train.

    A function missing from an exercises package skips the test instead of failing it,
    so Corey can implement one function at a time.
    """

    def __init__(self, package: str):
        self.package = package
        self.modules = []
        for name in MODULES:
            try:
                self.modules.append(importlib.import_module(f"{package}.{name}"))
            except ModuleNotFoundError:
                continue

    def __getattr__(self, name: str):
        for module in self.modules:
            if hasattr(module, name):
                return getattr(module, name)
        if self.package == "deephedge":
            raise AttributeError(name)
        pytest.skip(f"{self.package} does not implement {name}")


@pytest.fixture(scope="session")
def impl(request) -> Impl:
    return Impl(request.config.getoption("--impl"))
