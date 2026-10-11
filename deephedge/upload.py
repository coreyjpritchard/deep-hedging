"""Load a Python file that defines a `Market`, and check it.

`load_uploaded_market` executes the file. There is no sandbox. The
playground is for local use, with a file the user trusts. The static
web page does not run uploads.

The file defines `build_market()`, which returns a `Market`, or it
defines exactly one `Market` subclass and that class is constructed with
no arguments. Classes imported from `deephedge` do not count: only a
class whose body is in the file does.

`validate_market` checks shapes and a martingale condition. On `m`
paths, the mean of the discounted terminal spot should lie within
`se_tol` standard errors of `s0`. A correct martingale fails that test
about once in 16 000 tries at `se_tol` 4. A simulation drift different
from the rate used in the discount fails it on purpose. Heston Euler is
biased at short maturity, so this check is for an uploaded model, not
for the built-in Heston card.
"""

from __future__ import annotations

import math
import types

import numpy as np

from deephedge.env import Market, Paths

__all__ = ["MarketUploadError", "decode_text", "load_uploaded_market", "validate_market"]


class MarketUploadError(Exception):
    """The uploaded file did not produce a usable market."""


def decode_text(contents: bytes | str, filename: str) -> str:
    """UTF-8 text of an upload. A string is returned unchanged."""
    if isinstance(contents, str):
        return contents
    try:
        return bytes(contents).decode("utf-8-sig")
    except UnicodeError as exc:
        raise MarketUploadError(f"{filename} is not UTF-8 text.") from exc


def load_uploaded_market(source: str, filename: str = "upload.py") -> Market:
    """Execute `source` and return the `Market` it builds.

    This runs whatever the file contains. Call it only on a local file
    you trust.
    """
    if not source or not source.strip():
        raise MarketUploadError(f"{filename} is empty.")
    module = types.ModuleType("uploaded_market")
    module.__dict__["__file__"] = filename
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as exc:
        raise MarketUploadError(
            f"{filename} is not valid Python: line {exc.lineno}: {exc.msg}."
        ) from exc
    try:
        exec(code, module.__dict__)  # noqa: S102
    except Exception as exc:
        raise MarketUploadError(
            f"{filename} failed while running: {type(exc).__name__}: {exc}"
        ) from exc
    if callable(module.__dict__.get("build_market")):
        try:
            market = module.build_market()
        except Exception as exc:
            raise MarketUploadError(
                f"build_market() in {filename} failed: {type(exc).__name__}: {exc}"
            ) from exc
    else:
        market = _only_local_market(module, filename)
    if isinstance(market, type):
        raise MarketUploadError(
            "build_market() returned a class. Return an instance, as the template does."
        )
    if not isinstance(market, Market):
        raise MarketUploadError(
            f"{filename} produced {type(market).__name__}, which is not a Market. "
            "Return an instance of a Market subclass from build_market()."
        )
    return market


def _only_local_market(module: types.ModuleType, filename: str) -> Market:
    found: list[type] = []
    for obj in vars(module).values():
        if not isinstance(obj, type) or obj is Market:
            continue
        try:
            local = issubclass(obj, Market) and obj.__module__ == module.__name__
        except TypeError:
            local = False
        if local:
            found.append(obj)
    if len(found) != 1:
        names = ", ".join(cls.__name__ for cls in found) or "none"
        raise MarketUploadError(
            f"{filename} needs a build_market() function, or exactly one Market "
            f"subclass defined in the file (found {names}). The Merton template defines both."
        )
    try:
        return found[0]()
    except Exception as exc:
        raise MarketUploadError(
            f"Could not construct {found[0].__name__}() with no arguments: "
            f"{type(exc).__name__}: {exc}. Define build_market() if the class "
            "needs arguments."
        ) from exc


def validate_market(
    market: Market,
    m: int = 1024,
    n: int = 8,
    seed: int = 0,
    se_tol: float = 4.0,
) -> str:
    """Check shapes and the discounted-spot martingale. Return a short note.

    Raises `MarketUploadError` with the numbers when a check fails.
    """
    if m < 2 or n < 1:
        raise MarketUploadError("The check needs at least two paths and one step.")
    try:
        paths = market.simulate(m, n, seed)
    except MarketUploadError:
        raise
    except NotImplementedError as exc:
        raise MarketUploadError(str(exc)) from exc
    except Exception as exc:
        raise MarketUploadError(
            f"simulate({m}, {n}, seed={seed}) failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(paths, Paths):
        raise MarketUploadError(
            f"simulate returned {type(paths).__name__}, not a Paths batch."
        )
    _check_shapes(market, paths, m, n)
    payoff = _call("payoff", lambda: market.payoff(paths))
    _check_array("payoff", payoff, (m,))
    price = _call("price_along", lambda: market.price_along(paths))
    _check_array("price_along", price, (m, n + 1))
    bench = _call("benchmark_positions", lambda: market.benchmark_positions(paths))
    _check_array("benchmark_positions", bench, (m, n, paths.n_instruments))
    premium = _call("premium", market.premium)
    premium_arr = np.asarray(premium, dtype=float)
    if premium_arr.shape != ():
        raise MarketUploadError(
            f"premium should be one number, got shape {premium_arr.shape}."
        )
    if not np.isfinite(premium_arr):
        raise MarketUploadError(f"premium is {premium_arr}.")
    mean, se, s0 = _martingale(market, paths)
    gap = abs(mean - s0)
    if se == 0.0:
        ok = gap <= 1e-8 * max(1.0, abs(s0))
        ratio = math.inf if not ok else 0.0
    else:
        ratio = gap / se
        ok = gap <= se_tol * se
    if not ok:
        raise MarketUploadError(
            "Martingale check failed. The mean of the discounted terminal spot "
            f"is {mean:.6g}, s0 is {s0:.6g}, and the gap is {gap:.6g}. "
            f"That is {ratio:.2f} standard errors (SE {se:.6g}, m={m}, "
            f"tolerance {se_tol:g}). Under the pricing measure the discounted "
            "spot averages to s0. A simulation drift different from r fails "
            "this on purpose. Set the drift equal to r, or use a larger sample "
            "if the gap is noise."
        )
    return (
        f"Market accepted. instruments {tuple(paths.instruments.shape)}, "
        f"features {tuple(paths.features.shape)}, "
        f"benchmark {tuple(np.shape(bench))}, premium {float(premium_arr):.6g}. "
        f"Discounted terminal spot averages to {mean:.6g} against s0 {s0:.6g} "
        f"({ratio:.2f} SE, m={m})."
    )


def _call(name: str, fn):
    try:
        return fn()
    except MarketUploadError:
        raise
    except Exception as exc:
        raise MarketUploadError(f"{name} failed: {type(exc).__name__}: {exc}") from exc


def _check_shapes(market: Market, paths: Paths, m: int, n: int) -> None:
    try:
        n_instruments = int(market.n_instruments)
        n_features = int(market.n_features)
        instrument_names = tuple(market.instrument_names)
        feature_names = tuple(market.feature_names)
    except Exception as exc:
        raise MarketUploadError(
            f"Could not read instrument or feature names: {type(exc).__name__}: {exc}"
        ) from exc
    if paths.instruments.shape != (m, n + 1, n_instruments):
        raise MarketUploadError(
            f"instruments have shape {paths.instruments.shape}, "
            f"expected {(m, n + 1, n_instruments)}."
        )
    if paths.features.shape != (m, n + 1, n_features):
        raise MarketUploadError(
            f"features have shape {paths.features.shape}, "
            f"expected {(m, n + 1, n_features)}."
        )
    if len(instrument_names) != n_instruments:
        raise MarketUploadError(
            f"instrument_names has length {len(instrument_names)}, "
            f"n_instruments is {n_instruments}."
        )
    if len(feature_names) != n_features:
        raise MarketUploadError(
            f"feature_names has length {len(feature_names)}, n_features is {n_features}."
        )
    if paths.spot.shape != (m, n + 1):
        raise MarketUploadError(f"spot has shape {paths.spot.shape}, expected {(m, n + 1)}.")
    if not np.isfinite(paths.instruments).all():
        raise MarketUploadError("instruments contain a non-finite value.")
    if not np.isfinite(paths.features).all():
        raise MarketUploadError("features contain a non-finite value.")


def _check_array(name: str, value, shape: tuple[int, ...]) -> None:
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise MarketUploadError(f"{name} has shape {arr.shape}, expected {shape}.")
    if not np.isfinite(arr).all():
        raise MarketUploadError(f"{name} contains a non-finite value.")


def _martingale(market: Market, paths: Paths) -> tuple[float, float, float]:
    if hasattr(market, "s0"):
        s0 = float(market.s0)
    else:
        s0 = float(paths.spot[0, 0])
    terminal = np.asarray(paths.spot[:, -1] * paths.discount_terminal(), dtype=float)
    if terminal.shape != (paths.spot.shape[0],):
        raise MarketUploadError(
            f"discounted terminal spot has shape {terminal.shape}, "
            f"expected {(paths.spot.shape[0],)}."
        )
    if not np.isfinite(terminal).all():
        raise MarketUploadError("The discounted terminal spot is not finite.")
    mean = float(terminal.mean())
    se = float(terminal.std(ddof=1) / math.sqrt(terminal.size))
    return mean, se, s0
