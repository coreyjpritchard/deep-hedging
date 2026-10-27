"""Block types for the graph spec: names, parameter schemas, wiring metadata.

`Registry.register(type_name, cls, param_schema)` maps a `type` string in a
spec node, such as `generator.heston`, to a class and the schema of its
parameters. `Registry.build(node)` checks the node's parameters, fills the
defaults and calls `cls(**params)`. Every block class has a class-level
`param_schema` and a `params()` method returning what it was built with.

A block class may also define a classmethod `describe(params, **context)`
returning what the validator needs to wire the graph without building the
block: a generator's instrument and state names and whether it has a
pricer; a feature block's output width and the names it reads; a hedger's
input and output widths; a cost model's per-instrument parameter lengths.
`describe` must not import torch or simulate anything.

This module holds stubs only: classes that carry the type name, the
parameter schema and `describe`, and nothing else. They let `validate` be
tested before any block is written. Later modules (M02 to M08, M11 to M14)
replace each stub by the real class under the same type name, keeping the
schema or extending it.
"""

from __future__ import annotations

import copy
import difflib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "REQUIRED",
    "Block",
    "Param",
    "Registry",
    "default_registry",
    "json_schema",
]


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()

_KINDS = ("float", "int", "bool", "str", "floats", "float_or_floats", "strs")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return (isinstance(value, int | float) and not isinstance(value, bool)) and math.isfinite(value)


@dataclass(frozen=True)
class Param:
    """One parameter: its kind, default and allowed range or choices.

    `kind` is one of `float`, `int`, `bool`, `str`, `floats` (a list of
    numbers), `float_or_floats` (a number, or one number per instrument) and
    `strs` (a list of strings). `default=REQUIRED` makes it mandatory;
    `default=None` makes it optional with no value. Bounds are inclusive
    unless `open_lower` or `open_upper` is set. `choices` restricts a string
    or each element of a string list.
    """

    kind: str
    default: Any = REQUIRED
    lower: float | None = None
    upper: float | None = None
    open_lower: bool = False
    open_upper: bool = False
    choices: tuple[str, ...] | None = None
    min_items: int = 0
    unique: bool = False
    doc: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown parameter kind {self.kind!r}; expected one of {_KINDS}")

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def _range(self, x: float) -> str | None:
        lo, hi = self.lower, self.upper
        if lo is not None and (x <= lo if self.open_lower else x < lo):
            return f"must be {'>' if self.open_lower else '>='} {lo}, got {x!r}"
        if hi is not None and (x >= hi if self.open_upper else x > hi):
            return f"must be {'<' if self.open_upper else '<='} {hi}, got {x!r}"
        return None

    def check(self, value: Any) -> str | None:
        """None when `value` is allowed, else a message."""
        if value is None and self.default is None:
            return None
        kind = self.kind
        if kind == "bool":
            return None if isinstance(value, bool) else f"must be true or false, got {value!r}"
        if kind == "int":
            if not _is_int(value):
                return f"must be an integer, got {value!r}"
            return self._range(value)
        if kind == "float":
            if not _is_number(value):
                return f"must be a finite number, got {value!r}"
            return self._range(value)
        if kind == "str":
            if not isinstance(value, str):
                return f"must be a string, got {value!r}"
            if self.choices is not None and value not in self.choices:
                return f"must be one of {', '.join(self.choices)}, got {value!r}"
            return None
        if kind == "float_or_floats" and _is_number(value):
            return self._range(value)
        if not isinstance(value, list | tuple):
            want = {"floats": "a list of numbers", "strs": "a list of strings"}.get(
                kind, "a number or a list of numbers"
            )
            return f"must be {want}, got {value!r}"
        if len(value) < max(self.min_items, 1 if kind == "float_or_floats" else 0):
            return f"needs at least {max(self.min_items, 1)} entries, got {len(value)}"
        if self.unique and len(set(value)) != len(value):
            return f"entries must be distinct, got {list(value)!r}"
        for i, item in enumerate(value):
            if kind == "strs":
                if not isinstance(item, str):
                    return f"entry {i} must be a string, got {item!r}"
                if self.choices is not None and item not in self.choices:
                    return f"entry {i} must be one of {', '.join(self.choices)}, got {item!r}"
            else:
                if not _is_number(item):
                    return f"entry {i} must be a finite number, got {item!r}"
                problem = self._range(item)
                if problem:
                    return f"entry {i} {problem}"
        return None

    def to_json_schema(self) -> dict[str, Any]:
        """JSON Schema (draft 2020-12) of the value. Bounds and choices included."""
        number: dict[str, Any] = {"type": "number"}
        if self.lower is not None:
            number["exclusiveMinimum" if self.open_lower else "minimum"] = self.lower
        if self.upper is not None:
            number["exclusiveMaximum" if self.open_upper else "maximum"] = self.upper
        string: dict[str, Any] = {"type": "string"}
        if self.choices is not None:
            string["enum"] = list(self.choices)
        if self.kind == "float":
            out = number
        elif self.kind == "int":
            out = {**number, "type": "integer"}
        elif self.kind == "bool":
            out = {"type": "boolean"}
        elif self.kind == "str":
            out = string
        elif self.kind == "floats":
            out = {"type": "array", "items": number, "minItems": self.min_items}
        elif self.kind == "strs":
            out = {"type": "array", "items": string, "minItems": self.min_items}
            if self.unique:
                out["uniqueItems"] = True
        else:
            out = {"oneOf": [number, {"type": "array", "items": number, "minItems": 1}]}
        if self.default is None:
            out = {"oneOf": [out, {"type": "null"}]}
        if self.doc:
            out = {**out, "description": self.doc}
        return out


class Block:
    """Base of every registered block. Keeps the parameters it was built with."""

    param_schema: ClassVar[dict[str, Param]] = {}

    def __init__(self, **params: Any) -> None:
        self._params = dict(params)

    def params(self) -> dict[str, Any]:
        return dict(self._params)

    @classmethod
    def describe(cls, params: Mapping[str, Any], **context: Any) -> dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"{type(self).__name__}({args})"


# Defaults filled in by `Registry.describe`, per family, so the validator can
# read every key without checking for it.
_DESCRIBE_DEFAULTS: dict[str, dict[str, Any]] = {
    "generator": {"instruments": ("spot",), "state": (), "has_pricer": False},
    "features": {"out_dim": 0, "requires": ()},
    "hedger": {"learned": False, "n_in": None, "n_out": None, "min_instruments": 1},
    "cost": {"per_instrument": {}},
}

RUN_SCHEMA: dict[str, Param] = {
    "m_train": Param("int", 2_000, lower=1, doc="training paths"),
    "m_validation": Param("int", 2_000, lower=1, doc="validation paths"),
    "m_test": Param("int", 2_000, lower=1, doc="test paths"),
    "n_steps": Param("int", 30, lower=1, doc="hedging dates from 0 to maturity"),
    "epochs": Param("int", 20, lower=0),
    "batch": Param("int", 2_000, lower=1),
    "lr": Param("float", 0.005, lower=0.0, open_lower=True),
    "mc_fallback": Param(
        "bool", False, doc="allow Monte Carlo greeks for classical hedgers without a pricer"
    ),
}


class Registry:
    """Type name -> (class, parameter schema). Add with `register`."""

    run_schema = RUN_SCHEMA

    def __init__(self) -> None:
        self._entries: dict[str, tuple[type, dict[str, Param]]] = {}

    def register(
        self, type_name: str, cls: type, param_schema: Mapping[str, Param] | None = None
    ) -> None:
        if "." not in type_name and type_name != "instruments":
            raise ValueError(f"type names are 'family.name', got {type_name!r}")
        schema = dict(getattr(cls, "param_schema", {}) if param_schema is None else param_schema)
        for key, param in schema.items():
            if not isinstance(param, Param):
                raise TypeError(f"{type_name}.{key}: schema entries must be Param")
        self._entries[type_name] = (cls, schema)

    def __contains__(self, type_name: object) -> bool:
        return type_name in self._entries

    def __iter__(self):
        return iter(sorted(self._entries))

    def types(self) -> list[str]:
        return sorted(self._entries)

    def near(self, type_name: str) -> str | None:
        found = difflib.get_close_matches(type_name, self._entries, n=1, cutoff=0.6)
        return found[0] if found else None

    def cls(self, type_name: str) -> type:
        return self._entries[type_name][0]

    def param_schema(self, type_name: str) -> dict[str, Param]:
        return dict(self._entries[type_name][1])

    @staticmethod
    def check_params(
        schema: Mapping[str, Param], params: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[tuple[str, str]]]:
        """Resolved parameters (defaults filled) and a list of (name, problem)."""
        problems: list[tuple[str, str]] = []
        resolved: dict[str, Any] = {}
        for key in sorted(set(params) - set(schema)):
            allowed = ", ".join(sorted(schema)) or "none"
            problems.append((key, f"unknown parameter; allowed: {allowed}"))
        for key, param in schema.items():
            if key not in params:
                if param.required:
                    problems.append((key, "missing required parameter"))
                else:
                    resolved[key] = copy.deepcopy(param.default)
                continue
            value = params[key]
            message = param.check(value)
            if message:
                problems.append((key, message))
            resolved[key] = list(value) if isinstance(value, tuple) else value
        return resolved, problems

    def build(self, node) -> Any:
        """The block for `node` (a `spec.Node`), after checking its parameters."""
        if node.type not in self._entries:
            near = self.near(node.type)
            hint = f"; did you mean {near!r}?" if near else ""
            raise KeyError(f"unknown block type {node.type!r}{hint}")
        cls, schema = self._entries[node.type]
        resolved, problems = self.check_params(schema, node.params)
        if problems:
            lines = "; ".join(f"{k}: {m}" for k, m in problems)
            raise ValueError(f"{node.type}: {lines}")
        return cls(**resolved)

    def describe(self, type_name: str, params: Mapping[str, Any], **context: Any) -> dict:
        cls = self.cls(type_name)
        family = type_name.split(".", 1)[0]
        out = dict(_DESCRIBE_DEFAULTS.get(family, {}))
        describe = getattr(cls, "describe", None)
        if describe is not None:
            out.update(describe(params, **context))
        return out

    def json_schema(self) -> dict[str, Any]:
        """JSON Schema of a `deephedge.graph/v1` file for the registered types.

        Structural only: types, ranges and required fields. Wiring, widths
        and seed distinctness are checked by `deephedge.spec.validate`.
        """
        node_variants = []
        for name in self.types():
            schema = self.param_schema(name)
            params = {
                "type": "object",
                "properties": {k: p.to_json_schema() for k, p in sorted(schema.items())},
                "additionalProperties": False,
            }
            required = sorted(k for k, p in schema.items() if p.required)
            if required:
                params["required"] = required
            variant: dict[str, Any] = {
                "type": "object",
                "properties": {"type": {"const": name}, "params": params},
                "required": ["type"] + (["params"] if required else []),
                "additionalProperties": False,
            }
            node_variants.append(variant)
        seed = {"type": "integer", "minimum": 0}
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "deephedge.graph/v1",
            "title": "deep-hedging graph spec, deephedge.graph/v1",
            "description": (
                "Generated by `python -m deephedge.registry`. Structural checks only; "
                "deephedge.spec.validate checks wiring, widths and seed distinctness."
            ),
            "type": "object",
            "properties": {
                "schema": {"const": "deephedge.graph/v1"},
                "id": {"type": "string", "pattern": "^[0-9a-f]{16}$"},
                "seeds": {
                    "type": "object",
                    "properties": {k: seed for k in ("train", "validation", "test", "torch")},
                    "required": ["train", "validation", "test", "torch"],
                    "additionalProperties": False,
                },
                "nodes": {"type": "object", "additionalProperties": {"oneOf": node_variants}},
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "run": {
                    "type": "object",
                    "properties": {
                        k: p.to_json_schema() for k, p in sorted(self.run_schema.items())
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["schema", "seeds", "nodes", "edges"],
            "additionalProperties": False,
        }


def _num(default, lower=None, upper=None, *, open_lower=False, open_upper=False, doc=""):
    return Param(
        "float",
        default,
        lower=lower,
        upper=upper,
        open_lower=open_lower,
        open_upper=open_upper,
        doc=doc,
    )


_POSITIVE = {"lower": 0.0, "open_lower": True}

_MARKET = {
    "s0": _num(100.0, **_POSITIVE, doc="spot at t = 0"),
    "maturity": _num(30 / 365, **_POSITIVE, doc="years"),
    "r": _num(0.0, doc="risk-free rate, continuously compounded"),
    "mu": _num(0.0, doc="drift of the simulation measure"),
}


# ---- generators ---------------------------------------------------------------


class BlackScholesStub(Block):
    """`generator.black_scholes`: spot only, closed-form pricer (`deephedge.generators.bs`)."""

    param_schema = {**_MARKET, "sigma": _num(0.2, **_POSITIVE)}

    @classmethod
    def describe(cls, params, **context):
        return {"instruments": ("spot",), "state": (), "has_pricer": True}


class HestonStub(Block):
    """`generator.heston`: spot and variance swap, COS pricer (`deephedge.generators.heston`)."""

    param_schema = {
        **_MARKET,
        "kappa": _num(1.0, **_POSITIVE),
        "theta": _num(0.04, 0.0),
        "xi": _num(2.0, 0.0, doc="vol of vol; 0 is the Black-Scholes limit"),
        "rho": _num(-0.7, -1.0, 1.0),
        "v0": _num(0.04, 0.0),
        "kappa_sim": _num(None, **_POSITIVE, doc="simulation kappa; unset means kappa"),
        "theta_sim": _num(None, 0.0, doc="simulation theta; unset means theta"),
        "n_cos": Param("int", 512, lower=16, doc="COS terms along a path"),
        "truncation": _num(20.0, **_POSITIVE, doc="COS range in standard deviations"),
    }

    @classmethod
    def describe(cls, params, **context):
        return {
            "instruments": ("spot", "variance_swap"),
            "state": ("variance",),
            "has_pricer": True,
        }


class RoughBergomiStub(Block):
    """`generator.rough_bergomi`: adapter over `roughvol.rbergomi.RoughBergomi` (M11a).

    Replaced at registry build time by `deephedge.generators.rbergomi.RoughBergomi`
    when the `[rough]` extra is installed; this stub keeps `validate` working
    without it. `H` defaults to 0.1 (D5). `nu` is the annual vol of vol; a
    per-day estimate converts with `roughvol.estimate.rbergomi_nu`. Prices are
    conditional Monte Carlo with a standard error; the classical benchmark
    greek is the Black-Scholes delta at `sqrt(v_t)`, so `has_pricer` is true.
    """

    param_schema = {
        **_MARKET,
        "H": _num(0.1, 0.0, 1.0, open_lower=True, open_upper=True, doc="Hurst exponent"),
        "nu": _num(1.5, 0.0, doc="annual vol of vol"),
        "rho": _num(-0.7, -1.0, 0.0),
        "xi0": _num(0.04, **_POSITIVE, doc="flat initial forward variance"),
        "xi_tenors": Param("floats", None, lower=0.0, open_lower=True),
        "n_price_paths": Param("int", 4_000, lower=2, doc="paths for price_with_error"),
        "price_seed": Param("int", 0, doc="seed for price_with_error"),
    }

    def __new__(cls, **params):
        try:
            from deephedge.generators.rbergomi import RoughBergomi
        except ImportError:
            return super().__new__(cls)
        return RoughBergomi(**params)

    @classmethod
    def describe(cls, params, **context):
        del params, context
        return {
            "instruments": ("spot", "variance_swap"),
            "state": ("variance", "xi"),
            "has_pricer": True,
        }


# ---- contract, instruments, cost ------------------------------------------------


class VanillaStub(Block):
    """`contract.vanilla`: European call or put on the spot, read at maturity."""

    param_schema = {
        "kind": Param("str", "call", choices=("call", "put")),
        "strike": _num(100.0, **_POSITIVE),
    }


class InstrumentsStub(Block):
    """`instruments`: the generator's instruments that the hedgers trade. Spot first."""

    param_schema = {"names": Param("strs", REQUIRED, min_items=1, unique=True)}


def _length(value) -> int | None:
    return len(value) if isinstance(value, list | tuple) else None


class ProportionalStub(Block):
    """`cost.proportional`: `rate * |trade| * price` (`deephedge.costs`)."""

    param_schema = {"rate": Param("float_or_floats", 0.0, lower=0.0)}

    @classmethod
    def describe(cls, params, **context):
        return {"per_instrument": {"rate": _length(params["rate"])}}


class FixedPlusProportionalStub(Block):
    """`cost.fixed_plus_proportional`: cash per trade plus a proportional charge."""

    param_schema = {
        "fixed": Param("float_or_floats", 0.0, lower=0.0),
        "rate": Param("float_or_floats", 0.0, lower=0.0),
        "width": _num(1e-2, **_POSITIVE, doc="tanh surrogate width used in training"),
    }

    @classmethod
    def describe(cls, params, **context):
        return {
            "per_instrument": {
                "fixed": _length(params["fixed"]),
                "rate": _length(params["rate"]),
            }
        }


class QuadraticStub(Block):
    """`cost.quadratic`: `impact * trade^2 * price`."""

    param_schema = {"impact": Param("float_or_floats", 0.0, lower=0.0)}

    @classmethod
    def describe(cls, params, **context):
        return {"per_instrument": {"impact": _length(params["impact"])}}


# ---- features ----------------------------------------------------------------------

# name -> (width, names read). Width "d" means one column per traded instrument.
FEATURE_BLOCKS: dict[str, tuple[int | str, tuple[str, ...]]] = {
    "time": (1, ()),
    "spot": (1, ("spot",)),
    "log_moneyness": (1, ("spot",)),
    "variance": (1, ("variance",)),
    "prev_position": ("d", ()),
}


class ConcatStub(Block):
    """`features.concat`: the named feature blocks side by side (M06 fills them in)."""

    param_schema = {
        "blocks": Param("strs", REQUIRED, choices=tuple(FEATURE_BLOCKS), min_items=1, unique=True)
    }

    @classmethod
    def describe(cls, params, *, n_instruments: int = 1, **context):
        width = 0
        requires: list[str] = []
        for name in params["blocks"]:
            w, reads = FEATURE_BLOCKS[name]
            width += n_instruments if w == "d" else w
            requires += [r for r in reads if r not in requires]
        return {"out_dim": width, "requires": tuple(requires)}


# ---- hedgers -----------------------------------------------------------------------


class DeltaStub(Block):
    """`hedger.delta`: the pricer's delta in the spot, nothing in other instruments."""


class DeltaVegaStub(Block):
    """`hedger.delta_vega`: delta and vega neutral with the spot and a second instrument."""

    @classmethod
    def describe(cls, params, **context):
        return {"min_instruments": 2}


class WhalleyWilmottStub(Block):
    """`hedger.whalley_wilmott`: the asymptotic no-trade band around delta."""

    param_schema = {
        "risk_aversion": _num(1.0, **_POSITIVE, doc="gamma in the band formula"),
    }


_WIDTHS = {
    "n_in": Param("int", None, lower=1, doc="checked against the features width"),
    "n_out": Param("int", None, lower=1, doc="checked against the instrument count"),
}


class FeatureMLPStub(Block):
    """`hedger.feature_mlp`: a feed-forward network on the features (M05)."""

    param_schema = {
        "width": Param("int", 64, lower=1),
        "depth": Param("int", 3, lower=1),
        "activation": Param("str", "relu", choices=("relu", "tanh", "silu")),
        **_WIDTHS,
    }

    @classmethod
    def describe(cls, params, **context):
        return {"learned": True, "n_in": params["n_in"], "n_out": params["n_out"]}


class GRUStub(Block):
    """`hedger.gru`: a recurrent policy carrying a hidden state (M05, from PR 4)."""

    param_schema = {"hidden": Param("int", 32, lower=1), **_WIDTHS}

    @classmethod
    def describe(cls, params, **context):
        return {"learned": True, "n_in": params["n_in"], "n_out": params["n_out"]}


# ---- objective, evaluator ----------------------------------------------------------


class EntropicStub(Block):
    """`objective.entropic`: `(1/lam) log E exp(-lam X)`."""

    param_schema = {"lam": _num(1.0, **_POSITIVE)}


class CVaRStub(Block):
    """`objective.cvar`: expected shortfall of the loss at level `alpha`."""

    param_schema = {"alpha": _num(0.5, 0.0, 1.0, open_lower=True, open_upper=True)}


class MeanVarianceStub(Block):
    """`objective.mean_variance`: `-E X + lam Var X`. Not a convex risk measure."""

    param_schema = {"lam": _num(1.0, **_POSITIVE)}


REPORT_STATS = ("mean", "std", "cvar", "entropic", "mean_variance", "turnover", "cost")


class CompareStub(Block):
    """`evaluator.compare`: hedgers side by side on the test split (M10)."""

    param_schema = {
        "bootstrap": Param("int", 2_000, lower=0, doc="bootstrap resamples, 0 for none"),
        "level": _num(0.95, 0.0, 1.0, open_lower=True, open_upper=True),
        "report": Param(
            "strs", ["mean", "entropic"], choices=REPORT_STATS, min_items=1, unique=True
        ),
    }


STUBS: dict[str, type[Block]] = {
    "generator.black_scholes": BlackScholesStub,
    "generator.heston": HestonStub,
    "generator.rough_bergomi": RoughBergomiStub,
    "contract.vanilla": VanillaStub,
    "instruments": InstrumentsStub,
    "cost.proportional": ProportionalStub,
    "cost.fixed_plus_proportional": FixedPlusProportionalStub,
    "cost.quadratic": QuadraticStub,
    "features.concat": ConcatStub,
    "hedger.delta": DeltaStub,
    "hedger.delta_vega": DeltaVegaStub,
    "hedger.whalley_wilmott": WhalleyWilmottStub,
    "hedger.feature_mlp": FeatureMLPStub,
    "hedger.gru": GRUStub,
    "objective.entropic": EntropicStub,
    "objective.cvar": CVaRStub,
    "objective.mean_variance": MeanVarianceStub,
    "evaluator.compare": CompareStub,
}


def default_registry() -> Registry:
    """A fresh registry with every type known so far.

    Stubs only: `validate` must not import torch or a simulator. Building
    `generator.rough_bergomi` returns the real adapter when `[rough]` is
    installed (see `RoughBergomiStub.__new__`).
    """
    registry = Registry()
    for name, cls in STUBS.items():
        registry.register(name, cls)
    return registry


def json_schema() -> dict[str, Any]:
    return default_registry().json_schema()


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "docs" / "schema" / "graph-v1.json"


def schema_text() -> str:
    return json.dumps(json_schema(), indent=2) + "\n"


if __name__ == "__main__":
    SCHEMA_PATH.write_text(schema_text())
    print(f"wrote {SCHEMA_PATH}")
