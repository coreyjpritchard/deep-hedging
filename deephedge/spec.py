"""The graph spec, `deephedge.graph/v1`: one JSON file describes one run.

A spec is a set of named nodes (a market generator, a contract, an
instrument set, a cost model, feature blocks, hedgers, an objective and an
evaluator), the edges between them, the seeds and the run sizes. The UI,
a script and the experiment runner all write this file; the engine only
reads it. The layout is `docs/plan/01-interfaces.md`, section "Graph spec".

`GraphSpec.from_json` and `GraphSpec.to_json` round trip. `spec_id` is the
first 16 hex digits of the SHA-256 of the canonical JSON without the `id`
field: keys sorted, edges sorted, no whitespace. Two specs that differ only
in key order or edge order have the same id. `1` and `1.0` are different
JSON values and give different ids.

`validate(spec)` returns a list of `Problem`. An empty list means the spec
is valid. It reads the registry's parameter schemas and the blocks'
`describe` metadata. It builds nothing, simulates nothing and does not
import torch, so it can run in a UI before any work starts.

Checks, in order: the schema string; the stored id against the content;
seeds (train, validation, test and torch present, non-negative integers,
and train, validation and test pairwise distinct); run sizes; every node
type known and every parameter of the right type and range; every edge
between known nodes and of an allowed kind; one node of each single-use
family; the instrument set a subset of the generator's instruments; the
state each feature block reads provided by the generator; the feature
width equal to each learned hedger's input width; the instrument count
equal to each hedger's output width and to the length of per-instrument
cost parameters; a pricer for each classical hedger, unless the Monte Carlo
fallback is switched on in `run`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

__all__ = [
    "SCHEMA",
    "GraphSpec",
    "Node",
    "Problem",
    "canonical_json",
    "validate",
]

SCHEMA = "deephedge.graph/v1"
ID_LENGTH = 16
SEED_FIELDS = ("train", "validation", "test", "torch")
SPLIT_SEEDS = ("train", "validation", "test")


def _freeze(value: Any) -> Any:
    """Read-only copy: dicts become mapping proxies, lists become tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Plain JSON types back from `_freeze`."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw(v) for v in value]
    return value


def canonical_json(data: Mapping[str, Any]) -> str:
    """Sorted keys, no whitespace, ASCII. The input of `spec_id`."""
    return json.dumps(
        _thaw(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


@dataclass(frozen=True)
class Node:
    """One block: a registered `type` name and its parameters as written."""

    type: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(f"node type must be a non-empty string, got {self.type!r}")
        if not isinstance(self.params, Mapping):
            raise ValueError(f"node params must be an object, got {type(self.params).__name__}")
        object.__setattr__(self, "params", _freeze(self.params))

    @property
    def family(self) -> str:
        """The part of `type` before the first dot: `generator`, `hedger`, ..."""
        return self.type.split(".", 1)[0]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.params:
            out["params"] = _thaw(self.params)
        return out


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a spec. `where` is a dotted path into the JSON."""

    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


@dataclass(frozen=True)
class GraphSpec:
    """A parsed graph spec. Immutable; `to_json` writes the canonical id.

    `declared_id` is the `id` read from the file, if any. It is not part of
    the content and `validate` reports it when it differs from `spec_id`.
    """

    nodes: Mapping[str, Node]
    edges: tuple[tuple[str, str], ...] = ()
    seeds: Mapping[str, Any] = field(default_factory=dict)
    run: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA
    declared_id: str | None = None

    def __post_init__(self) -> None:
        nodes = {}
        for name, node in self.nodes.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"node names must be non-empty strings, got {name!r}")
            nodes[name] = node if isinstance(node, Node) else Node(**dict(node))
        object.__setattr__(self, "nodes", MappingProxyType(nodes))
        edges = []
        for edge in self.edges:
            if (
                not isinstance(edge, list | tuple)
                or len(edge) != 2
                or not all(isinstance(e, str) for e in edge)
            ):
                raise ValueError(f"an edge is a pair of node names, got {edge!r}")
            edges.append((edge[0], edge[1]))
        object.__setattr__(self, "edges", tuple(edges))
        object.__setattr__(self, "seeds", _freeze(dict(self.seeds)))
        object.__setattr__(self, "run", _freeze(dict(self.run)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GraphSpec:
        if not isinstance(data, Mapping):
            raise ValueError(f"a graph spec is a JSON object, got {type(data).__name__}")
        known = {"schema", "id", "seeds", "nodes", "edges", "run"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown top-level fields: {', '.join(unknown)}")
        raw_nodes = data.get("nodes", {})
        if not isinstance(raw_nodes, Mapping):
            raise ValueError("nodes must be an object of name -> {type, params}")
        nodes = {}
        for name, raw in raw_nodes.items():
            if not isinstance(raw, Mapping) or "type" not in raw:
                raise ValueError(f"node {name!r} must be an object with a 'type'")
            extra = sorted(set(raw) - {"type", "params"})
            if extra:
                raise ValueError(f"node {name!r} has unknown fields: {', '.join(extra)}")
            nodes[name] = Node(raw["type"], raw.get("params") or {})
        declared = data.get("id")
        return cls(
            nodes=nodes,
            edges=tuple(tuple(e) if isinstance(e, list) else e for e in data.get("edges", ())),
            seeds=data.get("seeds", {}),
            run=data.get("run", {}),
            schema=data.get("schema", ""),
            declared_id=None if declared is None else str(declared),
        )

    @classmethod
    def from_json(cls, text: str) -> GraphSpec:
        return cls.from_dict(json.loads(text))

    def content(self) -> dict[str, Any]:
        """The spec as plain JSON types, without `id`. Edges sorted."""
        return {
            "schema": self.schema,
            "seeds": _thaw(self.seeds),
            "nodes": {name: node.to_dict() for name, node in sorted(self.nodes.items())},
            "edges": [list(e) for e in sorted(self.edges)],
            "run": _thaw(self.run),
        }

    @property
    def spec_id(self) -> str:
        digest = hashlib.sha256(canonical_json(self.content()).encode("ascii")).hexdigest()
        return digest[:ID_LENGTH]

    def to_dict(self) -> dict[str, Any]:
        content = self.content()
        return {"schema": content.pop("schema"), "id": self.spec_id, **content}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True, allow_nan=False)

    def nodes_of(self, family: str) -> list[str]:
        """Names of the nodes whose type starts with `family.` (or equals it)."""
        return sorted(n for n, node in self.nodes.items() if node.family == family)

    def inputs(self, name: str) -> list[str]:
        """Nodes with an edge into `name`."""
        return sorted(src for src, dst in self.edges if dst == name)

    def outputs(self, name: str) -> list[str]:
        """Nodes `name` has an edge into."""
        return sorted(dst for src, dst in self.edges if src == name)


def validate(spec: GraphSpec, registry=None) -> list[Problem]:
    """Every problem with `spec`, or an empty list. Builds and simulates nothing."""
    from deephedge.registry import default_registry

    return _Validator(spec, registry or default_registry()).run()


# Allowed (source family, target family) pairs. Everything else is reported.
EDGE_KINDS = frozenset(
    {
        ("generator", "instruments"),
        ("generator", "features"),
        ("instruments", "cost"),
        ("instruments", "hedger"),
        ("features", "hedger"),
        ("contract", "objective"),
        ("contract", "hedger"),
        ("objective", "hedger"),
        ("objective", "evaluator"),
        ("hedger", "evaluator"),
        ("cost", "evaluator"),
        ("contract", "evaluator"),
    }
)

# Exactly one node of each of these families per spec. Cross-environment runs
# are several specs, not several generators in one spec.
SINGLE = ("generator", "contract", "instruments", "cost", "objective", "evaluator")


class _Validator:
    def __init__(self, spec: GraphSpec, registry) -> None:
        self.spec = spec
        self.registry = registry
        self.problems: list[Problem] = []
        self.resolved: dict[str, dict[str, Any]] = {}

    def add(self, where: str, message: str) -> None:
        self.problems.append(Problem(where, message))

    def run(self) -> list[Problem]:
        self.check_header()
        self.check_seeds()
        self.check_run()
        self.check_nodes()
        edges_ok = self.check_edges()
        counts_ok = self.check_counts()
        if edges_ok and counts_ok:
            self.check_wiring()
        return self.problems

    def check_header(self) -> None:
        spec = self.spec
        if spec.schema != SCHEMA:
            self.add("schema", f"expected {SCHEMA!r}, got {spec.schema!r}")
        if spec.declared_id is not None and spec.declared_id != spec.spec_id:
            self.add(
                "id",
                f"stored id {spec.declared_id!r} does not match the content, "
                f"whose id is {spec.spec_id!r}",
            )

    def check_seeds(self) -> None:
        seeds = self.spec.seeds
        for key in sorted(set(seeds) - set(SEED_FIELDS)):
            self.add(f"seeds.{key}", f"unknown seed; expected one of {', '.join(SEED_FIELDS)}")
        for key in SEED_FIELDS:
            if key not in seeds:
                self.add(f"seeds.{key}", "missing seed")
                continue
            value = seeds[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                self.add(f"seeds.{key}", f"must be a non-negative integer, got {value!r}")
        for i, a in enumerate(SPLIT_SEEDS):
            for b in SPLIT_SEEDS[i + 1 :]:
                if a in seeds and b in seeds and seeds[a] == seeds[b]:
                    self.add(
                        f"seeds.{a}",
                        f"seeds.{a} and seeds.{b} are both {seeds[a]!r}; train, validation "
                        "and test must use different paths",
                    )

    def check_run(self) -> None:
        resolved, problems = self.registry.check_params(self.registry.run_schema, self.spec.run)
        for key, message in problems:
            self.add(f"run.{key}", message)
        self.resolved["__run__"] = resolved

    def check_nodes(self) -> None:
        for name, node in sorted(self.spec.nodes.items()):
            if node.type not in self.registry:
                near = self.registry.near(node.type)
                hint = f"; did you mean {near!r}?" if near else ""
                self.add(f"nodes.{name}.type", f"unknown block type {node.type!r}{hint}")
                continue
            schema = self.registry.param_schema(node.type)
            resolved, problems = self.registry.check_params(schema, node.params)
            for key, message in problems:
                self.add(f"nodes.{name}.params.{key}", message)
            if not problems:
                self.resolved[name] = resolved

    def check_edges(self) -> bool:
        ok = True
        seen = set()
        nodes = self.spec.nodes
        for i, (src, dst) in enumerate(self.spec.edges):
            where = f"edges[{i}]"
            for end in (src, dst):
                if end not in nodes:
                    self.add(where, f"unknown node {end!r} in edge [{src!r}, {dst!r}]")
                    ok = False
            if src not in nodes or dst not in nodes:
                continue
            if (src, dst) in seen:
                self.add(where, f"duplicate edge [{src!r}, {dst!r}]")
            seen.add((src, dst))
            kind = (nodes[src].family, nodes[dst].family)
            if kind not in EDGE_KINDS:
                self.add(where, f"an edge from a {kind[0]} to a {kind[1]} is not allowed")
                ok = False
        return ok

    def check_counts(self) -> bool:
        ok = True
        for family in SINGLE:
            found = self.spec.nodes_of(family)
            if len(found) != 1:
                what = "none" if not found else ", ".join(found)
                self.add("nodes", f"needs exactly one {family} node, found {what}")
                ok = False
        if not self.spec.nodes_of("hedger"):
            self.add("nodes", "needs at least one hedger node")
            ok = False
        known = set(SINGLE) | {"hedger", "features"}
        for name, node in sorted(self.spec.nodes.items()):
            if node.family not in known:
                self.add(f"nodes.{name}.type", f"unknown block family {node.family!r}")
                ok = False
        return ok

    def describe(self, name: str, **context) -> dict[str, Any] | None:
        """The block's metadata, or None when its parameters already failed."""
        if name not in self.resolved:
            return None
        node = self.spec.nodes[name]
        return self.registry.describe(node.type, self.resolved[name], **context)

    def check_wiring(self) -> None:
        spec = self.spec
        (gen,) = spec.nodes_of("generator")
        (ins,) = spec.nodes_of("instruments")
        (cost,) = spec.nodes_of("cost")
        run = self.resolved.get("__run__", {})
        g = self.describe(gen)
        if g is None:
            return

        for target, source in ((ins, gen), (cost, ins)):
            if source not in spec.inputs(target):
                self.add(f"nodes.{target}", f"needs an edge from {source!r}")

        names = list(self.resolved.get(ins, {}).get("names", ()))
        missing = [n for n in names if n not in g["instruments"]]
        if missing:
            self.add(
                f"nodes.{ins}.params.names",
                f"{', '.join(missing)} not offered by {gen!r} ({spec.nodes[gen].type}), "
                f"which offers {', '.join(g['instruments'])}",
            )
        if names and names[0] != "spot":
            self.add(f"nodes.{ins}.params.names", "the first instrument must be 'spot'")
        d = len(names)

        c = self.describe(cost)
        if c is not None and d:
            for key, length in c.get("per_instrument", {}).items():
                if length is not None and length != d:
                    self.add(
                        f"nodes.{cost}.params.{key}",
                        f"has {length} entries but {ins!r} trades {d} instruments",
                    )

        widths: dict[str, int] = {}
        for feat in spec.nodes_of("features"):
            if gen not in spec.inputs(feat):
                self.add(f"nodes.{feat}", f"needs an edge from {gen!r}")
            f = self.describe(feat, n_instruments=d)
            if f is None:
                continue
            widths[feat] = f["out_dim"]
            provided = set(g["state"]) | {"spot", "time"}
            for need in f["requires"]:
                if need not in provided:
                    self.add(
                        f"nodes.{feat}",
                        f"reads {need!r}, which {gen!r} ({spec.nodes[gen].type}) does not "
                        f"provide; it provides {', '.join(sorted(provided))}",
                    )

        (evaluator,) = spec.nodes_of("evaluator")
        for hedger in spec.nodes_of("hedger"):
            h = self.describe(hedger)
            if h is None:
                continue
            where = f"nodes.{hedger}"
            if ins not in spec.inputs(hedger):
                self.add(where, f"needs an edge from the instrument set {ins!r}")
            if evaluator not in spec.outputs(hedger):
                self.add(where, f"needs an edge to the evaluator {evaluator!r}")
            if h["min_instruments"] and d < h["min_instruments"]:
                self.add(
                    where,
                    f"{spec.nodes[hedger].type} needs at least {h['min_instruments']} "
                    f"instruments, {ins!r} has {d}",
                )
            if h.get("n_out") is not None and d and h["n_out"] != d:
                self.add(
                    f"{where}.params.n_out",
                    f"is {h['n_out']} but {ins!r} trades {d} instruments",
                )
            feats = [n for n in spec.inputs(hedger) if spec.nodes[n].family == "features"]
            if h["learned"]:
                if len(feats) != 1:
                    self.add(where, f"a learned hedger needs one features input, has {len(feats)}")
                elif feats[0] in widths:
                    width = widths[feats[0]]
                    if h.get("n_in") is not None and h["n_in"] != width:
                        self.add(
                            f"{where}.params.n_in",
                            f"is {h['n_in']} but {feats[0]!r} produces {width} features",
                        )
            elif not g["has_pricer"] and not run.get("mc_fallback", False):
                self.add(
                    where,
                    f"classical hedger needs a pricer; {spec.nodes[gen].type} has none "
                    "in closed form. Set run.mc_fallback to true to use Monte Carlo greeks",
                )
