"""Graph spec and registry (M01): round trip, stable id, and what `validate` rejects.

No Monte Carlo here, so no tolerances: every check is exact. The example spec is
read from `docs/plan/01-interfaces.md`, so the documented example and the code
cannot drift apart.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from deephedge.registry import (
    REQUIRED,
    SCHEMA_PATH,
    Block,
    Param,
    Registry,
    default_registry,
    schema_text,
)
from deephedge.spec import SCHEMA, GraphSpec, Node, canonical_json, validate

ROOT = Path(__file__).resolve().parent.parent


def example_text() -> str:
    text = (ROOT / "docs" / "plan" / "01-interfaces.md").read_text()
    match = re.search(r"## Graph spec\n\n```json\n(.*?)```", text, re.S)
    assert match, "the Graph spec example is missing from 01-interfaces.md"
    return match.group(1)


@pytest.fixture()
def example() -> dict:
    """The documented example without its stored id, so a test can edit it."""
    data = json.loads(example_text())
    del data["id"]
    return data


def problems(data: dict) -> list[str]:
    return [str(p) for p in validate(GraphSpec.from_dict(data))]


def only(data: dict, *fragments: str) -> str:
    """The single problem in `data`, which must contain every fragment."""
    found = problems(data)
    assert len(found) == 1, found
    for fragment in fragments:
        assert fragment in found[0], found[0]
    return found[0]


# ---- round trip and id -------------------------------------------------------


def test_example_validates():
    assert validate(GraphSpec.from_json(example_text())) == []


def test_round_trip(example):
    spec = GraphSpec.from_dict(example)
    again = GraphSpec.from_json(spec.to_json())
    assert again.content() == spec.content()
    assert again.spec_id == spec.spec_id
    assert again.declared_id == spec.spec_id
    assert validate(again) == []
    assert json.loads(again.to_json()) == json.loads(spec.to_json())


def test_spec_id_stable_under_key_and_edge_order(example):
    shuffled = {k: example[k] for k in reversed(list(example))}
    shuffled["nodes"] = {k: example["nodes"][k] for k in reversed(list(example["nodes"]))}
    for node in shuffled["nodes"].values():
        if "params" in node:
            node["params"] = dict(reversed(list(node["params"].items())))
    shuffled["edges"] = list(reversed(example["edges"]))
    a, b = GraphSpec.from_dict(example), GraphSpec.from_dict(shuffled)
    assert a.spec_id == b.spec_id
    assert canonical_json(a.content()) == canonical_json(b.content())


def test_spec_id_is_16_hex_and_moves_with_content(example):
    spec = GraphSpec.from_dict(example)
    assert re.fullmatch(r"[0-9a-f]{16}", spec.spec_id)
    changed = copy.deepcopy(example)
    changed["nodes"]["mkt"]["params"]["rho"] = -0.5
    assert GraphSpec.from_dict(changed).spec_id != spec.spec_id
    reseeded = copy.deepcopy(example)
    reseeded["seeds"]["test"] = 54321
    assert GraphSpec.from_dict(reseeded).spec_id != spec.spec_id


def test_stored_id_is_checked(example):
    documented = json.loads(example_text())
    assert documented["id"] == GraphSpec.from_dict(example).spec_id
    example["id"] = "0" * 16
    only(example, "id:", "does not match")
    example["id"] = documented["id"]
    example["nodes"]["mkt"]["params"]["rho"] = -0.5
    only(example, "id:", "does not match")


def test_spec_is_read_only(example):
    spec = GraphSpec.from_dict(example)
    with pytest.raises(TypeError):
        spec.nodes["mkt"].params["rho"] = 0.0  # type: ignore[index]
    with pytest.raises(AttributeError):
        spec.run = {}  # type: ignore[misc]


def test_from_dict_rejects_unknown_fields(example):
    with pytest.raises(ValueError, match="unknown top-level"):
        GraphSpec.from_dict({**example, "notes": "x"})
    example["nodes"]["mkt"]["kind"] = "heston"
    with pytest.raises(ValueError, match="unknown fields"):
        GraphSpec.from_dict(example)


# ---- what validate catches ----------------------------------------------------


def test_unknown_type_is_named_with_a_suggestion(example):
    example["nodes"]["mkt"]["type"] = "generator.hestn"
    msg = only(example, "nodes.mkt.type", "unknown block type 'generator.hestn'")
    assert "generator.heston" in msg


def test_bad_parameter_type_range_and_name(example):
    bad_type = copy.deepcopy(example)
    bad_type["nodes"]["mkt"]["params"]["kappa"] = "fast"
    only(bad_type, "nodes.mkt.params.kappa", "number")
    bad_range = copy.deepcopy(example)
    bad_range["nodes"]["obj"]["params"]["alpha"] = 1.0
    only(bad_range, "nodes.obj.params.alpha", "< 1.0")
    bad_name = copy.deepcopy(example)
    bad_name["nodes"]["h_net"]["params"]["widht"] = 64
    only(bad_name, "nodes.h_net.params.widht", "unknown parameter")
    boolean = copy.deepcopy(example)
    boolean["nodes"]["h_net"]["params"]["depth"] = True
    only(boolean, "nodes.h_net.params.depth", "integer")


def test_dimension_mismatch_features_to_network(example):
    # time, log moneyness, variance, and one previous position per instrument (2).
    example["nodes"]["h_net"]["params"]["n_in"] = 4
    only(example, "nodes.h_net.params.n_in", "is 4", "produces 5 features")
    example["nodes"]["h_net"]["params"]["n_in"] = 5
    assert problems(example) == []


def test_dimension_mismatch_instruments_to_hedger(example):
    example["nodes"]["h_net"]["params"]["n_out"] = 1
    only(example, "nodes.h_net.params.n_out", "trades 2 instruments")


def test_dimension_mismatch_cost_rates(example):
    example["nodes"]["cost"]["params"]["rate"] = [0.005, 0.0, 0.001]
    only(example, "nodes.cost.params.rate", "3 entries", "trades 2 instruments")


def test_delta_vega_needs_two_instruments(example):
    example["nodes"]["ins"]["params"]["names"] = ["spot"]
    example["nodes"]["cost"]["params"]["rate"] = 0.005
    only(example, "nodes.h_delta", "at least 2 instruments")


@pytest.mark.parametrize("field", ["train", "validation", "test", "torch"])
def test_missing_seed(example, field):
    del example["seeds"][field]
    only(example, f"seeds.{field}", "missing seed")


def test_train_equal_to_test_is_rejected_naming_both(example):
    example["seeds"]["train"] = example["seeds"]["test"]
    msg = only(example, "seeds.train", "seeds.test")
    assert "seeds.train and seeds.test are both 12345" in msg


def test_validation_equal_to_test_is_rejected(example):
    example["seeds"]["validation"] = 12345
    only(example, "seeds.validation and seeds.test")


def test_torch_seed_may_equal_a_split_seed(example):
    assert example["seeds"]["torch"] == example["seeds"]["train"] == 0
    assert problems(example) == []


def test_negative_or_unknown_seed(example):
    example["seeds"]["test"] = -1
    only(example, "seeds.test", "non-negative")
    example["seeds"]["test"] = 12345
    example["seeds"]["bootstrap"] = 3
    only(example, "seeds.bootstrap", "unknown seed")


def test_unknown_edge(example):
    example["edges"].append(["feat", "h_gru"])
    only(example, "edges[", "unknown node 'h_gru'")


def test_edge_of_a_kind_that_is_not_allowed(example):
    example["edges"].append(["cost", "mkt"])
    only(example, "an edge from a cost to a generator is not allowed")


def test_duplicate_edge(example):
    example["edges"].append(["mkt", "ins"])
    only(example, "duplicate edge")


def test_feature_reading_state_the_generator_lacks(example):
    example["nodes"]["mkt"] = {"type": "generator.black_scholes", "params": {"sigma": 0.2}}
    example["nodes"]["ins"]["params"]["names"] = ["spot"]
    example["nodes"]["cost"]["params"]["rate"] = 0.005
    example["nodes"]["h_delta"]["type"] = "hedger.delta"
    only(example, "nodes.feat", "reads 'variance'", "generator.black_scholes")


def test_instrument_the_generator_does_not_offer(example):
    example["nodes"]["ins"]["params"]["names"] = ["spot", "listed_call"]
    only(example, "nodes.ins.params.names", "listed_call not offered")


def test_classical_hedger_without_pricer_needs_mc_fallback(example):
    example["nodes"]["mkt"] = {"type": "generator.rough_bergomi", "params": {}}
    only(example, "nodes.h_delta", "run.mc_fallback")
    example["run"]["mc_fallback"] = True
    assert problems(example) == []


def test_rough_bergomi_default_hurst_is_one_tenth():
    registry = default_registry()
    resolved, found = registry.check_params(registry.param_schema("generator.rough_bergomi"), {})
    assert found == []
    assert resolved["H"] == 0.1
    assert registry.build(Node("generator.rough_bergomi")).params()["H"] == 0.1


def test_two_generators_in_one_spec(example):
    example["nodes"]["mkt2"] = {"type": "generator.black_scholes"}
    only(example, "exactly one generator", "mkt, mkt2")


def test_run_sizes_checked(example):
    example["run"]["n_steps"] = 0
    only(example, "run.n_steps", ">= 1")


def test_schema_string(example):
    example["schema"] = "deephedge.graph/v0"
    only(example, "schema", SCHEMA)


def test_several_problems_are_all_reported(example):
    example["seeds"]["train"] = 12345
    example["nodes"]["obj"]["params"]["alpha"] = 2.0
    example["edges"].append(["mkt", "nowhere"])
    found = problems(example)
    assert len(found) == 3, found


# ---- registry ----------------------------------------------------------------


def test_every_registered_block_builds_with_params_and_schema():
    registry = default_registry()
    for name in registry:
        cls = registry.cls(name)
        assert isinstance(cls.param_schema, dict)
        schema = registry.param_schema(name)
        params = {
            k: ["spot"] if k == "names" else ["time"] for k, p in schema.items() if p.required
        }
        block = registry.build(Node(name, params))
        resolved, found = registry.check_params(schema, params)
        assert found == []
        assert block.params() == resolved


def test_build_rejects_bad_params_and_unknown_types():
    registry = default_registry()
    with pytest.raises(ValueError, match="alpha"):
        registry.build(Node("objective.cvar", {"alpha": 1.5}))
    with pytest.raises(KeyError, match="did you mean"):
        registry.build(Node("objective.cvarr"))


def test_register_a_new_block():
    class Digital(Block):
        param_schema = {"strike": Param("float", REQUIRED, lower=0.0, open_lower=True)}

    registry = Registry()
    registry.register("contract.digital", Digital)
    built = registry.build(Node("contract.digital", {"strike": 100.0}))
    assert isinstance(built, Digital) and built.params() == {"strike": 100.0}
    with pytest.raises(ValueError, match="missing required"):
        registry.build(Node("contract.digital"))
    with pytest.raises(ValueError, match="family.name"):
        registry.register("digital", Digital)


def test_committed_json_schema_matches_the_registry():
    assert SCHEMA_PATH.read_text() == schema_text(), "run python -m deephedge.registry"


def test_json_schema_lists_every_type(example):
    schema = json.loads(SCHEMA_PATH.read_text())
    variants = schema["properties"]["nodes"]["additionalProperties"]["oneOf"]
    names = {v["properties"]["type"]["const"] for v in variants}
    assert names == set(default_registry().types())
    assert {n["type"] for n in example["nodes"].values()} <= names


def test_validate_imports_neither_torch_nor_a_simulator():
    code = (
        "import sys\n"
        "from tests.test_spec import example_text\n"
        "from deephedge.spec import GraphSpec, validate\n"
        "assert validate(GraphSpec.from_json(example_text())) == []\n"
        "heavy = [m for m in ('torch', 'deephedge.env', 'deephedge.heston', 'scipy')"
        " if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
