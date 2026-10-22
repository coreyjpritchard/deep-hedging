"""The hedge JSON is a documented schema, and the example file stays under 1 MB."""

from __future__ import annotations

import numpy as np
import pytest

EXAMPLE = "docs/examples/bs_delta_50bp.json"
ONE_MB = 1_000_000


def test_export_roundtrip_keeps_shapes_and_stats(impl, tmp_path):
    time = np.linspace(0.0, 1.0, 4)
    instruments = np.arange(24, dtype=float).reshape(2, 4, 3) + 0.1234567
    positions = np.arange(18, dtype=float).reshape(2, 3, 3) * 0.1
    pnl = np.array([1.23456789, -0.5])
    stats = {"mean": 0.367283945, "std": 0.867283945, "cvar": 0.5, "n": 20}
    books = {
        "flat": {
            "positions": np.zeros_like(positions),
            "pnl": np.zeros(2),
            "stats": {"mean": 0.0, "std": 0.0, "n": 20},
        }
    }
    path = tmp_path / "hedge.json"
    impl.export_hedge(
        path,
        market="BlackScholes",
        cost="ProportionalCost(rate=0.0, unit=None)",
        time=time,
        instrument_names=("spot", "variance_swap", "bond"),
        instruments=instruments,
        positions=positions,
        pnl=pnl,
        stats=stats,
        books=books,
        seed=7,
        note="roundtrip",
        max_paths=1,
    )
    document = impl.load_hedge(path)
    assert document["schema"] == "deephedge.hedge/v1"
    assert document["seed"] == 7
    assert np.asarray(document["instruments"]).shape == (1, 4, 3)
    assert np.asarray(document["positions"]).shape == (1, 3, 3)
    assert len(document["pnl"]) == 1
    assert document["stats"]["n"] == 20
    assert document["stats"]["mean"] == pytest.approx(0.367284, abs=1e-6)
    assert np.asarray(document["books"]["flat"]["positions"]).shape == (1, 3, 3)
    assert path.stat().st_size < ONE_MB


def test_export_rejects_a_position_axis_that_includes_maturity(impl, tmp_path):
    with pytest.raises(ValueError):
        impl.export_hedge(
            tmp_path / "bad.json",
            market="BlackScholes",
            cost="zero",
            time=np.linspace(0, 1, 3),
            instrument_names=("spot",),
            instruments=np.ones((2, 3, 1)),
            positions=np.ones((2, 3, 1)),
            pnl=np.zeros(2),
            stats={"mean": 0.0, "std": 0.0, "n": 2},
        )


def test_documented_example_is_the_schema_and_under_one_megabyte(impl):
    from pathlib import Path

    file = Path(EXAMPLE)
    assert file.is_file(), "docs/examples/bs_delta_50bp.json is the widget example"
    assert file.stat().st_size < ONE_MB
    document = impl.load_hedge(file)
    assert document["schema"] == "deephedge.hedge/v1"
    assert document["market"] == "BlackScholes"
    instruments = np.asarray(document["instruments"])
    positions = np.asarray(document["positions"])
    assert instruments.ndim == 3 and instruments.shape[0] >= 1
    assert positions.shape[0] == instruments.shape[0]
    assert positions.shape[1] == instruments.shape[1] - 1
    assert document["stats"]["n"] >= positions.shape[0]
    assert "network" in document["books"]
