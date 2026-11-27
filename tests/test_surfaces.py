"""Tests for snapshot-calibrated training and evaluation (synthetic markets only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deephedge import surfaces
from deephedge.pricing import bs_call_price


def _calibration() -> dict:
    maturity = 30.0 / 365.0
    sigma = 0.2
    premium = float(bs_call_price(100.0, 100.0, maturity, sigma, 0.0))
    heston = {
        "v0": 0.04,
        "kappa": 1.0,
        "theta": 0.04,
        "xi": 0.5,
        "rho": -0.7,
        "premium": premium,
    }
    return {
        "schema": "volsurface.sanos_calibration/v1",
        "note": "synthetic test calibration; not a historical snapshot",
        "contract": {
            "s0": 100.0,
            "strike": 100.0,
            "maturity": maturity,
            "r": 0.0,
            "n_steps": 4,
        },
        "black_scholes": {"sigma": sigma, "premium": premium},
        "surface": {"atm_vol": sigma, "atm_skew": 0.0, "premium": premium},
        "heston_fits": {"full": heston, "contract": heston},
        "heston": heston,
    }


TINY = {"m_train": 64, "epochs": 1, "width": 8, "depth": 1, "batch_size": 32, "seed": 1}


def test_setups_read_the_calibration_contract():
    cal = _calibration()
    bs = surfaces.black_scholes_setup(cal, 0.005)
    heston = surfaces.heston_setup(cal, 0.005, fit="contract")
    assert bs.sigma == pytest.approx(0.2)
    assert bs.cost == pytest.approx(0.005)
    assert bs.s0 == 100.0
    assert heston.kappa == pytest.approx(1.0)
    assert heston.v0 == pytest.approx(0.04)
    assert heston.maturity == pytest.approx(cal["contract"]["maturity"])


def test_weights_path_is_not_a_session_checkpoint():
    path = surfaces.weights_path("snapshot_bs", 0.005)
    assert path.parent == surfaces.WEIGHTS_DIR
    assert "s00" not in path.name
    assert "s01" not in path.name
    assert "surfaces" in path.parts
    assert path.name == "snapshot_bs_c050bp.pt"


def test_parse_costs_accepts_rates_and_basis_points():
    assert surfaces._parse_costs("0,0.005") == (0.0, 0.005)
    assert surfaces._parse_costs("0,50") == (0.0, 0.005)


def test_train_and_evaluate_synthetic_markets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(surfaces, "WEIGHTS_DIR", tmp_path)
    cal = _calibration()
    summary = surfaces.train_markets(
        cal,
        markets=("snapshot_bs", "snapshot_heston"),
        costs=(0.0,),
        overrides=TINY,
    )
    assert {row["market"] for row in summary} == {"snapshot_bs", "snapshot_heston"}
    assert (tmp_path / "snapshot_bs_c000bp.pt").exists()
    assert (tmp_path / "snapshot_heston_c000bp.pt").exists()
    rows = surfaces.evaluate_markets(
        cal,
        markets=("snapshot_bs", "snapshot_heston"),
        costs=(0.0,),
        n_paths=64,
        seed=12345,
        overrides=TINY,
    )
    hedges = {(row["market"], row["hedge"]) for row in rows}
    assert ("snapshot_bs", "network") in hedges
    assert ("snapshot_bs", "delta") in hedges
    assert ("snapshot_bs", "none") in hedges
    assert ("snapshot_bs", "model") not in hedges
    assert ("snapshot_heston", "network") in hedges
    assert ("snapshot_heston", "model") in hedges
    assert ("snapshot_heston", "delta") in hedges
    for row in rows:
        assert row["n_paths"] == 64
        assert row["eval_seed"] == 12345
        assert np.isfinite(row["mean"])
        assert np.isfinite(row["cvar_50"])
        assert np.isfinite(row["entropic_1"])
        assert "s00" not in str(tmp_path)
