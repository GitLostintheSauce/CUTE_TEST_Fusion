"""Tests for the Grad-Shafranov dataset tooling.

These cover the parts that do not need the solver, so they run in CI where OFT
is not importable: the filter that decides whether a solve is usable, and the
statistic used to argue that q95, beta_pol and l_i are not merely restatements
of Ip.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load(script_name: str):
    """Import a script from scripts/ without executing its __main__ block."""
    path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load("generate_gs_dataset.py")
check = _load("check_gs_dataset.py")


def _stats(ip=1.0e5, a_geo=0.17, r_geo=0.32):
    return {"Ip": ip, "a_geo": a_geo, "R_geo": r_geo}


class TestSolveIsSane:
    def test_accepts_a_normal_solve(self):
        assert gen.solve_is_sane(_stats(), 1.0e5)

    def test_rejects_current_far_from_target(self):
        """A converged solve that missed the requested current teaches a
        relationship that was never asked for."""
        assert not gen.solve_is_sane(_stats(ip=5.0e4), 1.0e5)

    def test_accepts_small_current_miss(self):
        assert gen.solve_is_sane(_stats(ip=1.1e5), 1.0e5)

    def test_rejects_degenerate_cross_section(self):
        assert not gen.solve_is_sane(_stats(a_geo=0.001), 1.0e5)
        assert not gen.solve_is_sane(_stats(a_geo=0.9), 1.0e5)

    def test_rejects_plasma_outside_the_vessel(self):
        assert not gen.solve_is_sane(_stats(r_geo=1.5), 1.0e5)

    def test_rejects_non_finite_values(self):
        assert not gen.solve_is_sane(_stats(ip=np.nan), 1.0e5)
        assert not gen.solve_is_sane(_stats(a_geo=np.inf), 1.0e5)

    def test_rejects_missing_keys(self):
        assert not gen.solve_is_sane({"Ip": 1.0e5}, 1.0e5)
        assert not gen.solve_is_sane({}, 1.0e5)


class TestRSquared:
    def test_perfect_line_scores_one(self):
        """A quantity computed from Ip by formula, as in the reduced dataset."""
        x = np.linspace(1.0, 2.0, 50)
        assert check.r_squared_against(x, 3.0 * x + 1.0) == pytest.approx(1.0)

    def test_unrelated_quantity_scores_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        y = rng.normal(size=400)
        assert abs(check.r_squared_against(x, y)) < 0.1

    def test_partial_dependence_lands_between(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=800)
        y = x + rng.normal(size=800)
        assert 0.3 < check.r_squared_against(x, y) < 0.7


def test_shape_ranges_are_physical():
    """Ranges must be orderable and positive where physics demands it."""
    for name, (lo, hi) in gen.SHAPE_RANGES.items():
        assert lo < hi, f"{name} range is inverted"
    assert gen.SHAPE_RANGES["a"][1] < gen.SHAPE_RANGES["R0"][0], (
        "minor radius must stay smaller than the major radius, or the plasma "
        "would enclose the machine axis"
    )
    assert gen.SHAPE_RANGES["kappa"][0] >= 1.0, "elongation below 1 is not a shape"


def test_label_order_matches_the_reduced_surrogate():
    """A GS-trained model has to be swappable with the existing one."""
    from src.ml.dataset import PARAM_NAMES as REDUCED_NAMES

    assert gen.PARAM_NAMES == REDUCED_NAMES


def test_extras_exclude_the_primary_labels():
    assert not set(gen.EXTRA_NAMES) & set(gen.PARAM_NAMES)
