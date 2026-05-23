"""Unit tests for the rule-based entity extraction."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_run import build_re, load_gaz


def test_gazetteer_loads():
    terms = load_gaz("artifacts/components_gazetteer.txt")
    assert len(terms) > 50, "gazetteer should have at least 50 components"
    assert "seat" in terms
    assert "mbux" in terms


def test_regex_matches_longest_first():
    rgx = build_re(["seat", "driver seat", "driver seat heater"])
    m = rgx.search("the driver seat heater is broken")
    assert m is not None
    assert m.group(1) == "driver seat heater"   # longest wins


def test_regex_case_insensitive():
    rgx = build_re(["MBUX"])
    assert rgx.search("the mbux display is frozen") is not None


def test_regex_no_match():
    rgx = build_re(["spaceship"])
    assert rgx.search("the seat is broken") is None
