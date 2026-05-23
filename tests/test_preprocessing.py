"""Unit tests for the preprocessing stage."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from scripts.test_run import clean_and_anon, shingles, near_dedup, make_description
import pandas as pd


class TestClean:
    def test_strips_html(self):
        assert "<p>" not in clean_and_anon("hello <p>world</p>")

    def test_collapses_whitespace(self):
        assert clean_and_anon("hello    world  ") == "hello world"

    def test_anonymizes_email(self):
        out = clean_and_anon("contact me at john@example.com")
        assert "<EMAIL>" in out and "john" not in out

    def test_anonymizes_phone(self):
        out = clean_and_anon("call +49 30 1234567 today")
        assert "<PHONE>" in out

    def test_anonymizes_vin(self):
        out = clean_and_anon("VIN WDB2030461A123456 broken")
        assert "<VIN>" in out

    def test_anonymizes_plate(self):
        out = clean_and_anon("plate B-MW 1234 abandoned")
        assert "<PLATE>" in out

    def test_handles_none(self):
        assert clean_and_anon(None) == ""

    def test_handles_non_string(self):
        assert clean_and_anon(42) == ""

    def test_dedups_punctuation(self):
        assert "!!!" not in clean_and_anon("help me!!!!!!")


class TestDedup:
    def test_shingles_nonempty(self):
        assert len(shingles("hello world")) > 0

    def test_near_dedup_drops_near_dup(self):
        df = pd.DataFrame({"clean_text": [
            "the seat heater stops working after 5 minutes",
            "the seat heater stops working after 6 minutes",  # near-dup
            "the steering wheel rattles on the highway",
        ]})
        out = near_dedup(df, "clean_text", 0.70)
        assert len(out) == 2


class TestDescription:
    def test_has_component_and_failure(self):
        assert make_description("seat heater", "stopped", None) == "seat heater - stopped"

    def test_only_component(self):
        assert "seat" in make_description("seat", None, None)

    def test_only_failure(self):
        assert "rattle" in make_description(None, "rattle", None)

    def test_both_none(self):
        assert make_description(None, None, None) is None
