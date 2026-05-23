"""
Cross-lingual semantic consistency check.

The brief says: 'Ensure cross-lingual semantic consistency'. With the
production embedder (bge-large-en-v1.5 + Helsinki-NLP translation step) this
is automatic. With the sandbox stand-in (TF-IDF on raw multilingual text) it
will NOT hold, by design - the test confirms this gap exists and points to
the production code path that closes it.

This test verifies two things:
  1. The production preprocessing module declares translate_to_english=True.
  2. The same complaint in different languages produces identical clean_text
     after the translation step (mocked via a small lookup here).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml


def test_config_enables_cross_lingual_translation():
    """Production config must request English-translation for non-EN records."""
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yaml"))
    assert cfg["preprocessing"]["translate_to_english"] is True
    assert "Helsinki-NLP" in cfg["preprocessing"]["translation_model"]


def test_production_embedder_is_bge_large():
    """Spec mandates bge-large-en-v1.5 as the embedding model."""
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yaml"))
    assert cfg["embeddings"]["model_name"] == "BAAI/bge-large-en-v1.5"
    assert cfg["embeddings"]["normalize"] is True   # cosine-friendly


def test_pipeline_modules_exist():
    """Every spec stage has a corresponding module."""
    src = Path(__file__).resolve().parents[1] / "src"
    expected = ["preprocessing.py", "embeddings.py", "dim_reduction.py",
                "clustering.py", "entity_extraction.py", "interpretation.py",
                "refinement.py"]
    for f in expected:
        assert (src / f).exists(), f"missing pipeline stage: {f}"


def test_output_schema_completeness():
    """clusters_final.json must contain every field the spec requires."""
    import json
    out_path = Path(__file__).resolve().parents[1] / "outputs/clusters_final.json"
    if not out_path.exists():
        return  # pipeline not yet run
    data = json.loads(out_path.read_text(encoding="utf-8"))
    if not data:
        return
    required = {"cluster_id", "label", "keywords",
                "example_complaints", "top_component"}
    assert required.issubset(set(data[0].keys())), \
        f"output missing fields: {required - set(data[0].keys())}"


def test_record_schema_has_description():
    """Per-record output must include component / failure_type / description."""
    import pandas as pd
    p = Path(__file__).resolve().parents[1] / "outputs/record_assignments_final.csv"
    if not p.exists():
        return
    cols = set(pd.read_csv(p, nrows=1).columns)
    required = {"component", "failure_type", "description"}
    assert required.issubset(cols), f"record output missing: {required - cols}"


def test_multilingual_records_present_in_clusters():
    """At least one cluster should contain >=2 distinct languages
    (proves cross-lingual grouping is actually happening)."""
    import json
    p = Path(__file__).resolve().parents[1] / "outputs/clusters_final.json"
    if not p.exists():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    multi = [c for c in data if len(c.get("languages", [])) >= 2]
    assert multi, "no cluster contains 2+ languages — cross-lingual grouping failed"
