"""Unit tests for clustering & label heuristic."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts.test_run import merge_similar, heuristic_label


def test_merge_similar_merges_close():
    """Two near-identical centroids should collapse to one cluster id."""
    rng = np.random.default_rng(42)
    base = rng.normal(size=(10, 32)).astype(np.float32)
    # cluster 0 and cluster 1 share the same centroid pattern; cluster 2 is far
    emb = np.vstack([base, base + 1e-3, rng.normal(size=(10, 32))])
    labels = np.array([0]*10 + [1]*10 + [2]*10)
    merge_map = merge_similar(emb, labels, thr=0.99)
    # 0 and 1 should map to the same canonical id (the smaller one, 0)
    assert merge_map[1] == merge_map[0]
    # 2 should map to itself
    assert merge_map[2] == 2


def test_heuristic_label_uses_component_and_failure():
    out = heuristic_label("driver seat heater", "stops working", ["seat"], [])
    assert "Driver seat heater" in out["label"]
    assert "stops working" in out["label"]


def test_heuristic_label_fallback_to_keywords():
    out = heuristic_label(None, None, ["mbux", "freezes", "navigation"], [])
    assert out["label"]   # not empty
    assert "Unlabeled" not in out["label"]
