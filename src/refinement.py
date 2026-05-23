"""
refinement.py
=============
Stage 7 — post-process the cluster set:

    1. Build per-cluster centroid embeddings (mean of member embeddings).
    2. Compute pairwise cosine similarity between centroids.
    3. Greedily merge clusters with similarity >= threshold.
    4. Drop clusters smaller than `drop_clusters_smaller_than`.
    5. Optionally drop the noise label (-1).
    6. Re-emit final cluster table + per-record assignment file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from .utils import load_config, load_npy, load_parquet, save_json, save_parquet


def centroids(embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for cid in np.unique(labels):
        if cid == -1:
            continue
        v = embeddings[labels == cid].mean(axis=0)
        # re-normalize so cosine == dot
        v = v / (np.linalg.norm(v) + 1e-12)
        out[int(cid)] = v.astype(np.float32)
    return out


def find_merges(centroid_map: Dict[int, np.ndarray],
                threshold: float) -> Dict[int, int]:
    """Union-find: returns mapping old_id -> new_id (canonical)."""
    ids = sorted(centroid_map.keys())
    if not ids:
        return {}
    M = np.stack([centroid_map[c] for c in ids])
    sim = cosine_similarity(M)
    np.fill_diagonal(sim, 0.0)

    parent = {c: c for c in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # keep the smaller id as canonical for stability
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    pairs_merged = 0
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            if sim[i, j] >= threshold:
                union(a, ids[j])
                pairs_merged += 1
    logger.info(f"Refinement merged {pairs_merged} cluster pairs "
                f"(threshold={threshold})")
    return {c: find(c) for c in ids}


def run(cfg_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(cfg_path)
    rc = cfg["refinement"]

    embs = load_npy(Path(cfg["paths"]["embeddings_dir"]) / "embeddings.npy")
    assignments = load_parquet(
        Path(cfg["paths"]["artifacts_dir"]) / "cluster_assignments.parquet")
    labeled = load_parquet(
        Path(cfg["paths"]["artifacts_dir"]) / "clusters_labeled.parquet")

    labels = assignments["cluster_id"].to_numpy()
    cmap = centroids(embs, labels)
    merge_map = find_merges(cmap, rc["merge_cosine_threshold"])

    # Apply merges
    assignments["cluster_id_refined"] = assignments["cluster_id"].map(
        lambda c: merge_map.get(int(c), c) if c != -1 else -1)

    # Drop tiny clusters
    sizes = assignments[assignments["cluster_id_refined"] != -1] \
        .groupby("cluster_id_refined").size()
    keep = set(sizes[sizes >= rc["drop_clusters_smaller_than"]].index)
    logger.info(f"Keeping {len(keep)} clusters >= "
                f"{rc['drop_clusters_smaller_than']} members "
                f"(dropping {len(sizes) - len(keep)})")

    assignments["cluster_id_final"] = assignments["cluster_id_refined"].apply(
        lambda c: c if c in keep else -1)

    if rc.get("drop_noise_label", -1) is not None:
        # records labeled -1 are kept in the assignment file but tagged "noise"
        pass

    # Rebuild the cluster summary by aggregating from existing labeled rows
    refined_rows: List[dict] = []
    for new_cid in sorted(keep):
        member_old = [old for old, new in merge_map.items() if new == new_cid]
        sub = labeled[labeled["cluster_id"].isin(member_old)]
        if sub.empty:
            continue
        # pick the largest source-cluster's label as the canonical label
        sub = sub.sort_values("size", ascending=False)
        primary = sub.iloc[0]
        merged_keywords = []
        seen = set()
        for kws in sub["keywords"]:
            for k in kws:
                if k not in seen:
                    merged_keywords.append(k)
                    seen.add(k)
        refined_rows.append({
            "cluster_id": int(new_cid),
            "size": int(assignments[assignments["cluster_id_final"] == new_cid]
                        .shape[0]),
            "label": primary["label"],
            "summary": primary["summary"],
            "keywords": merged_keywords[:15],
            "top_component": primary["top_component"],
            "example_complaints": primary["example_complaints"],
            "domains": sorted({d for ds in sub["domains"] for d in ds}),
            "languages": sorted({l for ls in sub["languages"] for l in ls}),
            "merged_from": member_old,
        })

    final = pd.DataFrame(refined_rows).sort_values("size", ascending=False)

    out_dir = Path(cfg["paths"]["output_dir"])
    save_parquet(final, out_dir / "clusters_final.parquet")
    save_json(final.to_dict(orient="records"), out_dir / "clusters_final.json")
    save_parquet(assignments, out_dir / "record_assignments_final.parquet")
    logger.info(f"Wrote {len(final)} final clusters to {out_dir}")
    return final


if __name__ == "__main__":
    run()
