"""
clustering.py
=============
Stage 4 — HDBSCAN on the UMAP-reduced embeddings.

Outputs
-------
* `cluster_assignments.parquet` -> record_id, domain, source_lang, cluster_id, prob
* `clusterer.joblib`            -> the fitted HDBSCAN model (for prediction
                                   on new records via `approximate_predict`).
"""

from __future__ import annotations

from pathlib import Path

import hdbscan
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from .utils import load_config, load_npy, load_parquet, save_parquet


def cluster(reduced: np.ndarray, **hdb_kwargs) -> hdbscan.HDBSCAN:
    logger.info(f"HDBSCAN on {reduced.shape}, "
                f"min_cluster_size={hdb_kwargs['min_cluster_size']}")
    clusterer = hdbscan.HDBSCAN(**hdb_kwargs)
    clusterer.fit(reduced)
    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    n_noise = int((clusterer.labels_ == -1).sum())
    logger.info(f"-> {n_clusters} clusters, {n_noise:,} noise points "
                f"({n_noise / len(reduced):.1%})")
    return clusterer


def run(cfg_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(cfg_path)
    cl = cfg["clustering"]

    reduced = load_npy(Path(cfg["paths"]["embeddings_dir"]) /
                       "embeddings_umap.npy")
    idx = load_parquet(Path(cfg["paths"]["embeddings_dir"]) /
                       "embeddings_index.parquet")

    clusterer = cluster(
        reduced,
        min_cluster_size=cl["min_cluster_size"],
        min_samples=cl["min_samples"],
        metric=cl["metric"],
        cluster_selection_method=cl["cluster_selection_method"],
        prediction_data=cl["prediction_data"],
    )

    idx["cluster_id"] = clusterer.labels_.astype(int)
    idx["cluster_prob"] = clusterer.probabilities_.astype(float)

    out_dir = Path(cfg["paths"]["artifacts_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(idx, out_dir / "cluster_assignments.parquet")
    joblib.dump(clusterer, out_dir / "clusterer.joblib")
    logger.info(f"Saved clusterer -> {out_dir / 'clusterer.joblib'}")
    return idx


if __name__ == "__main__":
    run()
