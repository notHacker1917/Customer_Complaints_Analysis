"""
dim_reduction.py
================
Stage 3 — UMAP from 1024-D to ~15-D so HDBSCAN can find dense regions
without suffering from the curse of dimensionality.

Why these defaults
------------------
* metric="cosine"          embeddings are L2-normalized -> cosine is correct.
* n_neighbors=30           large enough to capture broad failure topics.
* min_dist=0.0             tight clusters; we *want* them packed for HDBSCAN.
* n_components=15          enough capacity for ~50-200 sub-topics, still cheap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import umap
from loguru import logger

from .utils import load_config, load_npy, save_npy


def reduce(embeddings: np.ndarray, **umap_kwargs) -> np.ndarray:
    logger.info(f"UMAP -> {umap_kwargs.get('n_components', 15)} dims  "
                f"on {embeddings.shape[0]:,} x {embeddings.shape[1]}")
    reducer = umap.UMAP(**umap_kwargs)
    out = reducer.fit_transform(embeddings).astype(np.float32)
    logger.info(f"UMAP done -> {out.shape}")
    return out


def run(cfg_path: str = "config.yaml") -> None:
    cfg = load_config(cfg_path)
    dr = cfg["dim_reduction"]

    src = Path(cfg["paths"]["embeddings_dir"]) / "embeddings.npy"
    dst = Path(cfg["paths"]["embeddings_dir"]) / "embeddings_umap.npy"

    embs = load_npy(src)
    reduced = reduce(
        embs,
        n_neighbors=dr["n_neighbors"],
        n_components=dr["n_components"],
        min_dist=dr["min_dist"],
        metric=dr["metric"],
        random_state=dr["random_state"],
        verbose=True,
    )
    save_npy(reduced, dst)


if __name__ == "__main__":
    run()
