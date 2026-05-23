"""
embeddings.py
=============
Stage 2 — generate dense semantic embeddings.

Model: BAAI/bge-large-en-v1.5
    * 1024-dim
    * Trained with cosine similarity in mind -> we always L2-normalize.
    * Cross-lingual consistency is achieved by FIRST translating non-English
      records to English (see preprocessing.py).  bge-large-en is a single-
      language model so this is the standard recipe; if you switch to
      `bge-m3` you can drop the translation step.

We embed the column `text_for_embed`, write a numpy `.npy` matrix shape
(N, 1024) and a sidecar parquet with `record_id` order so they stay aligned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from .utils import load_config, load_parquet, save_npy, save_parquet


def get_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed_texts(texts: list[str],
                model_name: str = "BAAI/bge-large-en-v1.5",
                batch_size: int = 64,
                device: str = "auto",
                normalize: bool = True) -> np.ndarray:
    """Encode a list of strings into a (N, dim) ndarray."""
    device = get_device(device)
    logger.info(f"Loading {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    logger.info(f"Encoded {embs.shape[0]:,} texts -> dim={embs.shape[1]}")
    return embs.astype(np.float32)


def run(cfg_path: str = "config.yaml") -> None:
    cfg = load_config(cfg_path)
    ec = cfg["embeddings"]

    src = Path(cfg["paths"]["processed_data_dir"]) / "feedback_clean.parquet"
    out_emb = Path(cfg["paths"]["embeddings_dir"]) / "embeddings.npy"
    out_idx = Path(cfg["paths"]["embeddings_dir"]) / "embeddings_index.parquet"

    df = load_parquet(src)
    embs = embed_texts(
        df["text_for_embed"].tolist(),
        model_name=ec["model_name"],
        batch_size=ec["batch_size"],
        device=ec["device"],
        normalize=ec["normalize"],
    )

    save_npy(embs, out_emb)
    save_parquet(df[["record_id", "domain", "source_lang"]], out_idx)


if __name__ == "__main__":
    run()
