"""
utils.py
========
Cross-cutting helpers: config loading, IO, logging, hashing, batching.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, List

import numpy as np
import pandas as pd
import yaml
from loguru import logger


# --------------------------------------------------------------------- config
def load_config(path: str | Path = "config.yaml") -> dict:
    """Load YAML config file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {path.resolve()}")
    return cfg


def ensure_dirs(cfg: dict) -> None:
    """Create all directories declared in cfg['paths']."""
    for k, v in cfg.get("paths", {}).items():
        Path(v).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- io helpers
def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Wrote {len(df):,} rows -> {path}")


def load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info(f"Read  {len(df):,} rows <- {path}")
    return df


def save_npy(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    logger.info(f"Wrote ndarray {arr.shape} -> {path}")


def load_npy(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    logger.info(f"Read  ndarray {arr.shape} <- {path}")
    return arr


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------------------------------------------- misc helpers
def stable_hash(text: str) -> str:
    """Deterministic short hash, useful for record IDs."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def batched(iterable: Iterable, n: int) -> Iterator[List]:
    """Yield successive n-sized chunks from iterable."""
    buf: list = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def setup_logging(cfg: dict) -> None:
    """Configure loguru sinks based on cfg."""
    level = cfg.get("runtime", {}).get("log_level", "INFO")
    logs_dir = Path(cfg["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda m: print(m, end=""), level=level)
    logger.add(logs_dir / "pipeline_{time}.log",
               level=level, rotation="10 MB", retention=5)
