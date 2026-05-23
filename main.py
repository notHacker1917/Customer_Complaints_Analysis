"""
main.py
=======
End-to-end orchestrator for the Mercedes-Benz multilingual feedback pipeline.

Usage
-----
    # full run
    python main.py --config config.yaml

    # run a subset of stages (e.g. re-cluster after tweaking UMAP)
    python main.py --stages reduce,cluster,interpret,refine

    # individual modules can also be run directly:
    python -m src.preprocessing
    python -m src.embeddings
    python -m src.dim_reduction
    python -m src.clustering
    python -m src.entity_extraction
    python -m src.interpretation
    python -m src.refinement
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict

from loguru import logger

from src import (clustering, dim_reduction, embeddings, entity_extraction,
                 interpretation, preprocessing, refinement)
from src.utils import load_config, setup_logging


STAGES: Dict[str, Callable[[str], object]] = {
    "preprocess": preprocessing.run,
    "embed":      embeddings.run,
    "reduce":     dim_reduction.run,
    "cluster":    clustering.run,
    "extract":    entity_extraction.run,
    "interpret":  interpretation.run,
    "refine":     refinement.run,
}

DEFAULT_ORDER = ["preprocess", "embed", "reduce", "cluster",
                 "extract", "interpret", "refine"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--stages", default=",".join(DEFAULT_ORDER),
                   help="Comma-separated subset of: " + ",".join(STAGES.keys()))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(requested) - set(STAGES.keys())
    if unknown:
        raise SystemExit(f"Unknown stage(s): {unknown}")

    logger.info("=" * 70)
    logger.info(f"Pipeline start — stages: {requested}")
    logger.info("=" * 70)

    for stage in requested:
        t0 = time.time()
        logger.info(f">>> stage:{stage}")
        STAGES[stage](args.config)
        logger.info(f"<<< stage:{stage}  done in {time.time() - t0:.1f}s")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
