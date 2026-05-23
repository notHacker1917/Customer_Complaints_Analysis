"""
interpretation.py
=================
Stage 6 — for each cluster produce:

    * top-K TF-IDF keywords (cluster-vs-rest c-TF-IDF style)
    * a representative example (medoid — record nearest cluster centroid)
    * the most common affected component (from the entity-extraction stage)
    * an LLM-generated short, human-readable label
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .utils import load_config, load_npy, load_parquet, save_json, save_parquet


# ----------------------------------------------------------- c-TF-IDF
def class_based_tfidf(docs_per_cluster: Dict[int, str],
                      top_k: int = 12) -> Dict[int, List[str]]:
    """Compute class-based TF-IDF (à la BERTopic).

    docs_per_cluster maps cluster_id -> a single string concatenating all
    documents in that cluster. We TF-IDF across these *cluster pseudo-docs*
    so the highest-weight terms are those distinguishing one cluster from the
    others.
    """
    cluster_ids = sorted(docs_per_cluster.keys())
    corpus = [docs_per_cluster[c] for c in cluster_ids]

    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.9,
        sublinear_tf=True, stop_words="english", max_features=50_000,
    )
    X = vec.fit_transform(corpus)
    terms = np.array(vec.get_feature_names_out())

    out: Dict[int, List[str]] = {}
    for row, cid in enumerate(cluster_ids):
        scores = X[row].toarray().ravel()
        top = scores.argsort()[::-1][:top_k]
        out[cid] = [terms[i] for i in top if scores[i] > 0]
    return out


# ------------------------------------------------------------- LLM labeller
_LABEL_PROMPT = """You are labelling clusters of Mercedes-Benz customer
complaints.  Below is one cluster: a list of representative complaint snippets
and its top distinctive keywords.

Return a JSON object with:
{
  "label":   "<5-9 word human-readable failure topic>",
  "summary": "<one sentence describing what unifies these complaints>"
}

Cluster keywords: {keywords}

Sample complaints:
{examples}
"""


def llm_label(provider: str, model: str,
              keywords: List[str], examples: List[str]) -> dict:
    if provider == "rule_only":
        return {"label": " / ".join(keywords[:3]) or "unlabeled",
                "summary": ""}
    prompt = _LABEL_PROMPT.format(
        keywords=", ".join(keywords),
        examples="\n".join(f"- {e[:240]}" for e in examples),
    )
    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=model, max_tokens=300,
                messages=[{"role": "user", "content": prompt}])
            raw = resp.content[0].text
        else:
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            raw = resp.choices[0].message.content
        m = re.search(r"\{[\s\S]*\}", raw)
        return json.loads(m.group(0)) if m else {"label": keywords[0],
                                                 "summary": ""}
    except Exception as e:
        logger.warning(f"LLM label failed: {e}")
        return {"label": " / ".join(keywords[:3]) or "unlabeled",
                "summary": ""}


# ------------------------------------------------------------- centroids
def cluster_medoids(reduced: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    """Return cluster_id -> row index of medoid (closest to centroid)."""
    out: Dict[int, int] = {}
    for cid in np.unique(labels):
        if cid == -1:
            continue
        mask = labels == cid
        members_idx = np.where(mask)[0]
        members = reduced[mask]
        centroid = members.mean(axis=0)
        dists = np.linalg.norm(members - centroid, axis=1)
        out[int(cid)] = int(members_idx[dists.argmin()])
    return out


# =============================================================== main entry
def run(cfg_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(cfg_path)
    ic = cfg["interpretation"]

    feedback = load_parquet(
        Path(cfg["paths"]["processed_data_dir"]) / "feedback_entities.parquet")
    assignments = load_parquet(
        Path(cfg["paths"]["artifacts_dir"]) / "cluster_assignments.parquet")
    reduced = load_npy(
        Path(cfg["paths"]["embeddings_dir"]) / "embeddings_umap.npy")

    df = feedback.merge(assignments[["record_id", "cluster_id", "cluster_prob"]],
                        on="record_id", how="inner")

    # group docs per cluster (skip noise -1)
    cluster_docs: Dict[int, List[str]] = {}
    for cid, sub in df[df["cluster_id"] != -1].groupby("cluster_id"):
        cluster_docs[int(cid)] = sub["text_for_embed"].tolist()

    pseudo_docs = {cid: " ".join(d) for cid, d in cluster_docs.items()}
    keywords = class_based_tfidf(pseudo_docs, top_k=ic["tfidf_top_k"])

    # medoids (representative records)
    medoids = cluster_medoids(reduced, df["cluster_id"].to_numpy())

    rows = []
    for cid, kw in tqdm(keywords.items(), desc="label-clusters"):
        members = df[df["cluster_id"] == cid]
        examples = (members.sort_values("cluster_prob", ascending=False)
                           ["text_for_embed"].head(ic["examples_per_cluster"])
                           .tolist())

        if ic["llm_label"]:
            lab = llm_label(ic["llm_provider"], ic["llm_model"], kw, examples)
        else:
            lab = {"label": " / ".join(kw[:3]), "summary": ""}

        comp_counter = Counter(members["component"].dropna().str.lower())
        top_comp = comp_counter.most_common(1)[0][0] if comp_counter else None

        rows.append({
            "cluster_id":   cid,
            "size":         len(members),
            "label":        lab["label"],
            "summary":      lab["summary"],
            "keywords":     kw,
            "top_component": top_comp,
            "example_complaints": examples,
            "medoid_record_id": df.iloc[medoids[cid]]["record_id"]
            if cid in medoids else None,
            "domains":      sorted(members["domain"].unique().tolist()),
            "languages":    sorted(members["source_lang"].unique().tolist()),
        })

    out = pd.DataFrame(rows).sort_values("size", ascending=False)
    save_parquet(out, Path(cfg["paths"]["artifacts_dir"]) / "clusters_labeled.parquet")
    save_json(out.to_dict(orient="records"),
              Path(cfg["paths"]["artifacts_dir"]) / "clusters_labeled.json")
    return out


if __name__ == "__main__":
    run()
