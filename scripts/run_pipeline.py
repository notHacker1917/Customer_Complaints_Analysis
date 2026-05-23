"""
run_pipeline.py
===============
Flexible CLI for the Mercedes-Benz feedback pipeline. Accepts any CSV schema
via column-mapping flags so no manual renaming is needed.

Usage examples
--------------
  # Default schema (feedback / language_hint / domain)
  python scripts/run_pipeline.py --input data/raw/all_feedback.csv

  # Emotional dataset with native schema
  python scripts/run_pipeline.py \
      --input data/raw/emotional_complaints_200.csv \
      --text-col complaint_text \
      --language-col detected_language \
      --id-col complaint_id

  # Override clustering / output location
  python scripts/run_pipeline.py \
      --input my_data.csv --text-col message \
      --min-cluster-size 30 --output-dir outputs/run_2026_q1

Outputs (in --output-dir, default outputs/):
  * clusters_final.json          one record per cluster
  * record_assignments_final.csv per-record cluster + entities + 2D coords
  * pipeline_stats.json          run metadata
  * embeddings_2d.csv            x/y coords for scatter visualization
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# reuse the well-tested helpers
from scripts.test_run import (
    clean_and_anon, near_dedup, build_re, load_gaz, make_description,
    class_tfidf, heuristic_label, llm_label, llm_extract, merge_similar,
    USE_LLM, LLM_MODEL,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the Mercedes feedback NLP pipeline on any CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Path to the input CSV file")
    p.add_argument("--text-col", default="feedback",
                   help="Column containing complaint text (default: feedback)")
    p.add_argument("--language-col", default="language_hint",
                   help="Column with language code (default: language_hint). "
                        "If missing, all records get 'unk'.")
    p.add_argument("--domain-col", default="domain",
                   help="Column with domain/category (default: domain). "
                        "If missing, all records get 'unknown'.")
    p.add_argument("--id-col", default=None,
                   help="Column with stable record IDs. "
                        "If omitted, IDs are auto-generated.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs",
                   help="Where to write outputs (default: outputs/)")
    p.add_argument("--min-cluster-size", type=int, default=None,
                   help="HDBSCAN min_cluster_size. Auto-tunes by row count "
                        "if not set: <500 -> 6, <5k -> 15, <50k -> 30, else 50")
    p.add_argument("--min-samples", type=int, default=None,
                   help="HDBSCAN min_samples (default: min_cluster_size/3)")
    p.add_argument("--dropping-threshold", type=int, default=None,
                   help="Drop clusters smaller than this after merge "
                        "(default: min_cluster_size * 2/3)")
    p.add_argument("--no-llm", action="store_true",
                   help="Force heuristic labels even if OPENAI_API_KEY is set")
    return p.parse_args()


def auto_min_cluster_size(n: int) -> int:
    if n < 500:   return 6
    if n < 5000:  return 15
    if n < 50000: return 30
    return 50


def main():
    args = parse_args()
    t0 = time.time()

    if not args.input.exists():
        sys.exit(f"ERROR: input file not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_llm = USE_LLM and not args.no_llm
    print(f"[1/10] Load   {args.input}")
    df_in = pd.read_csv(args.input)
    print(f"        rows = {len(df_in):,}, columns = {list(df_in.columns)}")
    print(f"        LLM enrichment: {'ENABLED (' + LLM_MODEL + ')' if use_llm else 'DISABLED'}")

    # ---- column mapping with friendly errors ----
    if args.text_col not in df_in.columns:
        sys.exit(f"ERROR: --text-col '{args.text_col}' not in CSV. "
                 f"Available columns: {list(df_in.columns)}")
    df = df_in.copy()
    df["_text"]     = df[args.text_col]
    df["_language"] = df[args.language_col] if args.language_col in df.columns else "unk"
    df["_domain"]   = df[args.domain_col]   if args.domain_col   in df.columns else "unknown"
    df["_record_id"] = (df[args.id_col].astype(str)
                        if args.id_col and args.id_col in df.columns
                        else [f"R{i:08d}" for i in range(len(df))])

    print("[2/10] Clean + anonymize")
    df["clean_text"] = df["_text"].map(clean_and_anon)
    df = df[df["clean_text"].str.len().between(5, 2000)].reset_index(drop=True)

    print("[3/10] Exact dedup")
    before = len(df); df = df.drop_duplicates("clean_text").reset_index(drop=True)
    print(f"        removed {before-len(df):,}; remaining {len(df):,}")

    print("[4/10] Near-duplicate dedup")
    before = len(df); df = near_dedup(df, "clean_text", 0.97)
    print(f"        removed {before-len(df):,}; remaining {len(df):,}")

    if len(df) < 20:
        sys.exit(f"ERROR: only {len(df)} records after dedup; need >=20 to cluster.")

    print("[5/10] Embed (TF-IDF -> SVD, normalized)")
    tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                          sublinear_tf=True, max_features=30000,
                          stop_words="english")
    X = tfv.fit_transform(df["clean_text"])
    n_comp = min(256, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(X)
    emb = normalize(emb, norm="l2").astype(np.float32)
    print(f"        emb shape = {emb.shape}")

    print("[6/10] UMAP -> 15d (clustering space) + 2d (visualization)")
    reducer_15 = umap.UMAP(n_neighbors=min(30, len(df) - 1),
                           n_components=15, min_dist=0.0,
                           metric="cosine", random_state=42)
    red = reducer_15.fit_transform(emb).astype(np.float32)
    reducer_2 = umap.UMAP(n_neighbors=min(15, len(df) - 1),
                          n_components=2, min_dist=0.1,
                          metric="cosine", random_state=42)
    coords_2d = reducer_2.fit_transform(emb).astype(np.float32)

    mcs = args.min_cluster_size or auto_min_cluster_size(len(df))
    ms  = args.min_samples or max(2, mcs // 3)
    drop_thr = args.dropping_threshold or max(3, int(mcs * 2 / 3))
    print(f"[7/10] HDBSCAN  min_cluster_size={mcs}, min_samples={ms}")
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                         metric="euclidean", cluster_selection_method="eom")
    labels = cl.fit_predict(red)
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"        clusters = {n_cl}, noise = {n_noise:,} ({n_noise / len(labels):.1%})")
    df["cluster_id"] = labels
    df["cluster_prob"] = cl.probabilities_
    df["x_2d"] = coords_2d[:, 0]
    df["y_2d"] = coords_2d[:, 1]

    print("[8/10] Entity extraction")
    import re as _re
    comp_re = build_re(load_gaz("artifacts/components_gazetteer.txt"))
    fail_re = build_re(load_gaz("artifacts/failure_types_gazetteer.txt"))
    comps, fails, syms, descs = [], [], [], []
    for t in df["clean_text"]:
        cm = comp_re.search(t) if comp_re else None
        fm = fail_re.search(t) if fail_re else None
        c = cm.group(1).lower() if cm else None
        f = fm.group(1).lower() if fm else None
        s = None
        if f:
            for sent in _re.split(r"(?<=[.!?])\s+", t):
                if f in sent.lower():
                    s = sent.strip()[:200]; break
        comps.append(c); fails.append(f); syms.append(s)
        descs.append(make_description(c, f, s))
    df["component"], df["failure_type"], df["symptom"], df["description"] = comps, fails, syms, descs

    if use_llm:
        miss_idx = [i for i, (c, f) in enumerate(zip(comps, fails)) if not (c and f)][:200]
        print(f"[8b]   LLM fill for {len(miss_idx)} records")
        for s in range(0, len(miss_idx), 25):
            chunk = miss_idx[s:s + 25]
            results = llm_extract([df.at[i, "clean_text"] for i in chunk])
            for i, r in zip(chunk, results):
                for k in ("component", "failure_type", "symptom", "description"):
                    if not df.at[i, k]:
                        df.at[i, k] = r.get(k)

    print(f"[9/10] c-TF-IDF labels{' + LLM' if use_llm else ' (heuristic)'}")
    pseudo = {int(c): " ".join(df.loc[df["cluster_id"] == c, "clean_text"])
              for c in df["cluster_id"].unique() if c != -1}
    keywords = class_tfidf(pseudo, top_k=12)
    cluster_rows = []
    for cid, kw in keywords.items():
        members = df[df["cluster_id"] == cid]
        examples = (members.sort_values("cluster_prob", ascending=False)
                          ["clean_text"].head(5).tolist())
        comp_top = Counter(members["component"].dropna()).most_common(1)
        fail_top = Counter(members["failure_type"].dropna()).most_common(1)
        comp = comp_top[0][0] if comp_top else None
        fail = fail_top[0][0] if fail_top else None
        lab = llm_label(comp, fail, kw, examples) if use_llm else heuristic_label(comp, fail, kw, examples)
        cluster_rows.append({
            "cluster_id": cid, "size": int(len(members)),
            "label": lab["label"], "summary": lab.get("summary", ""),
            "keywords": kw,
            "top_component": comp, "top_failure": fail,
            "domains": sorted(members["_domain"].unique().tolist()),
            "languages": sorted(members["_language"].unique().tolist()),
            "example_complaints": examples,
        })

    print("[10/10] Cosine-merge similar clusters")
    merge_map = merge_similar(emb, df["cluster_id"].to_numpy(), thr=0.92)
    df["cluster_id_final"] = df["cluster_id"].map(
        lambda c: merge_map.get(int(c), c) if c != -1 else -1)

    final = []
    for new_cid in sorted({v for v in merge_map.values()}):
        srcs = [k for k, v in merge_map.items() if v == new_cid]
        sub = [r for r in cluster_rows if r["cluster_id"] in srcs]
        if not sub: continue
        sub.sort(key=lambda r: r["size"], reverse=True)
        primary = sub[0]
        size = int((df["cluster_id_final"] == new_cid).sum())
        if size < drop_thr: continue
        merged_kw, seen = [], set()
        for r in sub:
            for k in r["keywords"]:
                if k not in seen:
                    merged_kw.append(k); seen.add(k)
        all_lang = sorted({l for r in sub for l in r["languages"]})
        all_dom  = sorted({d for r in sub for d in r["domains"]})
        examples = []
        for r in sub: examples.extend(r["example_complaints"][:2])
        final.append({
            "cluster_id": int(new_cid),
            "size": size,
            "label": primary["label"],
            "summary": primary["summary"],
            "keywords": merged_kw[:15],
            "top_component": primary["top_component"],
            "top_failure": primary["top_failure"],
            "domains": all_dom,
            "languages": all_lang,
            "example_complaints": examples[:6],
            "merged_from": srcs,
        })
    final.sort(key=lambda r: r["size"], reverse=True)

    # rename internal columns for output
    out_df = df.rename(columns={"_record_id": "record_id",
                                "_language": "language",
                                "_domain": "domain",
                                "_text": "original_text"})
    keep_cols = ["record_id", "domain", "language", "original_text", "clean_text",
                 "cluster_id", "cluster_prob", "cluster_id_final",
                 "component", "failure_type", "symptom", "description",
                 "x_2d", "y_2d"]
    out_df = out_df[[c for c in keep_cols if c in out_df.columns]]

    (args.output_dir / "clusters_final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    out_df.to_csv(args.output_dir / "record_assignments_final.csv", index=False)
    out_df[["record_id", "x_2d", "y_2d", "cluster_id_final"]].to_csv(
        args.output_dir / "embeddings_2d.csv", index=False)

    stats = {
        "input_file": str(args.input),
        "total_records_input": int(len(df_in)),
        "records_after_clean_dedup": int(len(df)),
        "n_clusters_raw": n_cl,
        "n_clusters_final": len(final),
        "n_noise": n_noise,
        "languages_seen": sorted(df["_language"].astype(str).unique().tolist()),
        "domains_seen": sorted(df["_domain"].astype(str).unique().tolist()),
        "llm_enabled": use_llm,
        "llm_model": LLM_MODEL if use_llm else None,
        "min_cluster_size": mcs,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (args.output_dir / "pipeline_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print()
    print(f"DONE in {stats['runtime_seconds']}s")
    print(f"  Final clusters: {len(final)}")
    print(f"  Outputs in: {args.output_dir}")
    for c in final[:5]:
        print(f"    #{c['cluster_id']:>2} size={c['size']:>4}  {c['label']}")


if __name__ == "__main__":
    main()
