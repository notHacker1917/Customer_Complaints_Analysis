"""
test_run.py
===========
End-to-end pipeline. Same logical stages as the production code, but with
TF-IDF + TruncatedSVD as a lightweight stand-in for the bge-large-en-v1.5
embedding step (avoids the ~2 GB torch + sentence-transformers download).

When OPENAI_API_KEY is set the script ALSO uses the OpenAI API for:
  * Stage 8b: LLM entity extraction on records the rule pass missed
  * Stage 9b: LLM-generated human-readable cluster labels & summaries
Without the key, both stages fall back to deterministic heuristics that
construct readable labels from the extracted component + failure type.

Pipeline:
    1. Load CSV
    2. Clean + ftfy
    3. PII anonymize (regex)
    4. Exact + near-duplicate removal
    5. Embed (TfidfVectorizer + TruncatedSVD -> 256d, L2-normalized)
    6. UMAP -> 15d
    7. HDBSCAN
    8. Rule-based entity extraction (component / failure_type / symptom / description)
    9. c-TF-IDF cluster keywords + label
   10. Cosine-merge similar clusters
   11. Write outputs/clusters_final.json + record_assignments_final.csv
"""
from __future__ import annotations
import json, os, re, sys, time
from collections import Counter
from pathlib import Path

import ftfy
import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------- OpenAI (optional)
USE_LLM = bool(os.environ.get("OPENAI_API_KEY"))
LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
_openai_client = None
def get_client():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI()
    return _openai_client

# ---------------------- 2/3. clean & anonymize -----------------------
URL_RE   = re.compile(r"https?://\S+|www\.\S+", flags=re.I)
HTML_RE  = re.compile(r"<[^>]+>")
WS_RE    = re.compile(r"\s+")
DUP_RE   = re.compile(r"([!?.\-_,])\1{2,}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")
VIN_RE   = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
PLATE_RE = re.compile(r"\b[A-Z]{1,3}-?[A-Z]{1,2}\s?\d{1,4}\b")

def clean_and_anon(t):
    if not isinstance(t, str): return ""
    t = ftfy.fix_text(t)
    t = HTML_RE.sub(" ", t); t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub("<EMAIL>", t)
    t = PHONE_RE.sub("<PHONE>", t)
    t = VIN_RE.sub("<VIN>", t)
    t = PLATE_RE.sub("<PLATE>", t)
    t = DUP_RE.sub(r"\1", t)
    t = WS_RE.sub(" ", t).strip()
    return t

# ---------------------- 4. near-dup ----------------------------------
def shingles(text, k=5):
    text = re.sub(r"\s+", " ", text.lower())
    return {hash(text[i:i+k]) for i in range(max(len(text)-k+1, 1))}

def near_dedup(df, col="clean_text", thr=0.97):
    seen = {}; keep = np.ones(len(df), dtype=bool)
    for i, t in enumerate(df[col].tolist()):
        sh = shingles(t)
        if not sh: continue
        bkey = min(sh)
        bucket = seen.setdefault(bkey, [])
        if any(len(sh & p)/max(len(sh|p),1) >= thr for p in bucket):
            keep[i] = False
        else:
            bucket.append(sh)
    return df[keep].reset_index(drop=True)

# ---------------------- 8. rule entity extract ------------------------
def load_gaz(p):
    fp = ROOT / p
    if not fp.exists(): return []
    return [ln.strip().lower() for ln in fp.open("r", encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]

def build_re(terms):
    if not terms: return None
    terms = sorted(set(terms), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)

def make_description(component, failure_type, symptom):
    """Build a one-line canonical phrase from extracted entities."""
    if component and failure_type:
        return f"{component} - {failure_type}"
    if component:
        return f"{component} - issue reported"
    if failure_type:
        return f"unidentified component - {failure_type}"
    return None

# --------- LLM entity extraction -----------
EXTRACT_PROMPT = """You are extracting structured failure information from
vehicle customer-feedback notes. For each text, return a JSON object:
{"component": "<short noun phrase or null>",
 "failure_type": "<short verb/noun or null>",
 "symptom": "<<=120 chars or null>",
 "description": "<one-sentence canonical summary, <=200 chars or null>"}
Rules: lowercase, no trailing punctuation, do not invent details.
Return STRICT JSON: {"results":[ {...}, {...}, ... ]} in the same order.

Texts:
"""
def llm_extract(texts):
    if not USE_LLM or not texts: return [{} for _ in texts]
    try:
        client = get_client()
        prompt = EXTRACT_PROMPT + json.dumps(texts, ensure_ascii=False, indent=2)
        resp = client.chat.completions.create(
            model=LLM_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role":"user","content":prompt}])
        data = json.loads(resp.choices[0].message.content)
        return data.get("results", [{} for _ in texts])
    except Exception as e:
        print(f"  ! llm_extract failed: {e}")
        return [{} for _ in texts]

# ---------------------- 9. c-TF-IDF + label --------------------------
def class_tfidf(pseudo_docs, top_k=12):
    cids = sorted(pseudo_docs.keys())
    corpus = [pseudo_docs[c] for c in cids]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9,
                          sublinear_tf=True, stop_words="english", max_features=20000)
    X = vec.fit_transform(corpus)
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for row, cid in enumerate(cids):
        s = X[row].toarray().ravel()
        top = s.argsort()[::-1][:top_k]
        out[cid] = [terms[i] for i in top if s[i] > 0]
    return out

LABEL_PROMPT = """You are labelling a cluster of Mercedes-Benz customer
complaints. Given the dominant component, dominant failure type, keywords
and example complaints, return a JSON object:
{"label":"<5-9 word concise failure topic, sentence case>",
 "summary":"<one sentence describing what unifies these complaints>"}
Be specific and engineering-actionable.

Dominant component: {comp}
Dominant failure: {fail}
Keywords: {kw}
Examples:
{ex}
"""
def llm_label(component, failure, keywords, examples):
    if not USE_LLM:
        return heuristic_label(component, failure, keywords, examples)
    try:
        client = get_client()
        prompt = LABEL_PROMPT.format(
            comp=component or "unknown",
            fail=failure or "unknown",
            kw=", ".join(keywords),
            ex="\n".join(f"- {e[:240]}" for e in examples[:5]))
        resp = client.chat.completions.create(
            model=LLM_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role":"user","content":prompt}])
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"  ! llm_label failed: {e}")
        return heuristic_label(component, failure, keywords, examples)

def heuristic_label(component, failure, keywords, examples):
    """Engineer-readable label built from extracted entities and keywords."""
    if component and failure:
        label = f"{component.capitalize()} - {failure}"
    elif component:
        # find the most action-y keyword that isn't the component itself
        comp_words = set(component.lower().split())
        action = next((k for k in keywords
                       if not any(w in comp_words for w in k.split())
                       and len(k) > 3 and not k.isdigit()), None)
        label = f"{component.capitalize()} - {action}" if action else component.capitalize()
    elif failure:
        label = f"Unidentified component - {failure}"
    else:
        label = " / ".join(k for k in keywords[:3] if not k.isdigit()) or "Unlabeled cluster"
    summary = (f"Cluster centered on '{component}' failures of type '{failure}'."
               if component and failure else
               f"Cluster grouped by keywords: {', '.join(keywords[:4])}.")
    return {"label": label, "summary": summary}

# ---------------------- 10. centroid-merge ---------------------------
def merge_similar(emb, labels, thr=0.92):
    cents = {}
    for c in np.unique(labels):
        if c == -1: continue
        v = emb[labels == c].mean(axis=0)
        v = v / (np.linalg.norm(v)+1e-12)
        cents[int(c)] = v.astype(np.float32)
    if not cents: return {}
    ids = sorted(cents.keys())
    M = np.stack([cents[c] for c in ids])
    sim = cosine_similarity(M); np.fill_diagonal(sim, 0.0)
    parent = {c: c for c in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if ra < rb: parent[rb] = ra
        else: parent[ra] = rb
    for i, a in enumerate(ids):
        for j in range(i+1, len(ids)):
            if sim[i, j] >= thr: union(a, ids[j])
    return {c: find(c) for c in ids}

# =====================================================================
def main():
    t_all = time.time()
    csv_path = ROOT / "data/raw/all_feedback.csv"
    out_dir  = ROOT / "outputs"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/10] Load   {csv_path}")
    df_in = pd.read_csv(csv_path)
    print(f"        rows = {len(df_in):,}")
    print(f"        LLM enrichment: {'ENABLED (' + LLM_MODEL + ')' if USE_LLM else 'DISABLED (set OPENAI_API_KEY to enable)'}")

    print("[2/10] Clean + anonymize")
    df = df_in.copy()
    df["clean_text"] = df["feedback"].map(clean_and_anon)
    df = df[df["clean_text"].str.len().between(5, 2000)].reset_index(drop=True)

    print("[3/10] Exact dedup")
    before = len(df); df = df.drop_duplicates("clean_text").reset_index(drop=True)
    print(f"        removed {before-len(df):,}; remaining {len(df):,}")

    print("[4/10] Near-duplicate dedup")
    before = len(df); df = near_dedup(df, "clean_text", 0.97)
    print(f"        removed {before-len(df):,}; remaining {len(df):,}")

    print("[5/10] Embed (TF-IDF -> SVD 256d, normalized)")
    tfv = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95,
                          sublinear_tf=True, max_features=30000,
                          stop_words="english")
    X = tfv.fit_transform(df["clean_text"])
    n_comp = min(256, X.shape[1]-1, X.shape[0]-1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(X)
    emb = normalize(emb, norm="l2").astype(np.float32)
    print(f"        emb shape = {emb.shape}")

    print("[6/10] UMAP -> 15d")
    reducer = umap.UMAP(n_neighbors=30, n_components=15, min_dist=0.0,
                        metric="cosine", random_state=42, verbose=False)
    red = reducer.fit_transform(emb).astype(np.float32)

    print("[7/10] HDBSCAN")
    cl = hdbscan.HDBSCAN(min_cluster_size=6, min_samples=2,
                        metric="euclidean", cluster_selection_method="eom")
    labels = cl.fit_predict(red)
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"        clusters = {n_cl}, noise = {n_noise:,} ({n_noise/len(labels):.1%})")
    df["cluster_id"] = labels
    df["cluster_prob"] = cl.probabilities_

    print("[8/10] Rule-based entity extraction")
    comp_re = build_re(load_gaz("artifacts/components_gazetteer.txt"))
    fail_re = build_re(load_gaz("artifacts/failure_types_gazetteer.txt"))
    comps, fails, syms, descs = [], [], [], []
    for t in df["clean_text"]:
        cm = comp_re.search(t) if comp_re else None
        fm = fail_re.search(t) if fail_re else None
        c = cm.group(1).lower() if cm else None
        f = fm.group(1).lower() if fm else None
        # symptom = sentence containing the failure word
        s = None
        if f:
            for sent in re.split(r"(?<=[.!?])\s+", t):
                if f in sent.lower():
                    s = sent.strip()[:200]; break
        comps.append(c); fails.append(f); syms.append(s)
        descs.append(make_description(c, f, s))
    df["component"], df["failure_type"], df["symptom"], df["description"] = comps, fails, syms, descs

    if USE_LLM:
        miss_idx = [i for i, (c, f) in enumerate(zip(comps, fails)) if not (c and f)][:200]
        print(f"[8b]   LLM fill for {len(miss_idx)} records (capped at 200)")
        for s in range(0, len(miss_idx), 25):
            chunk = miss_idx[s:s+25]
            results = llm_extract([df.at[i, "clean_text"] for i in chunk])
            for i, r in zip(chunk, results):
                if not df.at[i, "component"]: df.at[i, "component"] = r.get("component")
                if not df.at[i, "failure_type"]: df.at[i, "failure_type"] = r.get("failure_type")
                if not df.at[i, "symptom"]: df.at[i, "symptom"] = r.get("symptom")
                if not df.at[i, "description"]: df.at[i, "description"] = r.get("description")

    print("[9/10] c-TF-IDF labels" + (" + LLM" if USE_LLM else " (heuristic)"))
    pseudo = {int(c): " ".join(df.loc[df["cluster_id"]==c, "clean_text"])
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
        lab = llm_label(comp, fail, kw, examples)
        cluster_rows.append({
            "cluster_id": cid,
            "size": int(len(members)),
            "label": lab["label"],
            "summary": lab.get("summary",""),
            "keywords": kw,
            "top_component": comp,
            "top_failure": fail,
            "domains": sorted(members["domain"].unique().tolist()),
            "languages": sorted(members["language_hint"].unique().tolist()),
            "example_complaints": examples,
        })

    print("[10/10] Cosine-merge similar clusters")
    merge_map = merge_similar(emb, df["cluster_id"].to_numpy(), thr=0.92)
    df["cluster_id_final"] = df["cluster_id"].map(
        lambda c: merge_map.get(int(c), c) if c != -1 else -1)

    final = []
    for new_cid in sorted({v for v in merge_map.values()}):
        srcs = [k for k, v in merge_map.items() if v == new_cid]
        sub  = [r for r in cluster_rows if r["cluster_id"] in srcs]
        if not sub: continue
        sub.sort(key=lambda r: r["size"], reverse=True)
        primary = sub[0]
        size = int((df["cluster_id_final"] == new_cid).sum())
        if size < 10: continue
        merged_kw, seen = [], set()
        for r in sub:
            for k in r["keywords"]:
                if k not in seen:
                    merged_kw.append(k); seen.add(k)
        all_lang = sorted({l for r in sub for l in r["languages"]})
        all_dom  = sorted({d for r in sub for d in r["domains"]})
        examples = []
        for r in sub: examples.extend(r["example_complaints"][:2])
        examples = examples[:6]
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
            "example_complaints": examples,
            "merged_from": srcs,
        })
    final.sort(key=lambda r: r["size"], reverse=True)

    (out_dir / "clusters_final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out_dir / "record_assignments_final.csv", index=False)

    stats = {
        "total_records_input": int(len(df_in)),
        "records_after_clean_dedup": int(len(df)),
        "n_clusters_raw": n_cl,
        "n_clusters_final": len(final),
        "n_noise": n_noise,
        "languages_seen": sorted(df["language_hint"].unique().tolist()),
        "domains_seen": sorted(df["domain"].unique().tolist()),
        "llm_enabled": USE_LLM,
        "llm_model": LLM_MODEL if USE_LLM else None,
        "runtime_seconds": round(time.time() - t_all, 2),
    }
    (out_dir / "pipeline_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print()
    print(f"DONE in {stats['runtime_seconds']}s")
    print(f"  final clusters: {len(final)}")
    for c in final[:5]:
        print(f"  #{c['cluster_id']:>2} size={c['size']:>3}  "
              f"comp={(c['top_component'] or '-')[:18]:<18}  label='{c['label']}'")

if __name__ == "__main__":
    main()
