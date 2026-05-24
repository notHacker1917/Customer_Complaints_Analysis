# Multilingual Customer Feedback NLP Pipeline

End-to-end Python pipeline that converts ~109k raw, multilingual customer
complaints into structured, engineer-ready failure insights.

``` 
raw csv  ─►  preprocess  ─►  embed (bge-large-en-v1.5)  ─►  UMAP  ─►  HDBSCAN
                                                                       │
                                            ┌──────────────────────────┘
                                            ▼
                                 entity extraction (rule + LLM)
                                            │
                                            ▼
                              cluster interpretation (c-TF-IDF + LLM)
                                            │
                                            ▼
                                 refinement (cosine merge, prune)
                                            │
                                            ▼
                            outputs/clusters_final.{parquet,json}
```

## Project layout

```
Customer_Complaints_Analysis/
├── config.yaml              # central knobs for every stage
├── main.py                  # orchestrator: `python main.py --stages ...`
├── requirements.txt
├── artifacts/
│   ├── components_gazetteer.txt
│   └── failure_types_gazetteer.txt
├── data/
│   ├── raw/                 # drop CSVs here (interior_, powertrain_, display_)
│   ├── processed/           # cleaned + entity parquets
│   └── embeddings/          # numpy arrays + index
├── outputs/                 # final clusters + per-record assignment
└── src/
    ├── utils.py
    ├── preprocessing.py     # 1. clean, dedup, language-detect, anonymize, translate
    ├── embeddings.py        # 2. bge-large-en-v1.5 encoding
    ├── dim_reduction.py     # 3. UMAP 1024 → 15
    ├── clustering.py        # 4. HDBSCAN
    ├── entity_extraction.py # 5. rule + LLM (component / failure_type / symptom)
    ├── interpretation.py    # 6. c-TF-IDF + LLM label per cluster
    └── refinement.py        # 7. cosine-merge similar clusters, prune small ones
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# LLM creds for entity extraction + cluster labelling
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

## Input format

Place three CSVs under `data/raw/`:

| File                          | Rows   | Languages                |
|-------------------------------|-------:|--------------------------|
| `interior_feedback.csv`       | 75,000 | English                  |
| `powertrain_feedback.csv`     |  5,000 | English + German         |
| `display_feedback.csv`        | 29,000 | EN, DE, DA, TR, PL, …    |

Each file must have at least one column named **`feedback`** (configurable in
`config.yaml → domains.<name>.text_col`).  Any other columns (vehicle id,
timestamp, etc.) are preserved through processing but ignored by the model.

## Running

```bash
# full pipeline
python main.py

# only re-run the labelling/refinement stages after editing the LLM prompt
python main.py --stages interpret,refine

# quick smoke test using rule-only entity extraction (no LLM cost)
# (set entity_extraction.provider: "rule_only" in config.yaml first)
python main.py
```

## Output schema (`outputs/clusters_final.json`)

```json
{
  "cluster_id":   42,
  "size":         318,
  "label":        "MBUX touchscreen freezes during navigation",
  "summary":      "Customers report the central display becomes unresponsive ...",
  "keywords":     ["touchscreen", "freeze", "navigation", "mbux", "reboot", ...],
  "top_component":"infotainment head unit",
  "example_complaints": [
     "Screen went black while using nav, had to restart the car ...",
     "MBUX gefriert manchmal mitten in der Navigation ...",
     ...
  ],
  "domains":      ["display_infotainment"],
  "languages":    ["en", "de", "pl"],
  "merged_from":  [42, 87]
}
```

## Design notes

* **Cross-lingual consistency.** `bge-large-en-v1.5` is English-only, so
  non-English records are machine-translated to English with
  `Helsinki-NLP/opus-mt-mul-en` *before* embedding. This is much cheaper than
  switching to a multilingual model and preserves the high quality of `bge-large`.
  If you'd rather keep originals, swap to `BAAI/bge-m3` and disable
  `preprocessing.translate_to_english`.
* **Why UMAP → HDBSCAN.** Cosine in 1024-D is dominated by noise; dropping to
  ~15 dense dimensions makes HDBSCAN's density estimates meaningful and
  accelerates clustering by ~50× on 100k records.
* **Rule-first entity extraction.** A regex pass over component/failure
  gazetteers handles the easy 70-90% of records for free; only the ambiguous
  remainder is sent to the LLM, cutting API spend by an order of magnitude.
* **c-TF-IDF labels.** TF-IDF computed across cluster pseudo-documents (à la
  BERTopic) surfaces *distinguishing* terms, not just frequent ones. The LLM
  then turns those keywords + 5 representative complaints into a short
  human-readable label.
* **Refinement.** HDBSCAN often over-segments. We compute centroid cosine
  similarity in the original embedding space and union-find merge any
  clusters above 0.92 similarity; clusters smaller than 25 records are
  dropped as noise.

## Scaling notes

* Tested target: 100k+ records on a single A10/A100 GPU. Embedding is
  the dominant cost (~6-10 min on A100, ~30-40 min on CPU).
* For >1M records, switch to `faiss` for the dedup and centroid-merge steps,
  and use `cuml`'s GPU UMAP/HDBSCAN.
* Every stage writes its output to disk so you can iterate on later stages
  without rerunning embedding.
