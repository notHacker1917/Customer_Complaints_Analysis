"""
preprocessing.py
================
Stage 1 of the pipeline.

Responsibilities
----------------
1.  Load raw CSV files for each domain (interior / powertrain / display).
2.  Clean text:
        - fix encoding (ftfy)
        - strip HTML, URLs, control chars
        - normalize whitespace, quotes, dashes
        - drop emojis (configurable)
3.  Detect language per record (langdetect / fasttext fallback).
4.  Anonymize PII with Microsoft Presidio (PERSON, EMAIL, PHONE, VIN, ...).
5.  Optional: machine-translate non-English to English with a Helsinki-NLP
    multilingual model so downstream embeddings are semantically aligned.
6.  Drop exact and near-duplicate records (MinHash/SimHash style via
    character-shingle hashing).
7.  Emit a canonical processed parquet:

        record_id | domain | source_lang | original_text | clean_text | text_for_embed
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List

import ftfy
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from unidecode import unidecode

from .utils import load_config, save_parquet, stable_hash, ensure_dirs

# -------------------------------------------------------------------- regexes
_URL_RE        = re.compile(r"https?://\S+|www\.\S+", flags=re.I)
_HTML_RE       = re.compile(r"<[^>]+>")
_MULTI_WS_RE   = re.compile(r"\s+")
_CTRL_RE       = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_DUP_PUNCT_RE  = re.compile(r"([!?\.\-_,])\1{2,}")  # !!!! -> !
_EMOJI_RE      = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002700-\U000027BF"
    "\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE
)


# ------------------------------------------------------------ text utilities
def clean_text(raw: str) -> str:
    """Normalize a single string. Safe on None / non-str."""
    if raw is None or not isinstance(raw, str):
        return ""
    t = ftfy.fix_text(raw)
    t = _HTML_RE.sub(" ", t)
    t = _URL_RE.sub(" ", t)
    t = _CTRL_RE.sub(" ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = unicodedata.normalize("NFKC", t)
    t = _DUP_PUNCT_RE.sub(r"\1", t)
    t = _MULTI_WS_RE.sub(" ", t).strip()
    return t


def is_valid(text: str, min_chars: int, max_chars: int) -> bool:
    if not text:
        return False
    if len(text) < min_chars or len(text) > max_chars:
        return False
    # at least one alphabetic character
    return any(c.isalpha() for c in text)


# ----------------------------------------------------- language detection
def _safe_detect(text: str) -> str:
    """Returns ISO-639-1 code or 'unk'."""
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 42
        return detect(text)
    except Exception:
        return "unk"


def detect_languages(texts: List[str]) -> List[str]:
    return [_safe_detect(t) if t else "unk" for t in tqdm(texts, desc="lang-detect")]


# ----------------------------------------------------- anonymization (PII)
class Anonymizer:
    """Wrapper around Presidio. Lazily loaded so import is cheap."""

    def __init__(self, entities: List[str]):
        from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
        from presidio_anonymizer import AnonymizerEngine

        self.entities = entities
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # Custom recognizers Presidio doesn't ship with
        vin_pattern = Pattern(name="vin",
                              regex=r"\b[A-HJ-NPR-Z0-9]{17}\b", score=0.85)
        plate_pattern = Pattern(name="plate",
                                regex=r"\b[A-Z]{1,3}-?[A-Z]{1,2}\s?\d{1,4}\b",
                                score=0.6)
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(supported_entity="VIN", patterns=[vin_pattern]))
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(supported_entity="LICENSE_PLATE",
                              patterns=[plate_pattern]))

    def __call__(self, text: str, language: str = "en") -> str:
        if not text:
            return text
        try:
            results = self.analyzer.analyze(
                text=text, language=language, entities=self.entities
            )
            if not results:
                return text
            return self.anonymizer.anonymize(text=text,
                                             analyzer_results=results).text
        except Exception as e:        # never let PII pass on error
            logger.warning(f"Anonymizer failed ({e}); falling back to regex")
            return self._regex_fallback(text)

    @staticmethod
    def _regex_fallback(text: str) -> str:
        text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "<EMAIL>", text)
        text = re.sub(r"\+?\d[\d\-\s]{7,}\d", "<PHONE>", text)
        text = re.sub(r"\b[A-HJ-NPR-Z0-9]{17}\b", "<VIN>", text)
        return text


# ----------------------------------------------------- translation (mul->en)
class Translator:
    """Lazy translator using Helsinki-NLP/opus-mt-mul-en."""

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-mul-en",
                 device: str | None = None):
        from transformers import MarianMTModel, MarianTokenizer
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading translator {model_name} on {self.device}")
        self.tok = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def translate(self, texts: List[str], batch_size: int = 32) -> List[str]:
        import torch

        out: List[str] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="translate"):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, return_tensors="pt",
                           padding=True, truncation=True, max_length=256
                           ).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=256)
            out.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return out


# --------------------------------------------------- duplicate detection
def shingle_hash(text: str, k: int = 5) -> set[int]:
    text = re.sub(r"\s+", " ", text.lower())
    return {hash(text[i:i + k]) for i in range(max(len(text) - k + 1, 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def drop_near_duplicates(df: pd.DataFrame,
                         text_col: str = "clean_text",
                         threshold: float = 0.97) -> pd.DataFrame:
    """O(n) approximate dedup via bucketing on first-shingle prefix.

    For 100k rows this keeps memory bounded; exact pairwise is O(n^2).
    """
    seen_buckets: dict[int, list[set[int]]] = {}
    keep_mask = np.ones(len(df), dtype=bool)
    for i, txt in enumerate(tqdm(df[text_col].tolist(), desc="near-dedup")):
        sh = shingle_hash(txt)
        if not sh:
            continue
        bkey = min(sh)            # cheap LSH proxy
        bucket = seen_buckets.setdefault(bkey, [])
        if any(jaccard(sh, prev) >= threshold for prev in bucket):
            keep_mask[i] = False
        else:
            bucket.append(sh)
    return df[keep_mask].reset_index(drop=True)


# ============================================================== main entry
def run(cfg_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)

    pp = cfg["preprocessing"]
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    out_path = Path(cfg["paths"]["processed_data_dir"]) / "feedback_clean.parquet"

    # ---- load all domains -------------------------------------------------
    frames = []
    for domain, dcfg in cfg["domains"].items():
        fp = raw_dir / dcfg["file"]
        if not fp.exists():
            logger.warning(f"Skipping missing file {fp}")
            continue
        df = pd.read_csv(fp)
        df = df.rename(columns={dcfg["text_col"]: "original_text"})
        df["domain"] = domain
        frames.append(df[["domain", "original_text"]])
        logger.info(f"{domain}: loaded {len(df):,} rows")

    if not frames:
        raise RuntimeError("No raw input files found. Place CSVs under "
                           f"{raw_dir.resolve()}")

    df = pd.concat(frames, ignore_index=True)

    # ---- basic clean ------------------------------------------------------
    df["clean_text"] = [clean_text(t) for t in tqdm(df["original_text"],
                                                    desc="clean")]
    df = df[df["clean_text"].apply(
        lambda t: is_valid(t, pp["min_chars"], pp["max_chars"]))].reset_index(drop=True)
    logger.info(f"After cleaning + validation: {len(df):,}")

    # ---- exact dedup ------------------------------------------------------
    if pp["drop_exact_duplicates"]:
        before = len(df)
        df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
        logger.info(f"Exact dedup removed {before - len(df):,}")

    # ---- language detect --------------------------------------------------
    if pp["language_detection"]:
        df["source_lang"] = detect_languages(df["clean_text"].tolist())
    else:
        df["source_lang"] = "unk"

    # ---- anonymize --------------------------------------------------------
    if cfg["anonymization"]["enabled"]:
        anon = Anonymizer(cfg["anonymization"]["entities"])
        df["clean_text"] = [
            anon(t, language="en" if l not in ("en", "de") else l)
            for t, l in tqdm(zip(df["clean_text"], df["source_lang"]),
                             total=len(df), desc="anonymize")
        ]

    # ---- translate to English for cross-lingual embedding consistency ----
    if pp["translate_to_english"]:
        tr = Translator(pp["translation_model"])
        non_en_mask = df["source_lang"] != "en"
        translated = df["clean_text"].copy()
        if non_en_mask.any():
            translated.loc[non_en_mask] = tr.translate(
                df.loc[non_en_mask, "clean_text"].tolist())
        df["text_for_embed"] = translated
    else:
        df["text_for_embed"] = df["clean_text"]

    # ---- near-duplicate removal ------------------------------------------
    if pp["drop_near_duplicates"]:
        before = len(df)
        df = drop_near_duplicates(df, "text_for_embed",
                                  pp["near_duplicate_threshold"])
        logger.info(f"Near-dedup removed {before - len(df):,}")

    # ---- assign stable IDs -----------------------------------------------
    df["record_id"] = df["clean_text"].apply(stable_hash)
    df = df[["record_id", "domain", "source_lang",
             "original_text", "clean_text", "text_for_embed"]]

    save_parquet(df, out_path)
    return df


if __name__ == "__main__":
    run()
