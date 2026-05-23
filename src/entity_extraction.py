"""
entity_extraction.py
====================
Stage 5 — turn each free-text complaint into structured fields:

    {
      "component":    "infotainment head unit",
      "failure_type": "freeze",
      "symptom":      "screen goes black during navigation",
      "description":  "<short normalized phrase>"
    }

Strategy: rule-first, LLM-fallback.
    1. Try to match the text against gazetteers (component_list, failure_type_list)
       using fast regex / whole-word matching.  If both are found we accept
       the rule output and skip the LLM call (saves 90%+ on cost).
    2. Otherwise, batch the remaining records and send them to an LLM with
       a strict JSON schema.  Anthropic Claude or OpenAI both supported.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from .utils import batched, load_config, load_parquet, save_parquet


# --------------------------------------------------------------- data class
@dataclass
class Entities:
    component: Optional[str] = None
    failure_type: Optional[str] = None
    symptom: Optional[str] = None
    description: Optional[str] = None


# ---------------------------------------------------------------- gazetteer
def load_gazetteer(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        logger.warning(f"Gazetteer not found at {p} — using built-in defaults")
        return []
    with p.open("r", encoding="utf-8") as f:
        return [ln.strip().lower() for ln in f if ln.strip()
                and not ln.startswith("#")]


def _build_regex(terms: List[str]) -> Optional[re.Pattern]:
    if not terms:
        return None
    # longest first so "head unit" wins over "unit"
    terms = sorted(set(terms), key=len, reverse=True)
    pat = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
    return re.compile(pat, flags=re.I)


def rule_extract(text: str,
                 comp_re: Optional[re.Pattern],
                 fail_re: Optional[re.Pattern]) -> Entities:
    e = Entities()
    if comp_re:
        m = comp_re.search(text)
        if m:
            e.component = m.group(1).lower()
    if fail_re:
        m = fail_re.search(text)
        if m:
            e.failure_type = m.group(1).lower()
    if e.component and e.failure_type:
        # produce a short symptom = sentence containing the failure word
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if e.failure_type in sent.lower():
                e.symptom = sent.strip()[:240]
                break
        e.description = f"{e.component} -> {e.failure_type}"
    return e


# --------------------------------------------------- LLM client (Claude / OAI)
_PROMPT = """You are extracting structured failure information from a vehicle
customer-feedback note. Return STRICT JSON only, no preamble.

Schema:
{
  "component":    "<noun phrase, e.g. 'driver-side seat heater', 'infotainment head unit'>",
  "failure_type": "<verb or noun, e.g. 'freeze', 'rattle', 'no-power', 'overheat'>",
  "symptom":      "<short observable behaviour, max 160 chars>",
  "description":  "<one-sentence canonical summary, max 200 chars>"
}

Rules:
* If a field is unknown, set it to null (not the string "null").
* Do not invent details that are not in the text.
* Keep wording lowercase, no trailing punctuation.

Texts (return a JSON array of objects in the same order):
"""


class LLMExtractor:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"))
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif provider == "rule_only":
            self.client = None
        else:
            raise ValueError(f"Unknown provider {provider}")

    def __call__(self, texts: List[str]) -> List[Entities]:
        if self.provider == "rule_only" or self.client is None:
            return [Entities() for _ in texts]
        prompt = _PROMPT + json.dumps(texts, ensure_ascii=False, indent=2)

        try:
            if self.provider == "anthropic":
                resp = self.client.messages.create(
                    model=self.model, max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}])
                raw = resp.content[0].text
            else:                                # openai
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=0,
                    messages=[{"role": "user", "content": prompt}])
                raw = resp.choices[0].message.content
            parsed = json.loads(self._extract_json(raw))
            return [Entities(**item) for item in parsed]
        except Exception as e:
            logger.warning(f"LLM extract failed: {e}; returning empty entities")
            return [Entities() for _ in texts]

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Pull the first JSON array from a possibly-noisy LLM response."""
        m = re.search(r"\[[\s\S]*\]", raw)
        return m.group(0) if m else "[]"


# ============================================================ main entry
def run(cfg_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(cfg_path)
    ec = cfg["entity_extraction"]

    src = Path(cfg["paths"]["processed_data_dir"]) / "feedback_clean.parquet"
    dst = Path(cfg["paths"]["processed_data_dir"]) / "feedback_entities.parquet"

    df = load_parquet(src)

    components = load_gazetteer(ec["components_gazetteer"])
    failures = load_gazetteer(ec["failure_types_gazetteer"])
    comp_re = _build_regex(components)
    fail_re = _build_regex(failures)

    rule_results: List[Entities] = []
    needs_llm: List[int] = []

    if ec["rule_first"]:
        for i, txt in enumerate(tqdm(df["text_for_embed"], desc="rule-extract")):
            e = rule_extract(txt, comp_re, fail_re)
            rule_results.append(e)
            if not (e.component and e.failure_type):
                needs_llm.append(i)
    else:
        rule_results = [Entities() for _ in range(len(df))]
        needs_llm = list(range(len(df)))

    logger.info(f"Rule-extracted {len(df) - len(needs_llm):,}, "
                f"sending {len(needs_llm):,} to LLM "
                f"({len(needs_llm) / max(len(df), 1):.1%})")

    extractor = LLMExtractor(ec["provider"], ec["model"])
    bs = ec["llm_batch_size"]

    for chunk in tqdm(list(batched(needs_llm, bs)), desc="llm-extract"):
        texts = [df["text_for_embed"].iloc[i] for i in chunk]
        outs = extractor(texts)
        for idx, e in zip(chunk, outs):
            # only overwrite blank fields produced by the rule pass
            r = rule_results[idx]
            r.component   = r.component   or e.component
            r.failure_type = r.failure_type or e.failure_type
            r.symptom     = r.symptom     or e.symptom
            r.description = r.description or e.description

    ent_df = pd.DataFrame([asdict(e) for e in rule_results])
    out = pd.concat([df.reset_index(drop=True), ent_df], axis=1)
    save_parquet(out, dst)
    return out


if __name__ == "__main__":
    run()
