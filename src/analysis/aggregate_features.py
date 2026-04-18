"""Step 2 — Aggregate per-chunk LLM annotations to quarter-level NLP features.

Reads all annotation JSONL files (one per source type) from
data/processed/llm_annotations/ and produces:

  data/processed/features/nlp_features.parquet

One row per TD fiscal quarter (e.g., "FY2024Q2"), with columns:

  Sentiment (per source × speaker role)
  ───────────────────────────────────────
  transcript_ceo_prep_sentiment_mean
  transcript_cfo_prep_sentiment_mean
  transcript_exec_qa_sentiment_mean      (other_exec + ceo + cfo in Q&A)
  transcript_analyst_qa_sentiment_mean
  transcript_sentiment_std               (within-quarter volatility)
  news_sentiment_mean
  report_40f_sentiment_mean
  report_quarterly_sentiment_mean

  Topic share & sentiment (per label)
  ────────────────────────────────────
  topic_{label}_share                    (fraction of chunks with that label)
  topic_{label}_sentiment                (mean sentiment_score for that label)

  Diversity
  ─────────
  topic_entropy                          (Shannon entropy of topic distribution)
  n_chunks_annotated                     (total annotated chunks in quarter)

Usage:
    python -m src.analysis.aggregate_features
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import FEATURES_DIR, LLM_ANNOTATIONS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aggregate_features")

TOPICS = [
    "NIM", "credit_quality", "capital", "US_retail", "Canadian_personal",
    "wealth_wholesale", "regulatory_AML", "macro_outlook", "cost_efficiency",
    "guidance", "M_and_A", "other",
]

SOURCE_TYPES = ["transcripts", "news", "reports_40f", "reports_quarterly"]


def _load_annotations() -> list[dict]:
    records: list[dict] = []
    for src in SOURCE_TYPES:
        path = LLM_ANNOTATIONS_DIR / f"{src}.jsonl"
        if not path.exists():
            log.warning("%s annotation file missing, skipping", src)
            continue
        n = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                # Skip entries without a valid fiscal_quarter
                fq = r.get("fiscal_quarter") or r.get("td_fiscal_quarter_hint") or ""
                if not fq or fq == "nan":
                    continue
                r["_fq"] = fq
                r["_src"] = src
                records.append(r)
                n += 1
        log.info("Loaded %d annotations from %s", n, src)
    return records


def _safe_mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(np.mean(vals))


def _shannon_entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for v in counts.values():
        if v > 0:
            p = v / total
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def aggregate(records: list[dict]) -> pd.DataFrame:
    # Group by fiscal quarter
    by_fq: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_fq[r["_fq"]].append(r)

    rows: list[dict] = []
    for fq, recs in sorted(by_fq.items()):
        row: dict = {"fiscal_quarter": fq, "n_chunks_annotated": len(recs)}

        out_list = [r.get("output", {}) for r in recs]

        # -----------------------------------------------------------------
        # Sentiment features
        # -----------------------------------------------------------------

        # Helper: filter by source and optional (section, role) predicate
        def scores(src: str, section_pred=None, role_pred=None) -> list[float]:
            result = []
            for r, o in zip(recs, out_list):
                if r["_src"] != src:
                    continue
                if section_pred is not None and not section_pred(r.get("section", "")):
                    continue
                spk = r.get("speaker") or {}
                if role_pred is not None and not role_pred(spk.get("role", "")):
                    continue
                score = o.get("sentiment_score")
                if score is not None:
                    result.append(float(score))
            return result

        # Transcript — CEO prepared remarks
        row["transcript_ceo_prep_sentiment_mean"] = _safe_mean(
            scores("transcripts",
                   section_pred=lambda s: s == "prepared_remarks",
                   role_pred=lambda r: r == "ceo")
        )
        # Transcript — CFO prepared remarks
        row["transcript_cfo_prep_sentiment_mean"] = _safe_mean(
            scores("transcripts",
                   section_pred=lambda s: s == "prepared_remarks",
                   role_pred=lambda r: r == "cfo")
        )
        # Transcript — exec responses in Q&A (ceo+cfo+other_exec)
        exec_roles = {"ceo", "cfo", "other_exec"}
        row["transcript_exec_qa_sentiment_mean"] = _safe_mean(
            scores("transcripts",
                   section_pred=lambda s: s == "qa",
                   role_pred=lambda r: r in exec_roles)
        )
        # Transcript — analyst questions in Q&A
        row["transcript_analyst_qa_sentiment_mean"] = _safe_mean(
            scores("transcripts",
                   section_pred=lambda s: s == "qa",
                   role_pred=lambda r: r == "analyst")
        )
        # Transcript — within-quarter sentiment std (all transcript chunks)
        all_transcript_scores = scores("transcripts")
        row["transcript_sentiment_std"] = (
            float(np.std(all_transcript_scores)) if len(all_transcript_scores) > 1 else None
        )
        row["news_sentiment_mean"] = _safe_mean(scores("news"))
        row["report_40f_sentiment_mean"] = _safe_mean(scores("reports_40f"))
        row["report_quarterly_sentiment_mean"] = _safe_mean(scores("reports_quarterly"))

        # -----------------------------------------------------------------
        # Topic features
        # -----------------------------------------------------------------
        topic_counts: dict[str, int] = {t: 0 for t in TOPICS}
        topic_scores: dict[str, list[float]] = {t: [] for t in TOPICS}

        for r, o in zip(recs, out_list):
            score = o.get("sentiment_score")
            for topic in o.get("topics", []):
                if topic in topic_counts:
                    topic_counts[topic] += 1
                    if score is not None:
                        topic_scores[topic].append(float(score))

        n_total = len(recs)
        for topic in TOPICS:
            row[f"topic_{topic}_share"] = (
                round(topic_counts[topic] / n_total, 4) if n_total > 0 else 0.0
            )
            row[f"topic_{topic}_sentiment"] = _safe_mean(topic_scores[topic])

        # Shannon entropy across topic distribution (excluding "other")
        non_other = {t: c for t, c in topic_counts.items() if t != "other"}
        row["topic_entropy"] = _shannon_entropy(non_other)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("fiscal_quarter").reset_index(drop=True)
    return df


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    records = _load_annotations()
    if not records:
        log.error("No annotation records found — run annotate.py first")
        return

    log.info("Total annotation records: %d", len(records))
    df = aggregate(records)
    log.info("Quarters aggregated: %d", len(df))

    out = FEATURES_DIR / "nlp_features.parquet"
    df.to_parquet(out, index=False)
    log.info("Wrote %s  shape=%s", out, df.shape)

    # Print summary table
    sentiment_cols = [c for c in df.columns if "sentiment_mean" in c or "sentiment_std" in c]
    topic_share_cols = [c for c in df.columns if "_share" in c]
    log.info("\nSentiment summary (means across quarters):\n%s",
             df[["fiscal_quarter"] + sentiment_cols].to_string(index=False))
    log.info("\nTop topics by average share:\n%s",
             df[topic_share_cols].mean().sort_values(ascending=False).head(6).to_string())


if __name__ == "__main__":
    main()
