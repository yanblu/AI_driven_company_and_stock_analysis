"""Optional Step 2 — FinBERT sanity-check baseline.

Runs ProsusAI/finbert (a BERT model fine-tuned on financial sentiment) over
the transcript chunks and compares sign agreement with the gpt-4o-mini labels.

FinBERT outputs: positive | negative | neutral (with a confidence score).
We compare the sentiment label sign (positive vs. negative) against the LLM
annotation and report the agreement rate.

Usage:
    python -m src.analysis.finbert_baseline             # transcripts only
    python -m src.analysis.finbert_baseline --source news

Requirements (auto-installed if missing):
    pip install transformers torch

Output:
    data/processed/llm_annotations/finbert_transcripts.jsonl   (raw scores)
    data/processed/features/finbert_comparison.parquet         (agreement stats)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import FEATURES_DIR, LLM_ANNOTATIONS_DIR, CHUNKS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finbert_baseline")

MIN_TOKENS = 50
FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_TEXT_CHARS = 512 * 4  # FinBERT max sequence ≈ 512 tokens → ~2048 chars


def _load_transformers():
    try:
        from transformers import pipeline
        return pipeline
    except ImportError:
        log.error("transformers not installed. Run: pip install transformers torch")
        sys.exit(1)


def run_finbert(source_type: str) -> pd.DataFrame:
    pipeline_fn = _load_transformers()

    chunk_path = CHUNKS_DIR / f"{source_type}.jsonl"
    if not chunk_path.exists():
        log.error("%s chunk file not found", source_type)
        return pd.DataFrame()

    # Load chunks above threshold
    chunks: list[dict] = []
    with chunk_path.open() as fh:
        for line in fh:
            try:
                c = json.loads(line)
            except Exception:
                continue
            if c.get("token_count", 0) >= MIN_TOKENS:
                chunks.append(c)

    log.info("Loaded %d chunks from %s", len(chunks), source_type)

    log.info("Loading FinBERT model (%s) — first run downloads ~440 MB...", FINBERT_MODEL)
    classifier = pipeline_fn(
        "text-classification",
        model=FINBERT_MODEL,
        top_k=None,   # return all 3 labels with scores
        truncation=True,
        max_length=512,
    )

    # Run inference in batches
    texts = [c["text"][:MAX_TEXT_CHARS] for c in chunks]
    results: list[list[dict]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        out = classifier(batch)
        results.extend(out)
        log.info("FinBERT: %d / %d", min(i + BATCH_SIZE, len(texts)), len(texts))

    # Build output records
    records = []
    for chunk, label_list in zip(chunks, results):
        label_scores = {item["label"].lower(): item["score"] for item in label_list}
        top_label = max(label_scores, key=label_scores.get)
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_type": source_type,
                "fiscal_quarter": chunk.get("fiscal_quarter", ""),
                "section": chunk.get("section", ""),
                "speaker_role": (chunk.get("speaker") or {}).get("role", ""),
                "finbert_label": top_label,
                "finbert_positive": label_scores.get("positive", 0.0),
                "finbert_negative": label_scores.get("negative", 0.0),
                "finbert_neutral": label_scores.get("neutral", 0.0),
                # Continuous score: positive prob − negative prob
                "finbert_score": label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0),
            }
        )

    # Save raw FinBERT results
    out_path = LLM_ANNOTATIONS_DIR / f"finbert_{source_type}.jsonl"
    LLM_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    log.info("Wrote %d FinBERT annotations -> %s", len(records), out_path)

    return pd.DataFrame(records)


def compare_with_llm(source_type: str, finbert_df: pd.DataFrame) -> None:
    """Load LLM annotations and compute sign agreement rate."""
    llm_path = LLM_ANNOTATIONS_DIR / f"{source_type}.jsonl"
    if not llm_path.exists():
        log.warning("LLM annotation file %s not found — skipping comparison", llm_path)
        return

    llm_records = []
    with llm_path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            llm_records.append(
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "llm_sentiment": r.get("output", {}).get("sentiment", ""),
                    "llm_score": r.get("output", {}).get("sentiment_score", None),
                }
            )
    llm_df = pd.DataFrame(llm_records)

    merged = finbert_df.merge(llm_df, on="chunk_id", how="inner")
    if merged.empty:
        log.warning("No matching chunk IDs between FinBERT and LLM results")
        return

    # Sign agreement: both positive OR both negative (neutral is excluded from
    # the binary sign check — FinBERT neutral chunks are considered separately)
    def llm_sign(s):
        return 1 if s == "positive" else (-1 if s == "negative" else 0)

    def fb_sign(s):
        return 1 if s == "positive" else (-1 if s == "negative" else 0)

    merged["llm_sign"] = merged["llm_sentiment"].map(llm_sign)
    merged["fb_sign"] = merged["finbert_label"].map(fb_sign)

    # Restrict to non-neutral from both models for the sign agreement metric
    opinionated = merged[(merged["llm_sign"] != 0) & (merged["fb_sign"] != 0)]
    agree = (opinionated["llm_sign"] == opinionated["fb_sign"]).sum()
    total = len(opinionated)
    agreement_rate = agree / total if total > 0 else 0.0

    log.info(
        "Sign agreement (%s): %d/%d = %.1f%%",
        source_type, agree, total, agreement_rate * 100
    )
    if agreement_rate < 0.70:
        log.warning(
            "Agreement < 70%% — investigate divergent chunks for prompt refinement"
        )

    # Quarterly agreement breakdown
    quarterly = (
        opinionated.groupby("fiscal_quarter")
        .apply(lambda g: (g["llm_sign"] == g["fb_sign"]).mean())
        .rename("agreement_rate")
        .reset_index()
    )

    # Correlation of continuous scores
    corr = merged["llm_score"].corr(merged["finbert_score"])
    log.info("Pearson correlation of score values: %.3f", corr)

    # Save comparison
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    comp = merged[["chunk_id", "fiscal_quarter", "section", "speaker_role",
                   "finbert_label", "finbert_score",
                   "llm_sentiment", "llm_score"]].copy()
    comp["sign_agree"] = opinionated["llm_sign"] == opinionated["fb_sign"]
    out = FEATURES_DIR / f"finbert_comparison_{source_type}.parquet"
    comp.to_parquet(out, index=False)
    log.info("Wrote comparison -> %s", out)

    # Print summary
    print(f"\n{'='*50}")
    print(f"FinBERT vs LLM Comparison — {source_type}")
    print(f"{'='*50}")
    print(f"  Total chunks compared:      {len(merged):,}")
    print(f"  Opinionated (non-neutral):  {total:,}")
    print(f"  Sign agreement rate:        {agreement_rate:.1%}")
    print(f"  Score correlation (Pearson):{corr:.3f}")
    print(f"\nQuarterly agreement rates:")
    print(quarterly.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="FinBERT baseline for Step 2 sanity check")
    parser.add_argument(
        "--source",
        default="transcripts",
        choices=["transcripts", "news", "reports_40f", "reports_quarterly"],
        help="Source type to run FinBERT on (default: transcripts)",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip LLM comparison (useful if annotate.py hasn't been run yet)",
    )
    args = parser.parse_args()

    finbert_df = run_finbert(args.source)
    if finbert_df.empty:
        return

    if not args.skip_compare:
        compare_with_llm(args.source, finbert_df)
    else:
        log.info("Skipping LLM comparison (--skip-compare)")


if __name__ == "__main__":
    main()
