"""Rebuild data/processed/llm_annotations/transcripts.jsonl from chunks + LLM cache.

After build_chunks changes chunk_ids or speaker metadata, re-materialize annotation
rows: same text → same cache_key → reuse cached LLM output; update chunk_id/speaker
from current chunks. No API calls unless a chunk text is missing from cache.

Usage:
    python -m src.analysis.rebuild_transcripts_annotations
    python -m src.analysis.annotate --source transcripts   # fills cache misses only
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.annotate import (  # noqa: E402
    MAX_TABLE_RATIO,
    MIN_TOKENS,
    MODEL,
    PROMPT_VERSION,
    _is_admin_chunk,
    _is_table_heavy,
)
from src.preprocess.llm_cache import cache_key, load_cache  # noqa: E402
from src.utils.config import CHUNKS_DIR, LLM_ANNOTATIONS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rebuild_transcripts_annotations")


def main() -> None:
    chunk_path = CHUNKS_DIR / "transcripts.jsonl"
    out_path = LLM_ANNOTATIONS_DIR / "transcripts.jsonl"

    cache = load_cache("transcripts")
    log.info("Loaded %d cache keys from existing transcripts.jsonl", len(cache))

    records: list[dict] = []
    n_skip_short = n_skip_table = n_skip_admin = 0
    n_miss = 0

    with chunk_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if c.get("token_count", 0) < MIN_TOKENS:
                n_skip_short += 1
                continue
            if _is_table_heavy(c.get("text", ""), MAX_TABLE_RATIO):
                n_skip_table += 1
                continue
            if _is_admin_chunk(c):
                n_skip_admin += 1
                continue

            text = c.get("text", "")
            k = cache_key(text, PROMPT_VERSION, MODEL)
            hit = cache.get(k)
            if not hit or "output" not in hit:
                n_miss += 1
                continue

            rec = {
                "cache_key": k,
                "chunk_id": c["chunk_id"],
                "source_type": "transcripts",
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "output": hit["output"],
                "date": c.get("date", ""),
                "fiscal_quarter": c.get("fiscal_quarter", ""),
                "td_fiscal_quarter_hint": c.get("td_fiscal_quarter_hint", ""),
                "section": c.get("section", ""),
                "speaker": c.get("speaker"),
                "token_count": c.get("token_count", 0),
                "created_at": hit.get("created_at", ""),
            }
            records.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(
        "Wrote %d records -> %s  (skipped short=%d table=%d admin=%d, cache_miss=%d)",
        len(records),
        out_path,
        n_skip_short,
        n_skip_table,
        n_skip_admin,
        n_miss,
    )
    if n_miss:
        log.warning(
            "%d eligible chunks had no cache hit — run: python -m src.analysis.annotate --source transcripts",
            n_miss,
        )


if __name__ == "__main__":
    main()
