"""Refresh per-chunk metadata in llm_annotations JSONL from current chunk files.

Use after rebuild_chunks when only speaker / fiscal_quarter / section metadata
changed (LLM cache_key is text-only, so outputs stay valid).

Usage:
    python -m src.analysis.sync_chunk_metadata --source transcripts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import CHUNKS_DIR, LLM_ANNOTATIONS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sync_chunk_metadata")


def _load_chunks(source_type: str) -> dict[str, dict]:
    path = CHUNKS_DIR / f"{source_type}.jsonl"
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = c.get("chunk_id")
            if cid:
                out[cid] = c
    return out


def sync_source(source_type: str) -> tuple[int, int]:
    """Returns (n_updated, n_total)."""
    chunks = _load_chunks(source_type)
    ann_path = LLM_ANNOTATIONS_DIR / f"{source_type}.jsonl"
    if not ann_path.exists():
        log.warning("No annotation file: %s", ann_path)
        return 0, 0

    lines_out: list[str] = []
    n_updated = 0
    n_total = 0
    with ann_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            cid = rec.get("chunk_id")
            ch = chunks.get(cid) if cid else None
            if not ch:
                lines_out.append(json.dumps(rec, ensure_ascii=False))
                continue
            changed = False
            for key in ("speaker", "fiscal_quarter", "td_fiscal_quarter_hint", "section", "date", "token_count"):
                if key in ch and rec.get(key) != ch.get(key):
                    rec[key] = ch[key]
                    changed = True
            if changed:
                n_updated += 1
            lines_out.append(json.dumps(rec, ensure_ascii=False))

    ann_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    log.info("%s: updated %d / %d records", source_type, n_updated, n_total)
    return n_updated, n_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync annotation metadata from chunk JSONL files")
    parser.add_argument(
        "--source",
        choices=["transcripts", "news", "reports_40f", "reports_quarterly", "all"],
        default="transcripts",
    )
    args = parser.parse_args()
    sources = (
        ["transcripts", "news", "reports_40f", "reports_quarterly"]
        if args.source == "all"
        else [args.source]
    )
    for src in sources:
        sync_source(src)


if __name__ == "__main__":
    main()
