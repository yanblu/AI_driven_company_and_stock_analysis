"""Semantic chunking for LLM consumption.

Targets ~1-4K tokens per chunk, splitting on semantic boundaries in this
priority: section break > paragraph break > sentence. Chunks carry metadata
so Step 2 prompts can filter by source_type, section, speaker, date, etc.
"""

from __future__ import annotations

import re
from typing import Iterable

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover
    def count_tokens(text: str) -> int:
        # Fallback: ~4 chars per token
        return max(1, len(text) // 4)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return parts


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_text(
    text: str,
    *,
    target_tokens: int = 1800,
    max_tokens: int = 4000,
    min_tokens: int = 200,
) -> list[str]:
    """Greedy paragraph-aware chunking with sentence fallback for huge paragraphs.

    - Pack paragraphs up to target_tokens
    - If a single paragraph exceeds max_tokens, split it at sentence boundaries
    - Small trailing chunk merged with previous unless it meets min_tokens
    """
    if not text:
        return []

    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_tok = 0

    def flush():
        nonlocal buf, buf_tok
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_tok = 0

    for para in paragraphs:
        tok = count_tokens(para)
        if tok > max_tokens:
            # Flush existing, then split the oversize paragraph by sentences
            flush()
            sentences = _split_sentences(para)
            cur: list[str] = []
            cur_tok = 0
            for s in sentences:
                st = count_tokens(s)
                if cur_tok + st > target_tokens and cur:
                    chunks.append(" ".join(cur).strip())
                    cur, cur_tok = [], 0
                cur.append(s)
                cur_tok += st
            if cur:
                chunks.append(" ".join(cur).strip())
            continue

        if buf_tok + tok > target_tokens and buf_tok >= min_tokens:
            flush()
        buf.append(para)
        buf_tok += tok

    flush()

    # Merge last tiny chunk into previous if both exist
    if len(chunks) >= 2 and count_tokens(chunks[-1]) < min_tokens:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + "\n\n" + tail

    return chunks


def make_chunk_records(
    *,
    source_type: str,
    doc_id: str,
    date: str,
    fiscal_quarter: str,
    section: str,
    speaker: dict | None,
    text: str,
    extra_meta: dict | None = None,
    target_tokens: int = 1800,
) -> Iterable[dict]:
    """Yield chunk records with full metadata ready for JSONL output."""
    pieces = chunk_text(text, target_tokens=target_tokens)
    for i, piece in enumerate(pieces):
        rec = {
            "chunk_id": f"{source_type}:{doc_id}:{section}:{i:03d}",
            "source_type": source_type,
            "doc_id": doc_id,
            "date": date,
            "fiscal_quarter": fiscal_quarter,
            "section": section,
            "text": piece,
            "token_count": count_tokens(piece),
        }
        if speaker:
            rec["speaker"] = speaker
        if extra_meta:
            rec.update(extra_meta)
        yield rec
