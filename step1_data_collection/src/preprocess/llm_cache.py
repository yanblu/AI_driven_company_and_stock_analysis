"""LLM annotation cache scaffolding.

Purpose: re-running Step 2 with an unchanged prompt + model is free; prompt
iteration only re-annotates chunks whose (text, prompt_version, model) tuple
differs from cached entries.

Layout:
  data/processed/llm_annotations/{source_type}.jsonl

Each line:
  {
    "cache_key": sha256 of (chunk_text + prompt_version + model_name),
    "chunk_id": "...",
    "source_type": "...",
    "model": "...",
    "prompt_version": "...",
    "output": { ... LLM response ... },
    "created_at": ISO timestamp
  }

This module only provides helpers; actual LLM calls happen in Step 2 notebooks.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.config import LLM_ANNOTATIONS_DIR  # noqa: E402


def cache_key(chunk_text: str, prompt_version: str, model_name: str) -> str:
    h = hashlib.sha256()
    h.update(chunk_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(model_name.encode("utf-8"))
    return h.hexdigest()


def cache_path(source_type: str) -> Path:
    LLM_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return LLM_ANNOTATIONS_DIR / f"{source_type}.jsonl"


def load_cache(source_type: str) -> dict[str, dict]:
    """Load cache as dict keyed by cache_key. Missing file returns {}."""
    p = cache_path(source_type)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            k = rec.get("cache_key")
            if k:
                out[k] = rec
    return out


def append_records(source_type: str, records: Iterable[dict]) -> int:
    """Append new cache records. Returns count written."""
    p = cache_path(source_type)
    written = 0
    with p.open("a", encoding="utf-8") as fh:
        for r in records:
            r.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += 1
    return written
