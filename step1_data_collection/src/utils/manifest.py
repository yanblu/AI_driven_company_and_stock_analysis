"""Manifest bookkeeping for data provenance.

Every collected artifact should append a row here so that the data collection
methodology doc can summarize sources, retrieval dates, and SHA-256 checksums.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .config import MANIFEST_PATH

MANIFEST_COLUMNS = [
    "source",            # high-level source name (e.g., yfinance)
    "artifact_type",     # prices|macro|filing|transcript|news|chunks
    "identifier",        # ticker / accession / doc id / etc.
    "path",              # relative path from step1_data_collection/
    "url",               # origin URL (if applicable)
    "retrieved_at",      # ISO 8601 UTC timestamp
    "record_count",      # rows/items in the artifact (if known)
    "sha256",            # file checksum
    "notes",             # free-form
]


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_manifest() -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(MANIFEST_COLUMNS)


def record_artifact(
    *,
    source: str,
    artifact_type: str,
    identifier: str,
    path: Path,
    url: str = "",
    record_count: int | None = None,
    notes: str = "",
) -> None:
    """Append a provenance row for a local artifact.

    If the file doesn't exist, the call is a no-op (collectors may skip items).
    """
    if not path.exists():
        return
    _ensure_manifest()
    try:
        rel = path.relative_to(MANIFEST_PATH.parent.parent)
    except ValueError:
        rel = path
    row = [
        source,
        artifact_type,
        identifier,
        str(rel),
        url,
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "" if record_count is None else str(record_count),
        _sha256_of_file(path),
        notes,
    ]
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(row)
