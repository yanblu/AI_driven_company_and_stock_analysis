"""Step 2 — LLM annotation runner.

Sends each chunk through a zero-shot gpt-4o-mini prompt and stores the
structured result (sentiment + topics + key_quote) in the llm_cache.

Usage:
    python -m src.analysis.annotate                    # all source types
    python -m src.analysis.annotate --source transcripts
    python -m src.analysis.annotate --dry-run          # print first batch, no API call

Requires OPENAI_API_KEY in .env (or environment).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_STEP2_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _STEP2_DIR.parent
sys.path.insert(0, str(_STEP2_DIR))      # for src.config
sys.path.insert(0, str(_PROJECT_ROOT))   # for step1_data_collection imports

from step1_data_collection.src.preprocess.llm_cache import append_records, cache_key, load_cache  # noqa: E402
from src.config import CHUNKS_DIR, LLM_ANNOTATIONS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("annotate")

# ---------------------------------------------------------------------------
# Config — change MODEL or PROMPT_VERSION here to trigger a re-annotation pass
# ---------------------------------------------------------------------------
MODEL = "gpt-4o-mini"
PROMPT_VERSION = "v1"
BATCH_SIZE = 20        # chunks per API call (each call = one system msg + N user msgs)
MIN_TOKENS = 45        # skip stub chunks; 45 catches short CEO openers that are still substantive
MAX_TABLE_RATIO = 0.40 # skip chunks where >40% of tokens are numeric/symbolic
MAX_RETRIES = 2        # per-batch retry attempts on parse failure
REQUEST_DELAY = 0.25   # seconds between batches (rate-limit headroom)

SOURCE_TYPES = ["transcripts", "news", "reports_40f", "reports_quarterly"]

TOPICS = [
    "NIM",
    "credit_quality",
    "capital",
    "US_retail",
    "Canadian_personal",
    "wealth_wholesale",
    "regulatory_AML",
    "macro_outlook",
    "cost_efficiency",
    "guidance",
    "M_and_A",
    "other",
]

SYSTEM_PROMPT = (
    "You are a financial analyst specialising in Canadian banks. "
    "Analyse the following passage from a TD Bank document and respond with a "
    "JSON object. Be concise and precise. Respond ONLY with the JSON — no prose, "
    "no markdown fences."
)

TOPIC_LIST_STR = ", ".join(TOPICS)


_INTRO_KEYWORDS = frozenset(
    ["introduced", "introduce", "leadership team", "participants",
     "presenting", "joining", "welcome", "moderator", "operator"]
)


def _is_admin_chunk(chunk: dict) -> bool:
    """Return True for administrative/boilerplate transcript chunks that carry no sentiment signal.

    Two patterns:
    1. Speaker role is 'ir' — always a forward-looking-statement disclaimer or housekeeping preamble.
    2. Score would be 0.0 and key_quote contains intro-announcement language — operator/moderator
       lines introducing participants (e.g. "Good afternoon and welcome…").

    These chunks consistently score 0.000 and contribute only noise to speaker-level aggregations.
    """
    role = (chunk.get("speaker") or {}).get("role", "")
    if role == "ir":
        return True
    text_lower = (chunk.get("text") or "").lower()
    if len(text_lower.split()) < 80 and any(kw in text_lower for kw in _INTRO_KEYWORDS):
        return True
    return False


def _is_table_heavy(text: str, threshold: float = MAX_TABLE_RATIO) -> bool:
    """Return True if >threshold fraction of whitespace-split tokens are numeric/symbolic.

    Financial tables extracted from PDFs look like rows of numbers, '$', '%',
    column headers, and very few prose words. The LLM cannot produce reliable
    sentiment from these — it infers sentiment from the numbers themselves,
    which is closer to hallucination than annotation.
    """
    tokens = text.split()
    if not tokens:
        return True
    numeric = sum(1 for t in tokens if re.fullmatch(r"[\d,.\$%\(\)\-\/\|]+", t))
    return numeric / len(tokens) > threshold


def _build_user_message(chunk: dict) -> str:
    source = chunk.get("source_type", "")
    fq = chunk.get("fiscal_quarter", "")
    section = chunk.get("section", "")
    speaker = chunk.get("speaker") or {}
    spk_name = speaker.get("name", "")
    spk_role = speaker.get("role", "")

    lines: list[str] = []
    lines.append(f"Document type: {source}  Quarter: {fq}  Section: {section}")
    if spk_name:
        lines.append(f"Speaker: {spk_name} ({spk_role})")
    lines.append("")
    lines.append('Passage:')
    lines.append('"""')
    lines.append(chunk["text"])
    lines.append('"""')
    lines.append("")
    lines.append("Respond with exactly this JSON and nothing else:")
    lines.append("{")
    lines.append('  "sentiment": "positive" or "neutral" or "negative",')
    lines.append('  "sentiment_score": <float -1.0 to 1.0>,')
    lines.append(f'  "topics": [<1-3 labels from: {TOPIC_LIST_STR}>],')
    lines.append('  "key_quote": "<one sentence max 20 words capturing the core message>"')
    lines.append("}")
    return "\n".join(lines)


def _parse_response(text: str) -> dict | None:
    """Parse LLM response, stripping accidental markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().strip("`").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None

    # Validate required fields
    if not isinstance(obj.get("sentiment"), str):
        return None
    if obj["sentiment"] not in ("positive", "neutral", "negative"):
        return None
    if not isinstance(obj.get("sentiment_score"), (int, float)):
        return None
    if not isinstance(obj.get("topics"), list):
        return None
    return obj


def _call_openai(client, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content or ""


def annotate_source(source_type: str, dry_run: bool = False) -> None:
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_exponential

    chunk_path = CHUNKS_DIR / f"{source_type}.jsonl"
    if not chunk_path.exists():
        log.warning("%s chunk file not found, skipping", source_type)
        return

    # Load all chunks, applying token-count, table-content, and admin filters
    all_chunks: list[dict] = []
    n_short = 0
    n_table = 0
    n_admin = 0
    with chunk_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                c = json.loads(line)
            except Exception:
                continue
            if c.get("token_count", 0) < MIN_TOKENS:
                n_short += 1
                continue
            if _is_table_heavy(c.get("text", "")):
                n_table += 1
                continue
            if source_type == "transcripts" and _is_admin_chunk(c):
                n_admin += 1
                continue
            all_chunks.append(c)

    log.info(
        "%s: %d chunks eligible  (skipped %d short, %d table-heavy, %d admin)",
        source_type, len(all_chunks), n_short, n_table, n_admin,
    )

    # Load cache and filter already-annotated
    cache = load_cache(source_type)
    pending = [
        c for c in all_chunks
        if cache_key(c["text"], PROMPT_VERSION, MODEL) not in cache
    ]
    log.info(
        "%s: %d cached, %d to annotate",
        source_type,
        len(all_chunks) - len(pending),
        len(pending),
    )

    if not pending:
        log.info("%s: nothing to do", source_type)
        return

    if dry_run:
        log.info("[DRY RUN] first batch user message:")
        print(_build_user_message(pending[0]))
        return

    client = OpenAI()  # picks up OPENAI_API_KEY from env

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call_with_retry(msgs):
        return _call_openai(client, msgs)

    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    total_annotated = 0
    total_failed = 0

    for bi, batch in enumerate(batches):
        new_records: list[dict] = []

        for chunk in batch:
            user_msg = _build_user_message(chunk)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            raw = ""
            parsed = None
            for attempt in range(MAX_RETRIES):
                try:
                    raw = _call_with_retry(messages)
                    parsed = _parse_response(raw)
                    if parsed is not None:
                        break
                    log.warning(
                        "chunk %s attempt %d parse failed, raw=%r",
                        chunk["chunk_id"], attempt + 1, raw[:120]
                    )
                except Exception as exc:
                    log.warning("chunk %s attempt %d API error: %s", chunk["chunk_id"], attempt + 1, exc)

            if parsed is None:
                log.warning("chunk %s: all retries failed, skipping", chunk["chunk_id"])
                total_failed += 1
                continue

            # Clamp score to [-1, 1]
            score = max(-1.0, min(1.0, float(parsed["sentiment_score"])))

            # Filter topics to known taxonomy
            valid_topics = [t for t in parsed.get("topics", []) if t in TOPICS]
            if not valid_topics:
                valid_topics = ["other"]

            new_records.append({
                "cache_key": cache_key(chunk["text"], PROMPT_VERSION, MODEL),
                "chunk_id": chunk["chunk_id"],
                "source_type": source_type,
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "output": {
                    "sentiment": parsed["sentiment"],
                    "sentiment_score": score,
                    "topics": valid_topics,
                    "key_quote": str(parsed.get("key_quote", ""))[:200],
                },
                # carry-through metadata for aggregation (avoids re-loading chunks)
                "date": chunk.get("date", ""),
                "fiscal_quarter": chunk.get("fiscal_quarter", ""),
                "td_fiscal_quarter_hint": chunk.get("td_fiscal_quarter_hint", ""),
                "section": chunk.get("section", ""),
                "speaker": chunk.get("speaker"),
                "token_count": chunk.get("token_count", 0),
            })

        if new_records:
            append_records(source_type, new_records)
            total_annotated += len(new_records)

        done = (bi + 1) * BATCH_SIZE
        log.info(
            "%s: batch %d/%d  annotated=%d failed=%d",
            source_type, bi + 1, len(batches), total_annotated, total_failed,
        )
        time.sleep(REQUEST_DELAY)

    log.info(
        "%s: DONE  total_annotated=%d  total_failed=%d",
        source_type, total_annotated, total_failed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM annotation for Step 2")
    parser.add_argument(
        "--source",
        choices=SOURCE_TYPES + ["all"],
        default="all",
        help="Which chunk source type to annotate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt for each source and exit without calling the API",
    )
    args = parser.parse_args()

    sources = SOURCE_TYPES if args.source == "all" else [args.source]
    for src in sources:
        log.info("=== Annotating %s ===", src)
        annotate_source(src, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
