"""Light, LLM-oriented text cleaning utilities.

We deliberately preserve casing, punctuation, and numbers — LLMs use them
as signal. We only strip noise that burns tokens without carrying meaning:
  - repeated page headers / footers / page numbers (filings, transcripts)
  - safe-harbor / forward-looking-statement boilerplate paragraphs
  - legal disclaimer footers
  - excess whitespace
"""

from __future__ import annotations

import re
from bs4 import BeautifulSoup

# Boilerplate patterns (lines/paragraphs to drop).
# Matched case-insensitively as substrings against each paragraph.
BOILERPLATE_PATTERNS = [
    r"forward[-\s]?looking statements?",
    r"safe harbor",
    r"private securities litigation reform act",
    r"this document and other documents? filed or furnished",
    r"non[-\s]?gaap (financial )?measures",
    r"please refer to (our|td's|the bank's) annual report",
    r"toronto[-\s]dominion bank\s*\(td\)\s*operates",  # stock/company boilerplate
    r"for further information:?\s*$",
    r"members of the media",
    r"td bank group is a subsidiary of the toronto",
    r"the information in this document",
    r"the material in this report",
]

BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# Page headers/footers commonly seen in transcript PDFs
PAGE_HEADER_PATTERNS = [
    r"page\s+\d+\s+of\s+\d+",
    r"^\s*\d+\s*$",  # just a page number
    r"^\s*-\s*\d+\s*-\s*$",
    r"td bank group\s*\|\s*(q[1-4]|quarterly).*(conference call|transcript)",
]

PAGE_HEADER_RE = re.compile("|".join(PAGE_HEADER_PATTERNS), re.IGNORECASE)


def html_to_text(html: str) -> str:
    """Convert filing/news HTML to text, preserving paragraph breaks."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Block-level elements become paragraph breaks
    paragraphs = []
    for el in soup.find_all(
        ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "tr", "div"]
    ):
        t = el.get_text(" ", strip=True)
        if t:
            paragraphs.append(t)
    if not paragraphs:
        paragraphs = [soup.get_text(" ", strip=True)]
    return "\n\n".join(paragraphs)


def clean_text(text: str) -> str:
    """Drop boilerplate paragraphs, collapse whitespace, remove page chrome.

    Preserves casing, punctuation, and numbers by design.
    """
    if not text:
        return ""

    out_paragraphs = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        low = para.lower()
        # Single-line page chrome
        if PAGE_HEADER_RE.search(para) and len(para) < 100:
            continue
        # Boilerplate tends to trigger once per paragraph; drop paragraph
        # if boilerplate phrase appears AND paragraph is mostly that content.
        if BOILERPLATE_RE.search(low):
            # If the paragraph is long, keep it only if it also has substantive
            # non-boilerplate content. Heuristic: drop if < 400 chars.
            if len(para) < 400:
                continue
        # Collapse weird multi-space runs
        para = re.sub(r"[ \t]+", " ", para)
        para = re.sub(r"\s*\n\s*", " ", para)
        out_paragraphs.append(para.strip())

    return "\n\n".join(out_paragraphs)


# -----------------------------------------------------------------------------
# Filing section tagging (40-F / 6-K Report-to-Shareholders)
# -----------------------------------------------------------------------------

# Ordered by typical appearance in TD 40-F / Report to Shareholders
KNOWN_SECTIONS = [
    "business overview",
    "how we performed",
    "financial highlights",
    "management's discussion and analysis",
    "operating environment",
    "economic summary and outlook",
    "accounting policies and estimates",
    "financial results overview",
    "business segment results",
    "canadian personal and commercial banking",
    "u.s. retail",
    "wealth management and insurance",
    "wholesale banking",
    "corporate segment",
    "balance sheet review",
    "risk factors",
    "managing risk",
    "credit risk",
    "market risk",
    "liquidity risk",
    "operational risk",
    "capital position",
    "regulatory developments",
    "significant and subsequent events",
    "controls and procedures",
]

SECTION_RE = re.compile(
    r"(?im)^\s*(" + "|".join(re.escape(s) for s in KNOWN_SECTIONS) + r")\s*$"
)


def tag_sections(text: str) -> list[dict]:
    """Split a filing body into (section_name, text) chunks.

    Each returned dict: {"section": str, "text": str}.
    If no known sections are found, returns a single "unknown" block.
    """
    lines = text.splitlines()
    sections: list[dict] = []
    current_name = "preamble"
    current_lines: list[str] = []

    for line in lines:
        m = SECTION_RE.match(line)
        if m:
            if current_lines:
                sections.append(
                    {
                        "section": current_name,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
            current_name = m.group(1).lower().strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section": current_name,
                "text": "\n".join(current_lines).strip(),
            }
        )
    return [s for s in sections if s["text"]]


# -----------------------------------------------------------------------------
# Transcript speaker + Q&A tagging
# -----------------------------------------------------------------------------

# Matches lines like "Bharat Masrani – President and CEO, TD Bank Group" or
# "Gabriel Dechaine, NBF Analyst" or "Operator"
SPEAKER_LINE_RE = re.compile(
    r"^\s*(?P<name>[A-Z][A-Za-z'\-\.]+(?:\s+[A-Z][A-Za-z'\-\.]+){0,3})"
    r"(?:\s*[-–—,]\s*(?P<affil>[^,\n]{3,120}))?\s*$"
)

# Primary: explicit Q&A header line ("QUESTION AND ANSWER", "Q&A", etc.)
# Fallback: transcripts that skip the header and go straight to the operator
# taking questions (pattern: "We will now take questions from the telephone
# lines" or "We'll now take questions..."). When the fallback matches we
# include the matched text in the Q&A block rather than discarding it.
QA_HEADER_RE = re.compile(
    # Compact form: "QUESTION AND ANSWER", "Q&A", etc.
    # Spaced-letter form from some TD PDFs: "Q U E S T I O N  A N D  A N S W E R"
    r"^\s*(?:questions?\s*(?:and|&)\s*answers?|q\s*&\s*a|question[-\s]and[-\s]answer"
    r"|Q\s+U\s+E\s+S\s+T\s+I\s+O\s+N\s+A\s+N\s+D\s+A\s+N\s+S\s+W\s+E\s+R)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
QA_IMPLICIT_RE = re.compile(
    r"(?im)^(Thank you[,.]?\s+)?We(?:'ll| will) now take questions from the telephone",
)

OPERATOR_NAMES = {"operator"}

# Lines that match SPEAKER_LINE_RE but are mid-paragraph PDF artefacts (sentence
# starts like "In Wholesale Banking, …" or "… America's Most Convenient Bank …")
# incorrectly split from the prior executive's prepared remarks.
_BAD_SPEAKER_FIRST_TOKEN = frozenset(
    {
        # prepositions / articles / conjunctions
        "in", "on", "at", "to", "the", "we", "our", "as", "for",
        "this", "that", "these", "those", "a", "an", "and", "or", "but",
        "if", "when", "while", "with",
        # sentence-start words never seen as first names in bank transcripts
        "also", "thank", "convenient", "wholesale",
        # bank product / initiative words that appear as PDF-artefact "speakers"
        "td",        # TD Bank, TD Rewards, TD Auto Finance, …
        "wealth",    # Wealth Management
        "investor",  # Investor Day
        "complete",  # Complete Checking
        "income",    # Income Survey
        "united",    # United States
        "tech",      # Tech Global Markets
        "djsi",      # DJSI North American Index
    }
)

# Words that are clearly not surnames — if the *last* token of the parsed
# speaker line is one of these it is a segment/product/programme name, not a person.
_BAD_SPEAKER_LAST_TOKEN = frozenset(
    {
        "management", "markets", "rewards", "checking", "survey",
        "index", "banking", "technologies", "solutions", "payments",
        "program", "programme", "fund", "bank",
    }
)


def _is_plausible_speaker_name(name: str) -> bool:
    """Return True if *name* looks like a person / operator line, not a sentence stub."""
    n = (name or "").strip()
    if not n:
        return False
    low = n.lower()
    if low == "operator":
        return True
    parts = n.split()
    if len(parts) < 2:
        return False
    if parts[0].lower() in _BAD_SPEAKER_FIRST_TOKEN:
        return False
    if parts[-1].lower() in _BAD_SPEAKER_LAST_TOKEN:
        return False
    # Long phrases are usually titles or table fragments, not "Firstname Lastname"
    if len(parts) > 5:
        return False
    if low in ("convenient bank", "td bank group"):
        return False
    return True


def _inherit_exec_in_prepared(turns: list[dict]) -> list[dict]:
    """Re-label spurious ``other`` turns that are still the prior executive's remarks.

    Rare PDF layouts leave a plausible-looking *name* on a fragment line; if role is
    still ``other`` after parsing, copy the last CEO/CFO/other_exec speaker.
    """
    last: tuple[str, str, str] | None = None
    for t in turns:
        role = t.get("role", "other")
        name = (t.get("speaker_name") or "").strip()
        if role in ("ceo", "cfo", "other_exec"):
            last = (name, role, (t.get("affiliation") or "").strip())
        elif role == "other" and last and not _is_plausible_speaker_name(name):
            t["speaker_name"] = last[0]
            t["role"] = last[1]
            t["affiliation"] = last[2]
    return turns


def _classify_speaker_role(name: str, affiliation: str | None) -> str:
    affil = (affiliation or "").lower()
    nl = name.lower()
    if nl in OPERATOR_NAMES:
        return "operator"
    if "analyst" in affil or "research" in affil:
        return "analyst"
    if "chief executive" in affil or "ceo" in affil:
        return "ceo"
    if "chief financial" in affil or "cfo" in affil:
        return "cfo"
    if "group head" in affil or "president" in affil or "chief" in affil:
        return "other_exec"
    if any(k in affil for k in ("investor relations", "ir head")):
        return "ir"
    return "other"


def parse_transcript(text: str) -> dict:
    """Split a transcript into prepared-remarks vs Q&A plus speaker turns.

    Returns:
      {
        "sections": {
          "prepared_remarks": [ {speaker_name, role, affiliation, text}, ... ],
          "qa": [...]
        }
      }
    """
    # Split text at first Q&A header.
    # Primary: explicit standalone header → skip the header line itself.
    # Fallback: operator line that signals start of Q&A → keep it in qa_text.
    qa_match = QA_HEADER_RE.search(text)
    if qa_match:
        prepared_text = text[: qa_match.start()]
        qa_text = text[qa_match.end() :]
    else:
        implicit_match = QA_IMPLICIT_RE.search(text)
        if implicit_match:
            prepared_text = text[: implicit_match.start()]
            qa_text = text[implicit_match.start() :]  # include the operator line
        else:
            prepared_text = text
            qa_text = ""

    def _parse_turns(block: str) -> list[dict]:
        turns: list[dict] = []
        current = None
        for line in block.splitlines():
            ln = line.strip()
            if not ln:
                if current:
                    current["text"] = current["text"] + "\n"
                continue
            m = SPEAKER_LINE_RE.match(ln)
            # A speaker line is short-ish and mostly name/affiliation.
            if m and len(ln) < 160 and not ln.endswith("."):
                name = m.group("name").strip()
                affil = (m.group("affil") or "").strip()
                # Heuristic: avoid matching ordinary sentences that begin with
                # two capitalized words. Require either an affiliation OR the
                # name to be a known operator keyword.
                if not affil and name.lower() not in OPERATOR_NAMES:
                    # Still treat as continuation of previous speaker
                    if current:
                        current["text"] += " " + ln
                    continue
                if current:
                    current["text"] = current["text"].strip()
                    if current["text"]:
                        turns.append(current)
                current = {
                    "speaker_name": name,
                    "affiliation": affil,
                    "role": _classify_speaker_role(name, affil),
                    "text": "",
                }
            else:
                if current:
                    current["text"] += " " + ln if current["text"] else ln
        if current and current["text"].strip():
            turns.append(current)
        return turns

    return {
        "sections": {
            "prepared_remarks": _inherit_exec_in_prepared(_parse_turns(prepared_text)),
            "qa": _parse_turns(qa_text),
        },
        "has_qa_section": bool(qa_text),
    }
