"""
Canonical 7-fold expanding walk-forward splits (E10 / model plan).

Each tuple: (fold_id, train_end, test_start, test_end) as calendar strings.
Train: all rows with date <= train_end (last training observation).
Test: test_start … test_end inclusive — test_start is the first test day *after*
the 5-trading-day boundary gap from train_end (gap is baked into test_start).

Do not apply an additional 5-row iloc gap on the test block when using these
definitions — that would double-count the gap.
"""

from __future__ import annotations

from typing import List, Tuple

# fmt: off
E10_FOLDS: List[Tuple[str, str, str, str]] = [
    ("v1", "2022-12-30", "2023-01-10", "2023-06-30"),
    ("v2", "2023-06-30", "2023-07-11", "2023-12-29"),
    ("v3", "2023-12-29", "2024-01-09", "2024-06-28"),
    ("v4", "2024-06-28", "2024-07-09", "2024-12-31"),
    ("v5", "2024-12-31", "2025-01-09", "2025-06-30"),
    ("v6", "2025-06-30", "2025-07-09", "2025-12-31"),
    ("v7", "2025-12-31", "2026-01-09", "2026-04-09"),
]
# fmt: on

# Legacy notebook style: month-end train cutoff + calendar test start + 5-row gap slice.
# Kept for comparison only — numerically equivalent mean DirAcc on current data vs E10_FOLDS.
LEGACY_FOLDS: List[Tuple[str, str, str, str]] = [
    ("v1", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("v2", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("v3", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("v4", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("v5", "2024-12-31", "2025-01-01", "2025-06-30"),
    ("v6", "2025-06-30", "2025-07-01", "2025-12-31"),
    ("v7", "2025-12-31", "2026-01-01", "2026-04-30"),
]

BOUNDARY_GAP_ROWS = 5   # only used with LEGACY_FOLDS + gap slice
STRIDE_EVAL = 5
