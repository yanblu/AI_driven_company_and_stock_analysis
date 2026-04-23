"""Collect macroeconomic series from Bank of Canada Valet.

Valet is free, no API key required, and returns observations as JSON.
Docs: https://www.bankofcanada.ca/valet/docs

We focus on series that matter for a Canadian bank:
- overnight policy rate (NIM driver)
- Government of Canada benchmark yields (2y/5y/10y -> yield curve)
- USD/CAD FX rate (TD has large US ops, affects reported earnings)
- CPI all-items (inflation regime, drives BoC reaction function)

StatCan CPI is pulled via Valet's mirrored CPI series so we avoid a second
API client for the MVP.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.config import MACRO_DIR, WINDOW_END, WINDOW_START  # noqa: E402
from src.utils.manifest import record_artifact  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("macro")

VALET_BASE = "https://www.bankofcanada.ca/valet/observations"

# Curated series. Names are the BoC Valet "series name" or "V-number".
# Each entry: (series_name, friendly_label, description, human_notes)
SERIES: list[tuple[str, str, str]] = [
    # Policy / overnight rates
    ("V39079", "policy_target_rate", "Target for the overnight rate (policy rate)"),
    ("AVG.INTWO", "overnight_avg", "Average overnight rate (market)"),
    # Government of Canada benchmark yields
    ("BD.CDN.2YR.DQ.YLD", "goc_2y", "GoC benchmark bond yield, 2-year"),
    ("BD.CDN.5YR.DQ.YLD", "goc_5y", "GoC benchmark bond yield, 5-year"),
    ("BD.CDN.10YR.DQ.YLD", "goc_10y", "GoC benchmark bond yield, 10-year"),
    # FX
    ("FXUSDCAD", "fx_usdcad", "USD to CAD noon/indicative rate"),
    # CPI (monthly, StatCan mirrored by BoC Valet)
    ("V41690973", "cpi_all_items", "Consumer Price Index (CPI), all-items, monthly index"),
]


def _fetch_valet(series: str) -> pd.DataFrame:
    """Fetch a single Valet series over the configured window as a DataFrame."""
    url = f"{VALET_BASE}/{series}/json"
    params = {
        "start_date": WINDOW_START.isoformat(),
        "end_date": WINDOW_END.isoformat(),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    obs = payload.get("observations", [])
    if not obs:
        raise RuntimeError(f"empty observations for {series}")

    rows = []
    for o in obs:
        d = o.get("d")
        if not d or series not in o:
            continue
        v = o[series].get("v")
        if v in (None, ""):
            continue
        try:
            rows.append({"date": pd.to_datetime(d).date(), "value": float(v)})
        except (TypeError, ValueError):
            continue
    if not rows:
        raise RuntimeError(f"no usable observations for {series}")
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["series"] = series
    return df[["date", "series", "value"]]


def collect() -> None:
    MACRO_DIR.mkdir(parents=True, exist_ok=True)

    for series, label, desc in SERIES:
        safe = label.replace("/", "_")
        out = MACRO_DIR / f"{safe}.parquet"
        try:
            df = _fetch_valet(series)
        except Exception as exc:
            log.warning("Skipping %s (%s): %s", series, label, exc)
            continue
        df.to_parquet(out, index=False)
        log.info(
            "Saved %s (%s) rows=%d range=%s..%s",
            series,
            label,
            len(df),
            df["date"].min(),
            df["date"].max(),
        )
        record_artifact(
            source="boc_valet",
            artifact_type="macro",
            identifier=label,
            path=out,
            url=f"{VALET_BASE}/{series}/json",
            record_count=len(df),
            notes=desc,
        )


if __name__ == "__main__":
    collect()
