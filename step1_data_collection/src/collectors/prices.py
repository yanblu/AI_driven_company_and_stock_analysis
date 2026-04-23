"""Collect daily OHLCV for TD and context tickers via yfinance.

Outputs one Parquet per ticker in data/raw/prices/ and logs each artifact to
data/manifest.csv. Re-running overwrites files in place (idempotent).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.config import (  # noqa: E402
    ALL_TICKERS,
    PRICES_DIR,
    WINDOW_END,
    WINDOW_START,
)
from src.utils.manifest import record_artifact  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prices")


def _download_one(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV for a single ticker over the configured window.

    Returns a DataFrame indexed by date with standard columns.
    `auto_adjust=False` keeps both raw Close and Adj Close — we need raw for
    benchmarking/returns and Adj Close for total-return calculations downstream.
    """
    df = yf.download(
        ticker,
        start=WINDOW_START.isoformat(),
        end=WINDOW_END.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    # yfinance sometimes returns a MultiIndex on columns when a single ticker
    # is requested (newer versions). Flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = ticker
    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    return df[cols]


def collect(tickers: list[str] | None = None) -> None:
    tickers = tickers or ALL_TICKERS
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    for t in tickers:
        safe = t.replace("^", "").replace("=", "_").replace(".", "_")
        out = PRICES_DIR / f"{safe}.parquet"
        try:
            df = _download_one(t)
        except Exception as exc:
            log.error("Failed to download %s: %s", t, exc)
            continue
        df.to_parquet(out, index=False)
        log.info("Saved %s rows=%d -> %s", t, len(df), out.name)
        record_artifact(
            source="yfinance",
            artifact_type="prices",
            identifier=t,
            path=out,
            url=f"https://finance.yahoo.com/quote/{t}/history",
            record_count=len(df),
            notes=f"daily OHLCV {WINDOW_START.isoformat()}..{WINDOW_END.isoformat()}",
        )


if __name__ == "__main__":
    collect()
