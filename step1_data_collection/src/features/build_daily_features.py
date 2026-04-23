"""
Build the daily-grain model feature matrix.

Output: data/processed/features/model_features_daily.parquet

Row grain  : one trading day d (TD.TO trading calendar)
Target     : td_return_5d_fwd = (adj_close[d+5] - adj_close[d]) / adj_close[d]
             adj_close is used (not raw close) to strip dividend drops and splits,
             so the return series reflects pure market-driven price movement.
Feature groups:
    1. Price / momentum  — TD, XFN, GSPTSE
    2. FX                — USD/CAD
    3. Rates / yields    — BoC overnight, GoC 2/5/10y, yield curve slope
    4. CPI               — monthly, forward-filled to daily
    5. NLP               — quarterly nlp_features, activated on earnings call date,
                           projected via level / delta / surprise transforms
    6. News              — rolling 7d / 30d sentiment and volume aggregates
    7. Event flags       — is_earnings_week, is_news_burst

Point-in-time rule for NLP:
    NLP features for quarter Q become known on the earnings call date (not quarter end).
    All disclosures (transcript, quarterly report, 40-F) are treated as published on
    the call date — the earliest realistic public availability date.
    Days before the first call (2021-02-25) are dropped.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
RAW_PRICES = ROOT / "data/raw/prices"
RAW_MACRO = ROOT / "data/raw/macro"
NLP_PATH = ROOT / "data/processed/features/nlp_features.parquet"
NEWS_PATH = ROOT / "data/processed/llm_annotations/news.jsonl"
OUT_PATH = ROOT / "data/processed/features/model_features_daily.parquet"

# ---------------------------------------------------------------------------
# Earnings call dates (point-in-time activation for NLP features)
# Derived from transcript data; call date = publication date for all disclosures.
# ---------------------------------------------------------------------------
CALL_DATES = {
    "FY2021Q1": "2021-02-25",
    "FY2021Q2": "2021-05-27",
    "FY2021Q3": "2021-08-26",
    "FY2021Q4": "2021-12-02",
    "FY2022Q1": "2022-03-03",
    "FY2022Q2": "2022-05-26",
    "FY2022Q3": "2022-08-25",
    "FY2022Q4": "2022-12-01",
    "FY2023Q1": "2023-03-02",
    "FY2023Q2": "2023-05-26",
    "FY2023Q3": "2023-08-24",
    "FY2023Q4": "2023-11-30",
    "FY2024Q1": "2024-02-29",
    "FY2024Q2": "2024-05-23",
    "FY2024Q3": "2024-08-22",
    "FY2024Q4": "2024-12-05",
    "FY2025Q1": "2025-02-27",
    "FY2025Q2": "2025-05-22",
    "FY2025Q3": "2025-08-28",
    "FY2025Q4": "2025-12-04",
    "FY2026Q1": "2026-04-03",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_price(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_PRICES / f"{ticker}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _load_macro(name: str) -> pd.Series:
    df = pd.read_parquet(RAW_MACRO / f"{name}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")["value"]
    return df


def _rolling_ols_features(series: pd.Series, window: int, prefix: str) -> pd.DataFrame:
    """
    Rolling OLS of price (normalised by first-bar value) on time index [0..window-1].
    Returns three series: beta (slope), rsqr (R²), resi (last-bar residual).
    Normalising by first bar makes beta a %/day figure regardless of price level.
    """
    vals = series.values
    n = len(vals)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()

    beta_arr = np.full(n, np.nan)
    rsqr_arr = np.full(n, np.nan)
    resi_arr = np.full(n, np.nan)

    for i in range(window - 1, n):
        y_raw = vals[i - window + 1: i + 1]
        y0 = y_raw[0]
        if y0 == 0 or np.isnan(y0):
            continue
        y = y_raw / y0  # scale-invariant: window starts at 1.0
        y_mean = y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        b = ss_xy / ss_xx
        beta_arr[i] = b
        y_pred = b * (x - x_mean) + y_mean
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        rsqr_arr[i] = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        resi_arr[i] = y[-1] - y_pred[-1]

    return pd.DataFrame(
        {
            f"{prefix}_beta_{window}d": beta_arr,
            f"{prefix}_rsqr_{window}d": rsqr_arr,
            f"{prefix}_resi_{window}d": resi_arr,
        },
        index=series.index,
    )


# ---------------------------------------------------------------------------
# 1. Build daily price spine from TD trading calendar
# ---------------------------------------------------------------------------
def build_price_features() -> pd.DataFrame:
    td_raw = _load_price("TD_TO")
    # Adjust high/low by the same split/dividend factor as adj_close so that
    # rolling Min/Max comparisons are consistent across ex-dividend dates.
    adj_factor = (td_raw["adj_close"] / td_raw["close"].replace(0, np.nan)).fillna(1.0)
    td = pd.DataFrame({
        "date"     : td_raw["date"],
        "td_close" : td_raw["adj_close"],
        "td_high"  : td_raw["high"] * adj_factor,
        "td_low"   : td_raw["low"]  * adj_factor,
        "td_volume": td_raw["volume"],
    })
    xfn = _load_price("XFN_TO")[["date", "adj_close"]].rename(columns={"adj_close": "xfn_close"})
    tsx = _load_price("GSPTSE")[["date", "adj_close"]].rename(columns={"adj_close": "tsx_close"})

    df = td.merge(xfn, on="date", how="left").merge(tsx, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    for col, src in [("td", "td_close"), ("xfn", "xfn_close"), ("tsx", "tsx_close")]:
        df[f"{col}_return_1d"] = df[src].pct_change(1)
        df[f"{col}_return_5d"] = df[src].pct_change(5)
        df[f"{col}_return_20d"] = df[src].pct_change(20)

    df["td_volatility_20d"] = df["td_return_1d"].rolling(20).std()

    # Volume features — parallel structure to price return features
    td_vol = df["td_volume"]
    tsx_vol = _load_price("GSPTSE")[["date", "volume"]].rename(columns={"volume": "tsx_volume"})
    xfn_vol = _load_price("XFN_TO")[["date", "volume"]].rename(columns={"volume": "xfn_volume"})
    df = df.merge(tsx_vol, on="date", how="left")
    df = df.merge(xfn_vol, on="date", how="left")

    vol_mean_20     = td_vol.rolling(20).mean()
    vol_std_20      = td_vol.rolling(20).std()
    tsx_vol_mean_20 = df["tsx_volume"].rolling(20).mean()
    tsx_vol_std_20  = df["tsx_volume"].rolling(20).std()
    xfn_vol_mean_20 = df["xfn_volume"].rolling(20).mean()
    xfn_vol_std_20  = df["xfn_volume"].rolling(20).std()

    df["td_volume_zscore_20d"]    = (td_vol - vol_mean_20) / vol_std_20.replace(0, np.nan)
    df["td_volume_change_1d"]     = td_vol.pct_change(1)
    df["td_volume_change_5d"]     = td_vol.rolling(5).mean().pct_change(5)   # 5-day avg vs prior 5-day avg
    df["td_volume_change_20d"]    = td_vol.rolling(20).mean().pct_change(20).fillna(0) # 20-day avg vs prior 20-day avg; 0 at data boundary
    df["tsx_volume_zscore_20d"]   = (df["tsx_volume"] - tsx_vol_mean_20) / tsx_vol_std_20.replace(0, np.nan)
    df["xfn_volume_zscore_20d"]   = (df["xfn_volume"] - xfn_vol_mean_20) / xfn_vol_std_20.replace(0, np.nan)

    # ── Qlib Tier 1: CORR and WVMA — computed before volume drop ────────────
    # Price–Volume Correlation (CORR): rolling corr(1d return, 1d volume change).
    # Divergence (price up, volume down) is a fragility signal.
    df["td_corr_pv_5d"]  = df["td_return_1d"].rolling(5).corr(df["td_volume_change_1d"])
    df["td_corr_pv_20d"] = df["td_return_1d"].rolling(20).corr(df["td_volume_change_1d"])

    # Volume-Weighted Return Volatility (WVMA): Std(|ret|×vol_norm) / (Mean(|ret|×vol_norm) + ε).
    # Coefficient of variation → scale-invariant across price levels (matches Qlib Alpha158).
    # Spikes when large moves coincide with heavy participation — "active uncertainty".
    td_vol_norm = (td_vol / vol_mean_20.replace(0, np.nan)).fillna(1.0)
    active_ret  = df["td_return_1d"].abs() * td_vol_norm
    for _w in [5, 20]:
        _std  = active_ret.rolling(_w).std()
        _mean = active_ret.rolling(_w).mean()
        df[f"td_wvma_{_w}d"] = (_std / (_mean + 1e-12)).fillna(0)

    # Drop raw volume columns (not needed as model features)
    df = df.drop(columns=["td_volume", "tsx_volume", "xfn_volume"])

    df["td_vs_xfn_5d"] = df["td_return_5d"] - df["xfn_return_5d"]
    df["td_vs_tsx_5d"] = df["td_return_5d"] - df["tsx_return_5d"]

    # ── Qlib Tier 1: CNTD, SUMD, RSV ────────────────────────────────────────
    # Directional Count (CNTD): Mean(ret>0) - Mean(ret<0) → bounded [-1, +1].
    # Using mean (fraction of days) not raw count — window-size agnostic (matches Qlib Alpha158).
    up_days  = (df["td_return_1d"] > 0).astype(float)
    dn_days  = (df["td_return_1d"] < 0).astype(float)
    df["td_cntd_5d"]  = up_days.rolling(5).mean()  - dn_days.rolling(5).mean()
    df["td_cntd_20d"] = up_days.rolling(20).mean() - dn_days.rolling(20).mean()

    # Directional Return Mass (SUMD): (Σ(ret>0) - Σ|ret<0|) / (Σ|ret| + ε) → bounded [-1, +1].
    # Normalised by total absolute movement — comparable across high/low volatility regimes.
    # Distinguishes a +3% week from 5 small gains vs. 1 big spike (matches Qlib Alpha158).
    pos_ret = df["td_return_1d"].clip(lower=0)
    neg_ret = (-df["td_return_1d"]).clip(lower=0)
    abs_ret = df["td_return_1d"].abs()
    for _w in [5, 20]:
        _denom = abs_ret.rolling(_w).sum() + 1e-12
        df[f"td_sumd_{_w}d"] = (pos_ret.rolling(_w).sum() - neg_ret.rolling(_w).sum()) / _denom

    # Stochastic Position (RSV): (close - Min(low, d)) / (Max(high, d) - Min(low, d) + ε).
    # Uses true intraday high/low range — matches Qlib Alpha158 exact formula.
    # Captures whether close settled near the top or bottom of the full day's range,
    # including intraday wicks that close-only RSV misses.
    for w in [5, 14, 20]:
        lo  = df["td_low"].rolling(w).min()
        hi  = df["td_high"].rolling(w).max()
        rng = (hi - lo).replace(0, np.nan)
        df[f"td_rsv_{w}d"] = ((df["td_close"] - lo) / (rng + 1e-12)).fillna(0.5)

    # Mean-reversion features
    # RSI(14): 100 - 100/(1+RS) where RS = avg_gain/avg_loss over 14 days
    delta_1d = df["td_close"].diff(1)
    gain = delta_1d.clip(lower=0).rolling(14).mean()
    loss = (-delta_1d.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["td_rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)  # 50 = neutral when undefined

    # Bollinger Band position: (price - 20d mean) / (2 * 20d std), range ≈ [-1, +1]
    bb_mean = df["td_close"].rolling(20).mean()
    bb_std  = df["td_close"].rolling(20).std()
    df["td_bb_position"] = ((df["td_close"] - bb_mean) / (2 * bb_std.replace(0, np.nan))).fillna(0)

    # Distance from 52-week high: (close - 252d rolling max) / 252d rolling max, always <= 0
    high_252 = df["td_close"].rolling(252).max()
    df["td_dist_52w_high"] = ((df["td_close"] - high_252) / high_252.replace(0, np.nan)).fillna(0)

    # ── Qlib Tier 1: Trend Quality (BETA / RSQR / RESI) ─────────────────────
    # Rolling OLS of normalised price on time index.
    # BETA: trend slope (%/day); RSQR: trend cleanliness (0=choppy, 1=linear);
    # RESI: last-bar deviation from trend line (mean-reversion signal).
    for w in [5, 20, 60]:
        ols = _rolling_ols_features(df["td_close"], w, "td")
        for col in ols.columns:
            df[col] = ols[col].values

    # 5-day forward return — the prediction target
    df["td_return_5d_fwd"] = df["td_close"].pct_change(5).shift(-5)

    return df


# ---------------------------------------------------------------------------
# 2. FX features
# ---------------------------------------------------------------------------
def build_fx_features(spine_dates: pd.Series) -> pd.DataFrame:
    fx = _load_macro("fx_usdcad").reindex(spine_dates).ffill()
    fx_1d = fx.pct_change(1)
    fx_5d = fx.pct_change(5)
    fx_vol = fx_1d.rolling(20).std()

    return pd.DataFrame(
        {
            "date": spine_dates,
            "fx_usdcad_level": fx.values,
            "fx_usdcad_return_5d": fx_5d.values,
            "fx_usdcad_volatility_20d": fx_vol.values,
        }
    )


# ---------------------------------------------------------------------------
# 3. Rates / yield curve features
# ---------------------------------------------------------------------------
def build_rate_features(spine_dates: pd.Series) -> pd.DataFrame:
    overnight = _load_macro("overnight_avg").reindex(spine_dates).ffill()
    y2 = _load_macro("goc_2y").reindex(spine_dates).ffill()
    y5 = _load_macro("goc_5y").reindex(spine_dates).ffill()
    y10 = _load_macro("goc_10y").reindex(spine_dates).ffill()

    slope = y10 - y2

    out = pd.DataFrame({"date": spine_dates})
    out["rate_overnight_level"] = overnight.values
    out["rate_overnight_change_5d"] = overnight.diff(5).values
    out["yield_2y_level"] = y2.values
    out["yield_5y_level"] = y5.values
    out["yield_10y_level"] = y10.values
    out["yield_curve_slope"] = slope.values
    out["yield_curve_slope_change_5d"] = slope.diff(5).values
    return out


# ---------------------------------------------------------------------------
# 4. CPI features (monthly → forward-fill to daily)
# ---------------------------------------------------------------------------
def build_cpi_features(spine_dates: pd.Series) -> pd.DataFrame:
    # Load raw monthly CPI (now extended back to 2020-01-01)
    raw = _load_macro("cpi_all_items")  # monthly, date-indexed
    # Compute YoY at monthly grain before reindexing to daily
    raw_df = raw.reset_index()
    raw_df.columns = ["date", "cpi"]
    raw_df = raw_df.sort_values("date").reset_index(drop=True)
    raw_df["cpi_yoy"] = raw_df["cpi"].pct_change(12)  # exact 12-month monthly shift

    # Reindex to daily spine via forward-fill
    monthly_idx = pd.DatetimeIndex(raw_df["date"])
    cpi_s   = raw_df.set_index("date")["cpi"]
    yoy_s   = raw_df.set_index("date")["cpi_yoy"]

    cpi_daily = cpi_s.reindex(spine_dates).ffill()
    yoy_daily = yoy_s.reindex(spine_dates).ffill()

    return pd.DataFrame(
        {
            "date": spine_dates,
            "cpi_level": cpi_daily.values,
            "cpi_yoy_change": yoy_daily.values,
        }
    )


# ---------------------------------------------------------------------------
# 5. NLP features — point-in-time projection to daily
# ---------------------------------------------------------------------------
def build_nlp_features(spine_dates: pd.Series) -> pd.DataFrame:
    nlp = pd.read_parquet(NLP_PATH)

    # Build call-date → NLP row mapping
    call_map = {fq: pd.Timestamp(dt) for fq, dt in CALL_DATES.items()}
    nlp = nlp.copy()
    nlp["call_date"] = nlp["fiscal_quarter"].map(call_map)
    nlp = nlp.dropna(subset=["call_date"]).sort_values("call_date").reset_index(drop=True)

    # Columns to engineer (exclude metadata and the two sparse individual report columns).
    # report_sentiment_mean already combines them: quarterly report for Q1/Q2/Q3,
    # 40-F for Q4 — no gaps. Use that combined column; drop the sparse sub-columns.
    exclude = {
        "fiscal_quarter", "n_chunks_annotated", "call_date",
        "report_40f_sentiment_mean",       # Q4-only (5 values) — captured in report_sentiment_mean
        "report_quarterly_sentiment_mean", # Q1-Q3 only (16 values) — captured in report_sentiment_mean
    }
    nlp_cols = [c for c in nlp.columns if c not in exclude]

    # Add derived framing gap column
    if (
        "transcript_ceo_prep_sentiment_mean" in nlp.columns
        and "report_sentiment_mean" in nlp.columns
    ):
        nlp["framing_gap"] = (
            nlp["transcript_ceo_prep_sentiment_mean"] - nlp["report_sentiment_mean"]
        )
        nlp_cols.append("framing_gap")

    # Compute delta and surprise (trailing 4-quarter mean) at quarterly level
    for col in nlp_cols:
        nlp[f"{col}_delta"] = nlp[col].diff(1)
        nlp[f"{col}_surprise"] = nlp[col] - nlp[col].shift(1).rolling(4, min_periods=1).mean()

    # Build a daily series by forward-filling from each call date
    # Create a date-indexed frame of quarterly NLP values
    nlp_daily_cols = (
        nlp_cols
        + [f"{c}_delta" for c in nlp_cols]
        + [f"{c}_surprise" for c in nlp_cols]
    )

    # Reindex to spine dates using call_date as the activation point
    nlp_indexed = nlp.set_index("call_date")[nlp_daily_cols]

    # Reindex to spine, forward-fill from activation date
    spine_index = pd.DatetimeIndex(spine_dates)
    nlp_ff = nlp_indexed.reindex(spine_index.union(nlp_indexed.index)).sort_index().ffill()
    nlp_ff = nlp_ff.reindex(spine_index)

    # Rename all NLP columns to avoid clash: suffix _ffill on levels
    rename = {c: f"{c}_ffill" for c in nlp_cols}
    nlp_ff = nlp_ff.rename(columns=rename)

    # days_since_call
    call_dates_sorted = sorted(nlp["call_date"].tolist())

    def days_since(d: pd.Timestamp) -> int:
        past = [c for c in call_dates_sorted if c <= d]
        return (d - past[-1]).days if past else np.nan

    nlp_ff["days_since_call"] = [days_since(d) for d in spine_index]
    nlp_ff["is_earnings_week"] = (nlp_ff["days_since_call"] <= 5).astype(int)
    nlp_ff["date"] = spine_index

    return nlp_ff.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. News rolling features
# ---------------------------------------------------------------------------
def build_news_features(spine_dates: pd.Series) -> pd.DataFrame:
    rows = []
    with open(NEWS_PATH) as f:
        for line in f:
            r = json.loads(line)
            score = r.get("output", {}).get("sentiment_score")
            date = r.get("date")
            if score is not None and date is not None:
                rows.append({"date": pd.Timestamp(date), "sentiment_score": float(score)})

    news = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    records = []
    for d in spine_dates:
        d = pd.Timestamp(d)
        window_7 = news[(news["date"] >= d - pd.Timedelta(days=7)) & (news["date"] <= d)]
        window_30 = news[(news["date"] >= d - pd.Timedelta(days=30)) & (news["date"] <= d)]
        past_news = news[news["date"] <= d]
        days_since = (d - past_news["date"].max()).days if len(past_news) > 0 else np.nan

        records.append(
            {
                "date": d,
                # 0 when no articles published in window — no news = no sentiment signal,
                # not a missing value to be imputed from other days.
                "news_sent_mean_7d": window_7["sentiment_score"].mean() if len(window_7) else 0.0,
                "news_sent_mean_30d": window_30["sentiment_score"].mean() if len(window_30) else 0.0,
                "news_count_7d": len(window_7),
                "news_count_30d": len(window_30),
                "days_since_last_news": days_since,
            }
        )

    df = pd.DataFrame(records)
    df["is_news_burst"] = (df["news_count_7d"] > 10).astype(int)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# 6. Macro fear / uncertainty features (VIX, Gold, DXY)
# ---------------------------------------------------------------------------
def build_macro_fear_features(spine_dates: pd.Series) -> pd.DataFrame:
    rows = {"date": spine_dates}

    for name in ["vix", "gold", "dxy"]:
        raw = pd.read_parquet(RAW_MACRO / f"{name}.parquet")
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date")[name]
        s = raw.reindex(spine_dates).ffill()

        rows[f"{name}_level"]       = s.values
        rows[f"{name}_return_5d"]   = s.pct_change(5).values
        rows[f"{name}_volatility_20d"] = s.pct_change(1).rolling(20).std().values

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def build_features() -> pd.DataFrame:
    print("Building price features...")
    price = build_price_features()
    spine_dates = price["date"]

    print("Building FX features...")
    fx = build_fx_features(spine_dates)

    print("Building rate/yield features...")
    rates = build_rate_features(spine_dates)

    print("Building CPI features...")
    cpi = build_cpi_features(spine_dates)

    print("Building NLP features...")
    nlp = build_nlp_features(spine_dates)

    print("Building news rolling features...")
    news = build_news_features(spine_dates)

    print("Building macro fear features (VIX, Gold, DXY)...")
    fear = build_macro_fear_features(spine_dates)

    print("Joining all features...")
    df = price.copy()
    for part in [fx, rates, cpi, nlp, news, fear]:
        part["date"] = pd.to_datetime(part["date"])
        df = df.merge(part, on="date", how="left")

    # Drop rows before first NLP call date (2021-02-25 = FY2021Q1) and last 5 (no target).
    first_call = pd.Timestamp("2021-02-25")
    df = df[df["date"] >= first_call].copy()
    df = df.dropna(subset=["td_return_5d_fwd"])

    # Fill delta and surprise NaN (only present in FY2021Q1 window) with a sentinel value.
    # -9999 is far outside the natural feature range (sentiment: 0–1, shares: 0–1), so
    # tree models learn a dedicated split for "this feature was unavailable" rather than
    # treating it as a real value. This is more honest than 0 (which would imply "no change").
    MISSING_SENTINEL = -9999.0
    delta_surprise_cols = [c for c in df.columns if c.endswith(("_delta", "_surprise"))]
    df[delta_surprise_cols] = df[delta_surprise_cols].fillna(MISSING_SENTINEL)

    # OLS trend features have NaN during the warmup window (first 60 rows).
    # Fill with 0: neutral slope, no trend quality, no residual — honest "no info".
    ols_cols = [c for c in df.columns if any(c.endswith(s) for s in ("_beta_5d", "_beta_20d", "_beta_60d", "_rsqr_5d", "_rsqr_20d", "_rsqr_60d", "_resi_5d", "_resi_20d", "_resi_60d"))]
    df[ols_cols] = df[ols_cols].fillna(0)

    # Drop raw price columns used only for intermediate calculations
    drop_cols = ["td_close", "td_high", "td_low", "td_volume", "xfn_close", "tsx_close"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"Final dataset: {len(df)} rows × {df.shape[1]} columns")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Target range: {df['td_return_5d_fwd'].min():.3f} → {df['td_return_5d_fwd'].max():.3f}")
    missing = df.isnull().sum()
    top_missing = missing[missing > 0].sort_values(ascending=False).head(10)
    if len(top_missing):
        print(f"\nTop missing value columns:\n{top_missing}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    return df


if __name__ == "__main__":
    build_features()
